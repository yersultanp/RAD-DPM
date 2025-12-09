# main_train.py
import torch
import sys
import os
from tqdm import tqdm
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import random
sys.path.append('/content/drive/MyDrive/Adversarial-Diffusion-Distillation')
from configs.model_config import ModelConfig
from configs.train_config import TrainConfig
from configs.scheduler_config import SchedulerConfig
from models.teacher import load_teacher_model
from models.student import RecurrentScheduler
from ddim_utils import DifferentiableDPMSolverHandler
from train.train_step import train_one_step, generate_teacher_target
from losses import image_loss, HybridLatentLoss
from models.refiner import attach_refiner_lora
from eval.visualize_schedule import plot_scheduler_training_history
from eval.plot_results import comparison_pipeline

def main():
    student = RecurrentScheduler(SchedulerConfig.K_STEPS).to(TrainConfig.DEVICE)
    optimizer = torch.optim.AdamW(student.parameters(), lr=TrainConfig.LEARNING_RATE)
    pipe = load_teacher_model(TrainConfig.DEVICE)
    pipe = attach_refiner_lora(pipe)
    scaler = GradScaler()
    dpm_handler = DifferentiableDPMSolverHandler(pipe)
    loss_fn = HybridLatentLoss(alpha_mse=1.0, alpha_cos=0.1, alpha_stats=0.1).to(TrainConfig.DEVICE)

    loss_history = []
    schedule_history = []

    print("Starting Training...")
    pbar = tqdm(range(TrainConfig.EPOCHS))

    TRAIN_PROMPTS = [
    # --- Nature & Wildlife (Texture & High Frequency) ---
    "A macro photograph of dew drops on a spiderweb in the morning light",
    "A candid portrait of a grizzly bear catching a salmon in a river",
    "A majestic herd of wild horses galloping across a dusty plain",
    "A vibrant coral reef teeming with colorful tropical fish, underwater photography",
    "A close-up of a chameleon's eye, showing its intricate texture and color",
    "A lone wolf howling at a full moon on a snowy ridge",
    "A detailed photograph of a monarch butterfly resting on a purple lavender stalk",
    "A macro photograph of a bumblebee on a yellow flower",
    "A close-up portrait of a snowy owl with detailed feathers",
    "A cute corgi running through a field of tall green grass",

    # --- Landscapes (Depth & Lighting) ---
    "A dramatic coastal cliffside with crashing waves at sunset",
    "A winding path through an autumn forest with colorful falling leaves",
    "A panoramic view of a snow-capped mountain range under a clear blue sky",
    "A secluded tropical beach with turquoise water and palm trees",
    "A rolling vineyard in Tuscany bathed in golden hour light",
    "A frozen waterfall with large icicles hanging from a rock face",
    "A vast lavender field in Provence, stretching as far as the eye can see",
    "A serene misty mountain lake at sunrise, landscape photography",
    "A dusty desert road stretching into the horizon",

    # --- Urban & Architecture (Geometry & Structure) ---
    "A busy Tokyo street crossing at night, filled with pedestrians and neon signs",
    "A futuristic monorail passing through a glass tunnel in a metropolis",
    "An old, ivy-covered stone bridge crossing a quiet canal in Bruges",
    "A modern skyscraper with a vertical garden growing on its facade",
    "A bustling open-air market in Marrakech, filled with spices and textiles",
    "A retro-futuristic diner with flying cars parked outside",
    "An ancient temple hidden deep within a jungle, overgrown with vines",
    "A futuristic cyberpunk city at night with neon rain",

    # --- Objects & Food (Material & Reflection) ---
    "A rustic loaf of sourdough bread on a wooden cutting board with butter and a knife",
    "A steaming bowl of ramen with pork, egg, and nori, close-up shot",
    "A stack of fluffy pancakes topped with fresh berries and maple syrup",
    "A vintage film camera and a stack of old photographs on a desk",
    "A colorful arrangement of fresh fruits and vegetables at a farmer's market stand",
    "A detailed shot of a gourmet chocolate dessert with gold leaf garnish",
    "An antique brass telescope sitting by a window with a sea view",
    "A shiny red sports car driving on a coastal highway"
    ]

    train_data = generate_teacher_target(pipe, TRAIN_PROMPTS, TrainConfig.DEVICE)

    null_inputs = pipe.tokenizer("", return_tensors="pt", padding="max_length", truncation=True).to(TrainConfig.DEVICE)
    with torch.no_grad():
        null_emb = pipe.text_encoder(null_inputs.input_ids)[0]


    # ==========================================
    # PHASE 1: TRAIN SCHEDULER
    # ==========================================

    print("\n=== Phase 1: Training Scheduler ===")
    pipe.unet.disable_adapter_layers() # Disable LoRA
    for epoch in pbar:
        epoch_loss = 0
        current_schedule = []

        for i, data in enumerate(train_data):
            learned_schedule = []
            text_emb = data["emb"]
            student_latents = data["noise"].clone().requires_grad_(True)
            target_latents = data["target"]
            # Initialize History as None
            prev_noise = None
            prev_lambda = None

            # 1. Initial State
            # Start at t=1000 (Pure Noise)
            t_curr = torch.full((TrainConfig.BATCH_SIZE, 1), 1000.0, device=TrainConfig.DEVICE)
            hx = None # Hidden state starts empty

            optimizer.zero_grad()

            with autocast():
                # 1. Run Student Loop
                for k in range(SchedulerConfig.K_STEPS):
                    learned_schedule.append(t_curr.item())

                    if k < SchedulerConfig.K_STEPS - 1:
                        t_next, hx = student(student_latents, t_curr, hx)
                        t_next = torch.min(t_next, t_curr - 1).clamp(min=50)
                    else:
                        t_next = torch.zeros_like(t_curr)

                    # Step returns: New Latents, Current Noise, Current Lambda
                    student_latents, curr_noise, curr_lambda = dpm_handler.step(
                        student_latents, t_curr, t_next, text_emb,
                        prev_noise_pred=prev_noise, # Pass history
                        prev_lambda=prev_lambda,    # Pass history
                        guidance_scale=1.0
                    )

                    # Update History
                    prev_noise = curr_noise
                    prev_lambda = curr_lambda
                    t_curr = t_next

                # 2. Compute Loss (in fp16/fp32 automatically handled)
                loss = loss_fn(student_latents, target_latents)

            # 3. Backprop with Scaler
            scaler.scale(loss).backward()

            # Unscale grads to clip them
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            current_schedule.append(learned_schedule)

        avg_loss = epoch_loss / len(train_data)
        loss_history.append(avg_loss)
        schedule_history.append(current_schedule)
        pbar.set_description(f"Scheduler Loss: {avg_loss:.4f}")

    print("Scheduler Training Complete!")

    # ==========================================
    # PHASE 2: TRAINING REFINER (LoRA)
    # ==========================================

    print("\n=== Phase 2: Training Refiner (LoRA) ===")
    pipe.unet.enable_adapter_layers() # Enable LoRA
    pbar = tqdm(range(TrainConfig.EPOCHS))


    refiner_params = [p for p in pipe.unet.parameters() if p.requires_grad]
    refiner_optimizer = torch.optim.AdamW(refiner_params, lr=1e-4)

    # Freeze the Student Scheduler now
    student.requires_grad_(False)

    REFINER_EPOCHS = 20 # Increased slightly as LoRA needs a bit more time

    for epoch in range(REFINER_EPOCHS):
        epoch_loss = 0
        pbar_ref = tqdm(train_data, desc=f"Refiner Epoch {epoch+1}")

        for data in pbar_ref:
            text_emb = data["emb"]
            # Start from noise
            latents = data["noise"].clone()
            target_latents = data["target"]
            prev_noise = None
            prev_lambda = None
            t_curr = torch.full((TrainConfig.BATCH_SIZE, 1), 1000.0, device=TrainConfig.DEVICE)
            hx = None # Hidden state starts empty

            refiner_optimizer.zero_grad()

            with autocast():
                # A. Run Scheduler to get "Blurry Input" (No Grad on Scheduler)
                # We purposefully disable LoRA for the first K-1 steps to simulate
                # the "coarse" generation.
                pipe.unet.disable_adapter_layers()

                with torch.no_grad():
                    for k in range(SchedulerConfig.K_STEPS - 1): # Run all but last
                        t_next, hx = student(latents, t_curr, hx)
                        # Clamp logic
                        t_next = torch.min(t_next, t_curr - 1).clamp(min=50)
                        # Step returns: New Latents, Current Noise, Current Lambda
                        latents, curr_noise, curr_lambda = dpm_handler.step(
                            latents, t_curr, t_next, text_emb,
                            prev_noise_pred=prev_noise, # Pass history
                            prev_lambda=prev_lambda,    # Pass history
                            guidance_scale=1.0
                        )

                # B. The Final Step (Refinement)
                # Now we ENABLE LoRA. This is the only step we train.
                pipe.unet.enable_adapter_layers()

                # Get the final timestep (usually jump to 0)
                t_curr, hx = student(latents, t_curr, hx)
                t_next = torch.zeros_like(t_curr)

                # FIX: Blind Training. Use null_emb instead of text_emb.
                # This forces LoRA to look at pixels, avoiding prompt overfitting.
                # We use the diff_handler to execute the step WITH gradients.
                refined_latents, curr_noise, curr_lambda = dpm_handler.step(
                    latents, t_curr, t_next, null_emb,
                    prev_noise_pred=prev_noise, # Pass history
                    prev_lambda=prev_lambda,    # Pass history
                    guidance_scale=1.0
                )

                # Compute loss
                loss = loss_fn(refined_latents, target_latents)

            scaler.scale(loss).backward()
            scaler.unscale_(refiner_optimizer)
            torch.nn.utils.clip_grad_norm_(refiner_params, 1.0)
            scaler.step(refiner_optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar_ref.set_description(f"Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_data)
        print(f"Refiner Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

    print("Refiner Training Complete!")
    # ==========================================
    # EVALUATION
    # ==========================================
    # Visualize Scheduling Results
    if not os.path.exists(f"./final_results_{SchedulerConfig.K_STEPS}_STEPS"):
        os.makedirs(f"./final_results_{SchedulerConfig.K_STEPS}_STEPS")

    # we can sample a few data points to visualize
    # current schedule_history is a list of lists of lists (n epochs x n_data x K_STEPS)
    k = 1
    k_idx = random.sample([i for i in range(len(TRAIN_PROMPTS))], k)
    for i in k_idx:
        schedule_history_sample = [epoch_schedule[i] for epoch_schedule in schedule_history]
        plot_scheduler_training_history(loss_history, schedule_history_sample, i, save_dir=f"./final_results_{SchedulerConfig.K_STEPS}_STEPS")

    EVAL_PROMPTS = [
    # --- Category 1: Texture Stress Test (High Frequency) ---
    # Tests if the scheduler skips the critical end-steps (t < 100) needed for fine detail.
    "A close-up texture shot of a knitted wool sweater, high detail",
    "A macro shot of moss growing on a wet rock, detailed greenery",
    "A weathered wooden plank with peeling blue paint and rust",

    # --- Category 2: Lighting & Dynamic Range (Contrast) ---
    # Tests if the scheduler can resolve values early. Bad schedules make these gray/washed out.
    "A bioluminescent mushroom forest glowing in the dark",
    "A lighthouse beam cutting through thick fog at night",
    "A dramatic candlelight portrait of a woman, chiaroscuro style",

    # --- Category 3: Geometry & Structure (Low Frequency / Edges) ---
    # Tests if the scheduler commits to shapes early (t > 500).
    # Bad schedules make these lines wobbly or disconnected.
    "A perfectly folded origami crane sitting on a black table",
    "A complex circuit board with gold traces and chips",
    "A spiral staircase looking down from the top, fibonacci composition",

    # --- Category 4: Compositional Complexity (Object Separation) ---
    # Tests if the scheduler can separate distinct concepts without blending them.
    "A game of chess in progress, focus on the black king",
    "A vintage tea set arranged on a checkered picnic blanket",
    "A stack of balanced zen stones on a riverbank",

    # --- Category 5: Stylistic & Distribution Shift ---
    # Tests if the scheduler generalizes to non-photorealistic noise distributions.
    "A colorful stained glass window depicting a dragon",
    "A black and white ink sketch of a mountain range, minimalism",
    "A pixel art landscape of a sunset over a desert",
    ]
    mean_schedule = analyze_schedule_variance(student, TRAIN_PROMPTS + EVAL_PROMPTS)
    print(f"Optimal Hardcoded Schedule: {mean_schedule}")
    # Compare Methods Pipeline
    comparison_pipeline(pipe, dpm_handler, student, EVAL_PROMPTS, K_STEPS=SchedulerConfig.K_STEPS, 
                        DEVICE=TrainConfig.DEVICE, save_dir=f"./final_results_{SchedulerConfig.K_STEPS}_STEPS/full_comparison_grid.png")

if __name__ == "__main__":
    main()
