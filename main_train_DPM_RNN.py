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
from eval.plot_results import compare_methods_pipeline

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
        "A macro photograph of a bumblebee on a yellow flower",
        "A close-up portrait of a snowy owl with detailed feathers",
        "A cute corgi running through a field of tall green grass",
        "A futuristic cyberpunk city at night with neon rain",
        "A serene misty mountain lake at sunrise, landscape photography",
        "A dusty desert road stretching into the horizon",
        "A delicious pepperoni pizza with melting cheese",
        "A vintage typewriter sitting on an old wooden desk",
    ]
    #     "A shiny red sports car driving on a coastal highway",
    #     "A portrait of an old fisherman with a weathered face",
    #     "A professional headshot of a smiling woman in business attire",
    #     "A renaissance style oil painting of a young princess",
    #     "A cyberpunk android girl with glowing circuitry on her face",
    #     "A flat vector illustration of a rocket ship launching",
    #     "A watercolor painting of a cozy cafe in Paris",
    # ]

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
                        t_next = torch.min(t_next, t_curr - 1).clamp(min=0)
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
                        t_next = torch.min(t_next, t_curr - 1).clamp(min=0)
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
                loss = F.mse_loss(refined_latents, target_latents)

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
    if not os.path.exists("./final_results"):
        os.makedirs("./final_results")

    # we can sample a few data points to visualize
    # current schedule_history is a list of lists of lists (n epochs x n_data x K_STEPS)
    k = 4
    k_idx = random.sample([i for i in range(len(TRAIN_PROMPTS))], k)
    for i in k_idx:
        schedule_history_sample = [epoch_schedule[i] for epoch_schedule in schedule_history]
        plot_scheduler_training_history(loss_history, schedule_history_sample, SchedulerConfig.K_STEPS, save_dir="./final_results")

    EVAL_PROMPTS = [
        "A majestic lion roaring in the savannah at sunset",
        "A beautiful waterfall in a lush green forest",
        "A sleek black panther prowling through the jungle",
        "A colorful hot air balloon floating over a scenic landscape",
        "A futuristic city skyline with flying cars",
        "A close-up of a vibrant red rose with dew drops",
        "A snowy mountain peak under a starry night sky",
        "A delicious bowl of ramen with various toppings",
    ]
    # Compare Methods Pipeline
    compare_methods_pipeline(pipe, dpm_handler, student, EVAL_PROMPTS, K_STEPS=SchedulerConfig.K_STEPS, DEVICE=TrainConfig.DEVICE, save_dir="./final_results/full_comparison_grid.png")
