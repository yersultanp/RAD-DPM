# main_train.py
import torch
import sys
import random
sys.path.append('/content/drive/MyDrive/Adversarial-Diffusion-Distillation')
from configs.model_config import ModelConfig
from configs.train_config import TrainConfig
from configs.scheduler_config import SchedulerConfig
from models.teacher import load_teacher_model
from models.student import RobustLearnedScheduler
from ddim_utils import DifferentiableDiffusionHandler
from train.train_step import train_one_step, generate_teacher_target
import torch.nn.functional as F
from losses import image_loss
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from models.refiner import attach_refiner_lora
from eval.visualize_schedule import visualize_scheduling_results
from eval.evaluate import evaluation_pipeline
from losses import HybridLatentLoss

def main():
    student = RobustLearnedScheduler(SchedulerConfig.K_STEPS).to(TrainConfig.DEVICE)
    optimizer = torch.optim.AdamW(student.parameters(), lr=TrainConfig.LEARNING_RATE)
    pipe = load_teacher_model(TrainConfig.DEVICE)
    pipe = attach_refiner_lora(pipe)
    scaler = GradScaler() # Vital for float16 training stability
    diff_handler = DifferentiableDiffusionHandler(pipe)
    loss_fn = HybridLatentLoss(alpha_mse=1.0, alpha_cos=0.1, alpha_stats=0.1).to(TrainConfig.DEVICE)

    loss_history = []
    schedule_history = []

    print("Starting Training...")
    pbar = tqdm(range(TrainConfig.EPOCHS))

    # 3. Dummy Data (Replace with datasets/dataset_loader.py in real version)
    TRAIN_PROMPTS = [
        "A macro photograph of a bumblebee on a yellow flower",
        "A close-up portrait of a snowy owl with detailed feathers",
        "A cute corgi running through a field of tall green grass",
        "A wet otter swimming in a river, 4k high resolution",
        "A detailed close-up of a dragon eye with scales",
        "A futuristic cyberpunk city at night with neon rain",
        "A serene misty mountain lake at sunrise, landscape photography",
        "A dusty desert road stretching into the horizon",
        "A dense tropical rainforest with sunlight filtering through leaves",
        "A snowy cabin in the woods during a winter storm",
        "A delicious pepperoni pizza with melting cheese",
        "A vintage typewriter sitting on an old wooden desk",
        "A shiny red sports car driving on a coastal highway",
        "A steaming cup of coffee with latte art on a saucer",
        "A stack of old leather-bound books in a library",
        "A portrait of an old fisherman with a weathered face",
        "A professional headshot of a smiling woman in business attire",
        "A renaissance style oil painting of a young princess",
        "A cyberpunk android girl with glowing circuitry on her face",
        "A pencil sketch of a bearded man, rough charcoal style",
        "An impressionist oil painting of a water lily pond",
        "A flat vector illustration of a rocket ship launching",
        "A watercolor painting of a cozy cafe in Paris",
        "A low poly 3d render of a deer in a forest",
        "A pop art style poster of a banana, warhol style",
        ]
    train_data = generate_teacher_target(pipe, TRAIN_PROMPTS, TrainConfig.DEVICE)

    # FIX: Pre-compute Null Embedding for Blind LoRA Training
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
            text_emb = data["emb"]
            student_latents = data["noise"].clone().requires_grad_(True)
            target_latents = data["target"]

            optimizer.zero_grad()

            # === The Magic Wrapper for Float16 ===
            with autocast():
                # 1. Run Student Loop
                for k in range(SchedulerConfig.K_STEPS):
                    # Student MLP runs in fp32 inside, but accepts mixed inputs via autocast
                    t_curr = student(k, student_latents)

                    if i == 0: current_schedule.append(t_curr.item())

                    if k < SchedulerConfig.K_STEPS - 1:
                        t_next = student(k+1, student_latents)
                        t_next = torch.min(t_next, t_curr - 1).clamp(min=0)
                    else:
                        t_next = torch.zeros_like(t_curr)

                    student_latents = diff_handler.step(student_latents, t_curr, t_next, text_emb, guidance_scale=1.0)

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

        avg_loss = epoch_loss / len(train_data)
        loss_history.append(avg_loss)
        schedule_history.append(current_schedule)
        pbar.set_description(f"Scheduler Loss: {avg_loss:.4f}")

    print("Scheduler Training Complete!")

    # ==========================================
    # PHASE 2: TRAIN REFINER (LoRA)
    # ==========================================
    print("\n=== Phase 2: Training Refiner (LoRA) ===")
    pipe.unet.enable_adapter_layers() # Enable LoRA

    # FIX: Correct way to get LoRA parameters
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

            refiner_optimizer.zero_grad()

            with autocast():
                # A. Run Scheduler to get "Blurry Input" (No Grad on Scheduler)
                # We purposefully disable LoRA for the first K-1 steps to simulate
                # the "coarse" generation.
                pipe.unet.disable_adapter_layers()

                with torch.no_grad():
                    for k in range(SchedulerConfig.K_STEPS - 1): # Run all but last
                        t_curr = student(k, latents)
                        t_next = student(k+1, latents)
                        # Clamp logic
                        t_next = torch.min(t_next, t_curr - 1).clamp(min=0)
                        latents = diff_handler.step(latents, t_curr, t_next, text_emb, guidance_scale=1.0)

                # B. The Final Step (Refinement)
                # Now we ENABLE LoRA. This is the only step we train.
                pipe.unet.enable_adapter_layers()

                # Get the final timestep (usually jump to 0)
                t_curr = student(SchedulerConfig.K_STEPS - 1, latents)
                t_next = torch.zeros_like(t_curr)

                # FIX: Blind Training. Use null_emb instead of text_emb.
                # This forces LoRA to look at pixels, avoiding prompt overfitting.
                # We use the diff_handler to execute the step WITH gradients.
                refined_latents = diff_handler.step(latents, t_curr, t_next, null_emb, guidance_scale=1.0)

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


    # 4. Visualize (Optional)
    k = 8
    k_idx = random.sample([i for i in range(len(TRAIN_PROMPTS))], k)
    for i in k_idx:
      visualize_scheduling_results(student, diff_handler, pipe, train_data, loss_history, schedule_history, SchedulerConfig.K_STEPS, idx = i)


    # 5. Evaluate based on the LPIPS Score
    EVAL_PROMPTS = [
        "A extreme close-up of a fluffy cat, highly detailed fur texture",
        "A field of tall grass blowing in the wind, oil painting style",
        "A cyberpunk city street at night with wet pavement and neon lights",
        "A silhouette of a cowboy riding against a bright sunset",
        "A transparent glass cube sitting on a wooden table, raytracing",
        "A bowl of fresh fruit containing apples, bananas, and grapes",
        "Abstract geometric shapes, bauhaus style poster, red and blue",
    ]
    # Ensure LoRA is enabled for evaluation
    pipe.unet.enable_adapter_layers()
    evaluation_pipeline(pipe, diff_handler, student, EVAL_PROMPTS, SchedulerConfig.K_STEPS, DEVICE=ModelConfig.DEVICE)

    # 6. Save
    torch.save(student.state_dict(), "student_scheduler.pth")
    # Save LoRA weights as well
    pipe.unet.save_pretrained("refiner_lora")
    print("Done!")

if __name__ == "__main__":
    main()
