import torch
import sys
import os
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import random

# Adjust path if necessary
# sys.path.append('/content/drive/MyDrive/Adversarial-Diffusion-Distillation')

from configs.model_config import ModelConfig
from configs.train_config import TrainConfig
from configs.scheduler_config import SchedulerConfig
from models.teacher import load_teacher_model
from models.student import RecurrentScheduler
# Ensure this imports the NEW FP16-safe handler
from ddim_utils import DifferentiableDPMSolverHandler 
from train.train_step import generate_teacher_target
from losses import HybridLatentLoss
from models.refiner import attach_refiner_lora
# Ensure you added plot_refiner_history to this file/import
from eval.visualize_schedule import plot_scheduler_training_history, analyze_schedule_variance, plot_refiner_history
from eval.plot_results import comparison_pipeline, visualize_sequence_comparison

def run_training_for_k(k_steps):
    """
    Runs the entire training pipeline for a specific K step count.
    Returns a dictionary of results.
    """
    # 1. Update Config
    SchedulerConfig.K_STEPS = k_steps
    print(f"\n{'='*40}")
    print(f"STARTING TRAINING FOR K={k_steps}")
    print(f"{'='*40}\n")

    # 2. Re-Initialize Models (Fresh Start)
    student = RecurrentScheduler(input_dim=4).to(TrainConfig.DEVICE)
    # Lower LR slightly for K=8 stability
    lr = 1e-4 if k_steps >= 8 else TrainConfig.LEARNING_RATE
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    
    pipe = load_teacher_model(TrainConfig.DEVICE)
    pipe = attach_refiner_lora(pipe) 
    
    scaler = GradScaler()
    dpm_handler = DifferentiableDPMSolverHandler(pipe)
    loss_fn = HybridLatentLoss(alpha_mse=1.0, alpha_cos=0.1, alpha_stats=0.1).to(TrainConfig.DEVICE)

    loss_history = []
    schedule_history = []

    # 3. Data Setup (SFW / Texture Prompts)
    TRAIN_PROMPTS = [
    # --- Animals (Fur/Feathers - High Texture) ---
    "A fluffy red panda sleeping on a tree branch, 4k resolution",
    "A majestic lion with a thick mane standing in the savannah",
    "A macro photo of a peacock feather showing iridescent colors",
    "A close-up of a blue dragonfly resting on a green leaf",
    "A baby penguin standing on an iceberg in Antarctica",
    "A koala hugging a eucalyptus tree, detailed fur texture",
    "A vibrant green frog sitting on a lotus leaf",
    "A large elephant walking through dust at sunset",
    
    # --- Landscapes & Nature (Lighting/Depth) ---
    "A dense bamboo forest with sunlight filtering through the stalks",
    "A serene alpine lake reflecting snow-capped mountains",
    "A desert canyon with red rock formations under a blue sky",
    "A field of sunflowers facing the sun, bright and colorful",
    "A Northern Lights aurora borealis over a winter forest",
    "A tropical island beach with palm trees and turquoise water",
    "A misty morning in a pine forest, atmospheric lighting",
    "A volcanic eruption with flowing lava and smoke",

    # --- Architecture & Sci-Fi (Geometry/Lines) ---
    "A futuristic space station orbiting a blue planet",
    "A steampunk clock tower with brass gears and steam",
    "A cybernetic robot hand holding a glowing blue orb",
    "A medieval stone castle on a hill, cloudy sky",
    "A modern glass bridge connecting two skyscrapers",
    "A cozy wooden cabin interior with a fireplace and rug",
    "A high-speed train traveling through a futuristic tunnel",
    "An isometric view of a low-poly magical island",

    # --- Objects & Still Life (Material Properties) ---
    "A crystal chess set arranged on a glass table",
    "A vintage brass compass lying on an old map",
    "A basket of fresh red apples and green grapes",
    "A detailed macro shot of a mechanical watch movement",
    "A pile of gold coins and jewels in a treasure chest",
    "A ceramic teapot with floral patterns, studio lighting"
    ]     

    print("Generating Teacher Targets...")
    train_data = generate_teacher_target(pipe, TRAIN_PROMPTS, TrainConfig.DEVICE)

    null_inputs = pipe.tokenizer("", return_tensors="pt", padding="max_length", truncation=True).to(TrainConfig.DEVICE)
    with torch.no_grad():
        null_emb = pipe.text_encoder(null_inputs.input_ids)[0]

    # ==========================================
    # PHASE 1: TRAIN SCHEDULER
    # ==========================================
    print(f"\n[K={k_steps}] Phase 1: Training Scheduler...")
    pipe.unet.disable_adapter_layers() 
    
    pbar = tqdm(range(TrainConfig.EPOCHS))
    for epoch in pbar:
        epoch_loss = 0
        current_schedule_snapshot = [] 

        for i, data in enumerate(train_data):
            text_emb = data["emb"]
            latents = data["noise"].clone().requires_grad_(True)
            target_latents = data["target"]
            
            # RNN State Init
            t_curr = torch.full((TrainConfig.BATCH_SIZE, 1), 1000.0, device=TrainConfig.DEVICE)
            hx = None
            prev_noise = None
            prev_h = None 
            
            sample_schedule = [1000.0]

            optimizer.zero_grad()

            with autocast():
                for k in range(k_steps):
                    # Unpack RNN
                    t_next, hx = student(latents, t_curr, hx)
                    
                    if k < k_steps - 1:
                        # Adaptive constraint
                        min_step = 10.0 if k_steps >= 8 else 50.0
                        max_allowed = t_curr - 10.0
                        t_next = torch.min(t_next, max_allowed).clamp(min=min_step)
                    else:
                        t_next = torch.zeros_like(t_curr)

                    # Log schedule
                    if i == 0: sample_schedule.append(t_next.item())

                    # DPM Step (FP16-Native Handler)
                    latents, curr_noise, curr_h = dpm_handler.step(
                        latents, t_curr, t_next, text_emb,
                        prev_noise_pred=prev_noise, prev_h=prev_h, 
                        guidance_scale=1.0
                    )

                    prev_noise = curr_noise
                    prev_h = curr_h
                    t_curr = t_next

                loss = loss_fn(latents, target_latents)
            
            # NaN Guard
            if torch.isnan(loss):
                print(f"⚠️ Warning: NaN Loss at Epoch {epoch}. Skipping.")
                optimizer.zero_grad()
                continue 

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            if i == 0: current_schedule_snapshot.append(sample_schedule)

        avg_loss = epoch_loss / len(train_data)
        loss_history.append(avg_loss)
        if current_schedule_snapshot:
            schedule_history.append(current_schedule_snapshot[0]) 
        
        pbar.set_description(f"Loss: {avg_loss:.4f}")

    # ==========================================
    # PHASE 2: TRAIN REFINER (LoRA)
    # ==========================================
    print(f"\n[K={k_steps}] Phase 2: Training Refiner (LoRA)...")
    pipe.unet.enable_adapter_layers()
    
    refiner_params = [p for p in pipe.unet.parameters() if p.requires_grad]
    refiner_optimizer = torch.optim.AdamW(refiner_params, lr=1e-4)
    student.requires_grad_(False) 

    REFINER_EPOCHS = 15 
    
    # === NEW: Refiner History Container ===
    refiner_loss_history = [] 

    for epoch in range(REFINER_EPOCHS):
        epoch_loss = 0
        pbar_ref = tqdm(train_data, desc=f"Refiner Epoch {epoch+1}")

        for data in pbar_ref:
            text_emb = data["emb"]
            latents = data["noise"].clone()
            target_latents = data["target"]
            
            t_curr = torch.full((TrainConfig.BATCH_SIZE, 1), 1000.0, device=TrainConfig.DEVICE)
            hx = None; prev_noise = None; prev_h = None

            refiner_optimizer.zero_grad()

            with autocast():
                # A. Run Scheduler (LoRA OFF)
                pipe.unet.disable_adapter_layers()
                
                with torch.no_grad():
                    for k in range(k_steps - 1):
                        t_next, hx = student(latents, t_curr, hx)
                        max_allowed = t_curr - 10.0
                        t_next = torch.min(t_next, max_allowed).clamp(min=10.0)

                        latents, curr_noise, curr_h = dpm_handler.step(
                            latents, t_curr, t_next, text_emb,
                            prev_noise_pred=prev_noise, prev_h=prev_h,
                            guidance_scale=1.0
                        )
                        prev_noise = curr_noise
                        prev_h = curr_h
                        t_curr = t_next

                # B. Final Step (LoRA ON)
                pipe.unet.enable_adapter_layers()
                t_next = torch.zeros_like(t_curr)

                refined_latents, _, _ = dpm_handler.step(
                    latents, t_curr, t_next, null_emb, 
                    prev_noise_pred=prev_noise, prev_h=prev_h,
                    guidance_scale=1.0
                )

                loss = loss_fn(refined_latents, target_latents)

            scaler.scale(loss).backward()
            scaler.unscale_(refiner_optimizer)
            torch.nn.utils.clip_grad_norm_(refiner_params, 1.0)
            scaler.step(refiner_optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar_ref.set_description(f"Loss: {loss.item():.4f}")
        
        # === NEW: Log Refiner Loss ===
        avg_ref_loss = epoch_loss / len(train_data)
        refiner_loss_history.append(avg_ref_loss)

    # ==========================================
    # EVALUATION & SAVING
    # ==========================================
    save_dir = f"./final_results_K{k_steps}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(student.state_dict(), os.path.join(save_dir, "student_scheduler.pth"))
    
    print(f"Plotting Training History for K={k_steps}...")
    idx = 0
    plot_scheduler_training_history(loss_history, schedule_history, idx, save_dir=save_dir)
    
    # === NEW: Plot Refiner History ===
    plot_refiner_history(refiner_loss_history, save_dir=save_dir)

    # Eval Prompts
    EVAL_PROMPTS = [
    # --- Texture Stress Test (Fine Detail) ---
    "A close-up of a woven wicker basket, high detail",
    "A macro shot of snowflakes on a dark wool mitten",
    "A detailed painting of a dragon scale texture",
    
    # --- Contrast & Lighting Test (Dynamic Range) ---
    "A lighthouse beam cutting through a dark stormy night",
    "A neon sign reflecting in a rainy street puddle",
    "A bright solar eclipse with the diamond ring effect",

    # --- Geometry Stress Test (Straight Lines) ---
    "A blueprint of a complex engine on blue paper",
    "A perfectly symmetrical kaleidoscope pattern",
    "A wireframe 3D model of a sports car, neon lines",

    # --- Composition Test (Object Separation) ---
    "A red apple, a yellow banana, and a green pear in a row",
    "A stack of colorful hardcover books on a wooden shelf",
    "A collection of different seashells on sand"
    ]

    mean_schedule = analyze_schedule_variance(
        student, EVAL_PROMPTS, k_steps, 
        device=TrainConfig.DEVICE, save_dir=save_dir
    )
    
    pipe.unet.enable_adapter_layers()
    
    scores = comparison_pipeline(
        pipe, dpm_handler, student, 
        EVAL_PROMPTS, 
        K_STEPS=k_steps, 
        DEVICE=TrainConfig.DEVICE, 
        save_dir=os.path.join(save_dir, "comparison_grid.png")
    )
    
    visualize_sequence_comparison(
        pipe, dpm_handler, student, 
        prompt="A close-up of a woven wicker basket, high detail", 
        K_STEPS=k_steps, 
        DEVICE=TrainConfig.DEVICE, 
        save_path=os.path.join(save_dir, "denoising_sequence.png")
    )

    return {
        "K": k_steps,
        "Scores": scores, 
        "Mean Schedule": [int(t) for t in mean_schedule]
    }

if __name__ == "__main__":
    k_values = [2, 4, 8] # Run specific experiment
    final_results = []

    try:
        for k in k_values:
            result = run_training_for_k(k)
            final_results.append(result)
            torch.cuda.empty_cache()
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    print("\n" + "="*60)
    print("FINAL EXPERIMENT REPORT")
    print("="*60)
    
    print(f"{'K Steps':<10} | {'Baseline':<15} | {'Sched':<15} | {'Sched + LoRA':<15}")
    print("-" * 60)
    
    for res in final_results:
        k = res['K']
        s_base = res['Scores'].get('Simple DPM', 0.0)
        s_sched = res['Scores'].get('Scheduled', 0.0)
        s_lora = res['Scores'].get('Scheduled + LoRA', 0.0)
        
        print(f"{k:<10} | {s_base:.4f}          | {s_sched:.4f}          | {s_lora:.4f}")

    print("\n" + "="*60)
    print("LEARNED SCHEDULES")
    print("="*60)
    for res in final_results:
        print(f"K={res['K']}: {res['Mean Schedule']}")
    print("="*60)