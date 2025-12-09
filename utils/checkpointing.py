import torch
import numpy as np
from diffusers import DPMSolverMultistepScheduler

def verify_solver_parity(pipe, custom_handler, device="cuda"):
    # 1. Setup Inputs
    latents = torch.randn(1, 4, 64, 64).to(device)
    dummy_text_emb = torch.randn(1, 77, 768).to(device)
    
    # 2. Setup Diffusers Scheduler (The "Gold Standard")
    official_sched = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    official_sched.set_timesteps(4)
    timesteps = official_sched.timesteps # e.g. [999, 749, 499, 249]
    
    print(f"Testing Steps: {timesteps}")
    
    # 3. Step 0 Comparison (First Order Step)
    t_0 = timesteps[0] # 999
    t_1 = timesteps[1] # 749
    
    # Get Noise Pred (Mocking the UNet for consistency)
    noise_pred = torch.randn_like(latents) * 0.1
    
    # --- Run Official ---
    # We must mimic the scheduler internal update
    # Diffusers `step` updates internal history, so we must be careful
    out_official = official_sched.step(noise_pred, t_0, latents).prev_sample
    
    # --- Run Custom ---
    # Custom handler usually interpolates. To test parity, we must pass EXACT t
    # and ensure the handler treats step 0 as First Order (no history)
    t_now = torch.tensor([t_0]).to(device)
    t_next = torch.tensor([t_1]).to(device)
    
    out_custom, _, _ = custom_handler.step(
        latents, t_now, t_next, dummy_text_emb, 
        prev_noise_pred=None, # Crucial: First step has NO history
        prev_lambda=None,
        guidance_scale=1.0 # Disable CFG for math check
    )
    # Note: You'll need to mock `self.unet` in handler to return `noise_pred` 
    # or just copy the math logic out for this test.

    # 4. Measure Difference
    diff = (out_official - out_custom).abs().max().item()
    print(f"Step 0 Max Difference: {diff:.6f}")
    
    if diff > 1e-4:
        print(">> FAIL: Step 0 Divergence. Check Alpha/Sigma lookups or First-Order logic.")
    else:
        print(">> PASS: Step 0 matches.")