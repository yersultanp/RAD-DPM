# train/train_step.py

import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor

def generate_teacher_target(pipe, text_emb, device):
    """Generates the Ground Truth using 50 steps (No Grad)"""
    latents_shape = (1, 4, 64, 64)
    init_noise = randn_tensor(latents_shape, device=device, dtype=text_emb.dtype)
    
    teacher_latents = init_noise
    pipe.scheduler.set_timesteps(50)
    
    with torch.no_grad():
        for t in pipe.scheduler.timesteps:
            noise_pred = pipe.unet(teacher_latents, t, encoder_hidden_states=text_emb).sample
            teacher_latents = pipe.scheduler.step(noise_pred, t, teacher_latents).prev_sample
            
    return init_noise, teacher_latents

def train_one_step(student_scheduler, diff_handler, optimizer, pipe, text_emb, k_steps):
    device = pipe.device
    
    # 1. Get Ground Truth
    init_noise, target_latents = generate_teacher_target(pipe, text_emb, device)
    
    # 2. Run Student Loop (With Grad)
    student_latents = init_noise.clone()
    optimizer.zero_grad()
    
    for k in range(k_steps):
        # Predict t
        t_curr = student_scheduler(k, student_latents)
        
        # Lookahead logic (simple next step prediction)
        if k < k_steps - 1:
            t_next = student_scheduler(k+1, student_latents)
        else:
            t_next = torch.zeros_like(t_curr)
            
        # Ensure time flows backward
        t_next = torch.min(t_next, t_curr - 1).clamp(min=0)
        
        # Differentiable Step
        student_latents = diff_handler.step(student_latents, t_curr, t_next, text_emb)
    
    # 3. Loss & Update
    loss = F.mse_loss(student_latents, target_latents)
    loss.backward()
    optimizer.step()
    
    return loss.item()