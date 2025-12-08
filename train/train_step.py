# train/train_step.py

import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from configs.model_config import ModelConfig
from configs.train_config import TrainConfig
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


def generate_teacher_target(pipe, prompts, device):
    train_data = []

    print("Pre-computing Teacher Targets (float16)...")
    for p in tqdm(prompts):
        inputs = pipe.tokenizer(p, return_tensors="pt", padding="max_length", truncation=True).to(ModelConfig.DEVICE)
        with torch.no_grad():
            with autocast(): # Use AMP for generation too
                text_emb = pipe.text_encoder(inputs.input_ids)[0]

                latents = randn_tensor((1, 4, 64, 64), device=ModelConfig.DEVICE, dtype=text_emb.dtype)
                init_noise = latents.clone()

                pipe.scheduler.set_timesteps(ModelConfig.TEACHER_STEPS)
                for t in pipe.scheduler.timesteps:
                    noise_pred = pipe.unet(latents, t, encoder_hidden_states=text_emb).sample
                    latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        train_data.append({"emb": text_emb, "noise": init_noise, "target": latents})
    return train_data


def train_one_step(student, diff_handler, optimizer, pipe, train_data, K_STEPS, current_schedule = [], scaler=None):
    for i, data in enumerate(train_data):
        text_emb = data["emb"]
        student_latents = data["noise"].clone().requires_grad_(True)
        target_latents = data["target"]

        optimizer.zero_grad()

        # === The Magic Wrapper for Float16 ===
        with autocast():
            # 1. Run Student Loop
            for k in range(K_STEPS):
                # Student MLP runs in fp32 inside, but accepts mixed inputs via autocast
                t_curr = student(k, student_latents)

                if i == 0: current_schedule.append(t_curr.item())

                if k < K_STEPS - 1:
                    t_next = student(k+1, student_latents)
                    t_next = torch.min(t_next, t_curr - 1).clamp(min=0)
                else:
                    t_next = torch.zeros_like(t_curr)

                student_latents = diff_handler.step(student_latents, t_curr, t_next, text_emb)

            # 2. Compute Loss (in fp16/fp32 automatically handled)
            loss = F.mse_loss(student_latents, target_latents)

        # 3. Backprop with Scaler
        scaler.scale(loss).backward()

        # Unscale grads to clip them
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

    return loss.item()
