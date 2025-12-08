import torch
from configs.model_config import ModelConfig
from configs.scheduler_config import SchedulerConfig
from configs.train_config import TrainConfig

def extract_latent_stats(latents):
    # latents shape: [Batch, 4, 64, 64]

    # 1. Mean (Global Shift)
    mean = latents.mean(dim=[1, 2, 3], keepdim=True).squeeze(-1).squeeze(-1)

    # 2. Std (Global Contrast/Variance - critical for diffusion)
    std = latents.std(dim=[1, 2, 3], keepdim=True).squeeze(-1).squeeze(-1)

    # 3. Robust Max (95th Percentile) - Stable measurement of dynamic range
    # We flatten the spatial dims to compute quantile
    b, c, h, w = latents.shape
    flat = latents.view(b, -1)
    # kthvalue is faster than quantile for tensors
    k = int(0.95 * flat.shape[1])
    p95 = torch.kthvalue(flat, k, dim=1).values.unsqueeze(1)

    # 4. Center Magnitude (L1 Norm centered)
    # This helps distinguish "Gaussian noise" from "Sparse Image Edges"
    l1_mag = (latents - mean.unsqueeze(-1).unsqueeze(-1)).abs().mean(dim=[1, 2, 3])
    l1_mag = l1_mag.unsqueeze(1)

    return torch.cat([mean, std, p95, l1_mag], dim=1)

class DifferentiableDiffusionHandler:
    def __init__(self, pipe):
        self.unet = pipe.unet
        self.alphas_cumprod = pipe.scheduler.alphas_cumprod.to(pipe.device)

    def get_alpha_sigma(self, t):
        # Operations here are sensitive, keep high precision for calculation
        # but cast back for UNet
        t = t.squeeze()
        low_idx = t.floor().long().clamp(0, len(self.alphas_cumprod)-2)
        w = t - low_idx.float()

        alpha_low = self.alphas_cumprod[low_idx]
        alpha_high = self.alphas_cumprod[low_idx+1]

        alpha_t = (1 - w) * alpha_low + w * alpha_high

        alpha_t = alpha_t.view(-1, 1, 1, 1)
        sigma_t = (1 - alpha_t) ** 0.5
        alpha_t = alpha_t ** 0.5
        return alpha_t, sigma_t

    def step(self, latents, t_now, t_next, text_emb, guidance_scale=1.0):
        # Check if we are doing CFG (Inference) or Standard (Training)
        do_classifier_free_guidance = guidance_scale > 1.0
        
        if do_classifier_free_guidance:
            # --- INFERENCE MODE (CFG) ---
            # 1. Expand inputs
            latents_input = torch.cat([latents] * 2)
            # Ensure t is 1D [2]
            t_input = torch.cat([t_now] * 2).view(-1) 
            
            # 2. Predict
            noise_pred = self.unet(latents_input, t_input, encoder_hidden_states=text_emb).sample
            
            # 3. Guide
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
        else:
            # --- TRAINING MODE (Standard) ---
            # 1. No Expansion
            latents_input = latents
            t_input = t_now.view(-1) # Ensure 1D
            
            # 2. Predict (Single pass)
            noise_pred = self.unet(latents_input, t_input, encoder_hidden_states=text_emb).sample

        # --- INTEGRATION (DDIM) ---
        alpha_now, sigma_now = self.get_alpha_sigma(t_now)
        alpha_next, _ = self.get_alpha_sigma(t_next)

        # DDIM Update Rule
        pred_x0 = (latents - sigma_now * noise_pred) / alpha_now
        dir_xt = (1 - alpha_next**2)**0.5 * noise_pred
        prev_latents = alpha_next * pred_x0 + dir_xt

        return prev_latents
