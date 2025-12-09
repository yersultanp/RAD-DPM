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

        # instead of usual scale let's do log linear interpolation
        # to improve the results
        log_alpha_low = torch.log(alpha_low)
        log_alpha_high = torch.log(alpha_high)
        
        log_alpha_t = (1 - w) * log_alpha_low + w * log_alpha_high
        alpha_t = torch.exp(log_alpha_t)

        # 5. Calculate Sigma and Sqrt(Alpha) for DDIM
        # Reshape for broadcasting [Batch, 1, 1, 1]
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        
        # Derived values
        sigma_t = (1 - alpha_t) ** 0.5  # sqrt(1 - alpha_cumprod)
        alpha_t = alpha_t ** 0.5        # sqrt(alpha_cumprod)
        
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

class DifferentiableDPMSolverHandler:
    def __init__(self, pipe):
        self.unet = pipe.unet
        self.alphas_cumprod = pipe.scheduler.alphas_cumprod.to(pipe.device)

    def get_std_params(self, t):
        """
        Returns Alpha and Sigma directly (No Log-SNR).
        """
        t = t.squeeze()
        # Clamp indices
        low_idx = t.floor().long().clamp(0, len(self.alphas_cumprod)-2)
        high_idx = low_idx + 1
        w = t - low_idx.float()
        
        alpha_low = self.alphas_cumprod[low_idx]
        alpha_high = self.alphas_cumprod[high_idx]
        
        # Linear Interpolation in Log-Space for Alpha (Stable)
        # We can do this safely because Alpha is never 0 (always > 0.001)
        log_alpha_t = (1 - w) * torch.log(alpha_low) + w * torch.log(alpha_high)
        alpha_t = torch.exp(log_alpha_t)

        alpha_t = alpha_t.view(-1, 1, 1, 1)
        # Sigma derived from Alpha
        sigma_t = (1 - alpha_t).clamp(min=1e-8) ** 0.5 # Clamp inside sqrt
        alpha_t = alpha_t ** 0.5 
        
        return alpha_t, sigma_t

    def step(self, latents, t_now, t_next, text_emb, 
             prev_noise_pred=None, prev_h=None, guidance_scale=1.0):
        
        # 1. Predict Noise (Standard)
        do_cfg = guidance_scale > 1.0
        if do_cfg:
            latents_input = torch.cat([latents] * 2)
            t_input = torch.cat([t_now] * 2).view(-1)
            noise_pred = self.unet(latents_input, t_input, encoder_hidden_states=text_emb).sample
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
        else:
            t_input = t_now.view(-1)
            noise_pred = self.unet(latents, t_input, encoder_hidden_states=text_emb).sample

        # 2. Get Stats (No Lambda!)
        alpha_now, sigma_now = self.get_std_params(t_now)
        alpha_next, sigma_next = self.get_std_params(t_next)
        
        # 3. Calculate Step Size 'h' via Ratios
        # h = log( (sigma_next * alpha_now) / (sigma_now * alpha_next) )
        # This avoids calculating log(sigma) directly.
        
        # Guard against zero sigma for the ratio
        s_next_safe = sigma_next.clamp(min=1e-6)
        s_now_safe = sigma_now.clamp(min=1e-6)
        
        ratio = (s_next_safe * alpha_now) / (s_now_safe * alpha_next)
        h = torch.log(ratio)

        # 4. Data Prediction (DPM++ Form)
        # x0 = (x_t - sigma * eps) / alpha
        pred_x0 = (latents - sigma_now * noise_pred) / alpha_now

        # 5. Second Order Correction logic
        # We only apply 2nd order if:
        # A. We have history (prev_noise_pred)
        # B. We are NOT at the very last step (sigma_next is not too small)
        #    Calculating 'r' at the last step is unstable in FP16.
        
        is_last_step = sigma_next.mean() < 1e-4
        
        if prev_noise_pred is not None and prev_h is not None and not is_last_step:
            # r = h_current / h_previous
            r = h / (prev_h + 1e-8)
            
            # Correction term
            D_correction = (1.0 / (2.0 * r)) * (noise_pred - prev_noise_pred)
            D = noise_pred + D_correction
        else:
            # First Order (Euler) Fallback
            D = noise_pred

        # 6. Final Update (Standard DPM++ Equation)
        # x_next = alpha_next * x0 + sigma_next * D
        latents_next = alpha_next * pred_x0 + sigma_next * D

        # Return 'h' instead of 'lambda' for the history buffer
        return latents_next, noise_pred, h