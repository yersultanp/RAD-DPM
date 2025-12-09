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

    def get_lambda(self, t):
        """
        Computes Log-SNR (Lambda) differentiably.
        Lambda = log(alpha / sigma)
        """
        # 1. Get Alpha/Sigma (using Log-Linear Interpolation for stability)
        t = t.squeeze()
        low_idx = t.floor().long().clamp(0, len(self.alphas_cumprod)-2)
        high_idx = low_idx + 1
        w = t - low_idx.float()
        
        alpha_low = self.alphas_cumprod[low_idx]
        alpha_high = self.alphas_cumprod[high_idx]
        
        # Log-Linear Interpolation
        log_alpha_t = (1 - w) * torch.log(alpha_low) + w * torch.log(alpha_high)
        alpha_t = torch.exp(log_alpha_t)

        # Reshape
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        sigma_t = (1 - alpha_t) ** 0.5
        alpha_t = alpha_t ** 0.5 
        
        # 2. Compute Lambda
        # lambda = log(alpha/sigma)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        
        return lambda_t, alpha_t, sigma_t

    def step(self, latents, t_now, t_next, text_emb, 
             prev_noise_pred=None, prev_lambda=None, guidance_scale=1.0):
        
        # --- 1. PREDICT NOISE (With Optional CFG) ---
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

        # --- 2. DPM-SOLVER++ MATH ---
        
        # Get Parameters for current and next step
        lambda_now, alpha_now, sigma_now = self.get_lambda(t_now)
        lambda_next, alpha_next, sigma_next = self.get_lambda(t_next)
        
        # h = step size in log-SNR space
        h = lambda_next - lambda_now
        
        # Phi_1 (Exponential Integrator factor)
        # x_t = (sigma_t / sigma_s) * x_s - alpha_t * (exp(-h) - 1) * noise
        phi_1 = torch.expm1(-h) # exp(-h) - 1

        # A. First Order Update (Standard DPM/DDIM)
        # This is the baseline term
        x_inter = (sigma_next / sigma_now) * latents - (alpha_next * phi_1) * noise_pred

        # B. Second Order Correction (Multistep)
        # Only possible if we have history (prev_noise_pred)
        if prev_noise_pred is not None and prev_lambda is not None:
            # Calculate ratio r for the 2nd order term
            # r = h_current / h_previous
            h_prev = lambda_now - prev_lambda
            r = h / h_prev
            
            # D = (1 + 1/(2r)) * noise_now - (1/(2r)) * noise_prev
            # Correction term adds curvature awareness
            D = (1 + 1.0 / (2.0 * r)) * noise_pred - (1.0 / (2.0 * r)) * prev_noise_pred
            
            # Apply correction
            # Formula: - alpha_next * (exp(-h) - 1) * (D - noise_pred)
            # This simplifies to replacing noise_pred with D in the original equation, 
            # but usually implemented as an additive term.
            
            correction = 0.5 * (alpha_next * phi_1) * (noise_pred - prev_noise_pred) / r
            # Note: Implementations vary slightly, this matches the standard 2M Taylor expansion approximation
            
            # More robust DPM-Solver++ 2M formula:
            # x = (sigma_next/sigma_now) * latents - alpha_next * phi_1 * noise_pred 
            #     - 0.5 * alpha_next * phi_1 * (noise_pred - prev_noise_pred)
            # (Assuming step sizes are roughly similar, r approx 1. 
            # The 'r' term handles unequal steps).
            
            # Exact formulation:
            correction = alpha_next * phi_1 * (1.0 / (2.0 * r)) * (noise_pred - prev_noise_pred)
            
            latents_next = x_inter - correction
        else:
            # First step (or no history): Fallback to First Order
            latents_next = x_inter

        return latents_next, noise_pred, lambda_now
