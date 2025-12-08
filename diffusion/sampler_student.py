import torch

class DifferentiableDiffusionHandler:
    def __init__(self, pipe):
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(pipe.device)

    def get_alpha_sigma(self, t):
        """Interpolates alpha_cumprod for continuous timestep t."""
        t = t.squeeze()
        low_idx = t.floor().long().clamp(0, len(self.alphas_cumprod)-2)
        high_idx = low_idx + 1

        alpha_low = self.alphas_cumprod[low_idx]
        alpha_high = self.alphas_cumprod[high_idx]
        w = t - low_idx.float()
        
        # Linear Interpolation
        alpha_t = (1 - w) * alpha_low + w * alpha_high
        
        # Reshape for broadcasting
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        sigma_t = (1 - alpha_t) ** 0.5
        alpha_t = alpha_t ** 0.5 

        return alpha_t, sigma_t

    def step(self, latents, t_now, t_next, text_embeddings):
        """Differentiable DDIM Step"""
        alpha_now, sigma_now = self.get_alpha_sigma(t_now)
        alpha_next, _ = self.get_alpha_sigma(t_next) # sigma_next not strictly needed for deterministic DDIM

        # UNet Prediction
        noise_pred = self.unet(latents, t_now.squeeze(), encoder_hidden_states=text_embeddings).sample

        # Predict x0
        pred_x0 = (latents - sigma_now * noise_pred) / alpha_now

        # Calculate Direction
        dir_xt = (1 - alpha_next**2)**0.5 * noise_pred
        
        # Next Latent
        prev_latents = alpha_next * pred_x0 + dir_xt
        return prev_latents
