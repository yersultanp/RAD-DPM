class DifferentiableDiffusionHandler:
    """
    Helper to perform diffusion steps where 't' is a continuous variable
    that allows backpropagation.
    """
    def __init__(self, pipe):
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler
        self.alphas_cumprod = self.scheduler.alphas_cumprod.to(pipe.device)

    def get_alpha_sigma(self, t):
        """
        Interpolates alpha_cumprod for a continuous timestep t.
        This allows gradients to flow from the loss -> latent -> alpha -> t.
        """
        # t is shape [Batch, 1]
        t = t.squeeze()

        # Indices for interpolation
        low_idx = t.floor().long().clamp(0, len(self.alphas_cumprod)-2)
        high_idx = low_idx + 1

        alpha_low = self.alphas_cumprod[low_idx]
        alpha_high = self.alphas_cumprod[high_idx]

        # Weights for interpolation
        w = t - low_idx.float()

        # Linear interpolation of alpha_cumprod
        # (Log-linear interpolation is often more accurate for diffusion,
        # but linear is sufficient for a demo)
        alpha_t = (1 - w) * alpha_low + w * alpha_high

        # Expand for broadcasting [Batch, 1, 1, 1]
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        sigma_t = (1 - alpha_t) ** 0.5
        alpha_t = alpha_t ** 0.5 # We usually need sqrt(alpha_cumprod)

        return alpha_t, sigma_t

    def step(self, latents, t_now, t_next, text_embeddings):
        """
        One Differentiable DDIM Step.
        """
        # 1. Get noise levels for current t and next t
        alpha_now, sigma_now = self.get_alpha_sigma(t_now)
        alpha_next, sigma_next = self.get_alpha_sigma(t_next)

        # 2. Predict Noise (UNet is differentiable w.r.t input latents and t)
        # Note: diffusers unet accepts float t if provided
        noise_pred = self.unet(latents, t_now.squeeze(), encoder_hidden_states=text_embeddings).sample

        # 3. Predict x0 (clean image approximation)
        # latents = alpha_now * x0 + sigma_now * epsilon
        # x0 = (latents - sigma_now * epsilon) / alpha_now
        pred_x0 = (latents - sigma_now * noise_pred) / alpha_now

        # 4. Calculate direction to x_t_next
        # DDIM equation (deterministic)
        dir_xt = (1 - alpha_next**2)**0.5 * noise_pred # simplified sigma assumption

        # 5. Compute x_t_next
        prev_latents = alpha_next * pred_x0 + dir_xt

        return prev_latents

def ddim_step(x, eps, t, scheduler):
    # Implement DDIM update rule
    # alpha_t, sigma_t etc from scheduler


    return x_next
