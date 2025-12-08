import torch
from configs.model_config import ModelConfig
from configs.scheduler_config import SchedulerConfig
from configs.train_config import TrainConfig

def extract_latent_stats(latents):
    mean = latents.mean(dim=[1, 2, 3], keepdim=True).squeeze(-1).squeeze(-1)
    std = latents.std(dim=[1, 2, 3], keepdim=True).squeeze(-1).squeeze(-1)
    energy = (latents**2).mean(dim=[1, 2, 3], keepdim=True).squeeze(-1).squeeze(-1)
    maxval = latents.max(dim=[1, 2, 3], keepdim=True).values.squeeze(-1).squeeze(-1)
    return torch.cat([mean, std, energy, maxval], dim=1)


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

    def step(self, latents, t_now, t_next, text_emb):
        alpha_now, sigma_now = self.get_alpha_sigma(t_now)
        alpha_next, _ = self.get_alpha_sigma(t_next)

        # UNet expects float16 if loaded in float16
        # t needs to be passed as float16 for the embedding layer if strictly typed

        # NOTE: diffusers UNet `sample` method handles the casting if using autocast
        noise_pred = self.unet(latents, t_now.squeeze(), encoder_hidden_states=text_emb).sample

        # Cast back to match latents if needed (usually handled by autocast)
        noise_pred = noise_pred.to(dtype=latents.dtype)

        pred_x0 = (latents - sigma_now * noise_pred) / alpha_now
        dir_xt = (1 - alpha_next**2)**0.5 * noise_pred
        prev_latents = alpha_next * pred_x0 + dir_xt

        return prev_latents
