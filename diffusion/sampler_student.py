import torch
from diffusion.latent_stats import extract_latent_stats
from diffusion.timestep_utils import map_continuous_to_teacher_timestep
from diffusion.ddim_utils import ddim_step

def student_sample(K, scheduler_model, teacher_unet, text_cond, sched, latent_shape):
    x = torch.randn(latent_shape).to("cuda")

    for k in range(K):
        stats = extract_latent_stats(x)
        t_hat = scheduler_model(stats, torch.tensor(k).float().to("cuda"), text_cond)
        t = map_continuous_to_teacher_timestep(t_hat, sched)
        eps = teacher_unet(x, t, text_cond)
        x = ddim_step(x, eps, t, sched)

    return x
