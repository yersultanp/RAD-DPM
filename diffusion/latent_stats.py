import torch

def extract_latent_stats(latent):
    mean = latent.mean().unsqueeze(0)
    std = latent.std().unsqueeze(0)
    energy = (latent**2).mean().unsqueeze(0)
    maxval = latent.max().unsqueeze(0)
    return torch.cat([mean, std, energy, maxval], dim=0)
