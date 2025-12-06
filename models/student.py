import torch
import torch.nn as nn
from configs.model_config import ModelConfig

class LearnedStepScheduler(nn.Module):
    """
    A tiny MLP that predicts which timestep 't' to use for the current step.
    Inputs:
      - step_index: Current step number (0 to K-1)
      - latent_stats: Mean and Var of the current latent (simple content descriptor)
    Output:
      - t: A continuous value between 0 and 1000
    """
    def __init__(self, num_steps, input_dim=2):
        super().__init__()
        self.num_steps = num_steps

        # Simple embedding for the step index (0, 1, 2...)
        self.step_embed = nn.Embedding(num_steps, 32)

        # MLP to predict the timestep
        # Input: Step embedding (32) + Latent Stats (2: mean, std)
        self.net = nn.Sequential(
            nn.Linear(32 + input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1), # Output scalar t
            nn.Sigmoid()      # Bound output to [0, 1] then scale to [0, 1000]
        )

    def forward(self, step_idx, latents):
        # 1. Extract simple stats from latents to make the schedule adaptive
        # Shape: [Batch, Channels, H, W] -> [Batch, 2]
        mean = latents.mean(dim=[1, 2, 3], keepdim=True).squeeze(-1).squeeze(-1)
        std = latents.std(dim=[1, 2, 3], keepdim=True).squeeze(-1).squeeze(-1)
        stats = torch.cat([mean, std], dim=1)

        # 2. Get step embedding
        step_idx_tensor = torch.tensor([step_idx] * latents.shape[0], device=latents.device)
        emb = self.step_embed(step_idx_tensor)

        # 3. Predict t
        inp = torch.cat([emb, stats], dim=1)
        t_norm = self.net(inp)

        # Scale to diffusion range (usually 0-1000)
        # We clamp slightly to avoid 0 or 1000 exactly for numerical stability
        return t_norm * 999.0 + 0.1

    def get_all_timesteps(self, latents):
        batch_size = latents.shape[0]
        all_timesteps = []

        for step_idx in range(self.num_steps):
            t = self.forward(step_idx, latents)
            all_timesteps.append(t)

        return torch.cat(all_timesteps, dim=0).view(batch_size, self.num_steps)



class SchedulerMLP(nn.Module):
    def __init__(self, stat_dim, cond_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stat_dim + cond_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, stats, step_idx, cond):
        step_norm = step_idx / 10.0
        x = torch.cat([stats, cond, step_norm.unsqueeze(0)], dim=-1)
        return self.net(x)


def build_scheduler(cond_dim):
    cfg = ModelConfig()
    return SchedulerMLP(
        stat_dim=cfg.latent_stat_dim,
        cond_dim=cond_dim,
        hidden_dim=cfg.scheduler_hidden_dim
    )
