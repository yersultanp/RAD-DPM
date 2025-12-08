import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('/content/drive/MyDrive/Adversarial-Diffusion-Distillation')
from configs.scheduler_config import SchedulerConfig
from ddim_utils import extract_latent_stats

class RobustLearnedScheduler(nn.Module):
    def __init__(self, num_steps = SchedulerConfig.K_STEPS, input_dim=SchedulerConfig.LATENT_DIM, hidden_dim=SchedulerConfig.SCHEDULER_HIDDEN_DIM):
        super().__init__()
        self.num_steps = num_steps

        # 1. The Neural Network (The "Brain")
        # We use a residual connection logic.
        self.net = nn.Sequential(
            nn.Linear(input_dim + 32, hidden_dim), # +32 for step embedding
            nn.SiLU(), # SiLU (Swish) is better for diffusion than ReLU
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh() # Output range [-1, 1] for "nudging" the schedule
        )

        # Step Embedding
        self.step_embed = nn.Embedding(num_steps, SchedulerConfig.SCHEDULER_EMBED_DIM)

        # 2. Hardcoded "Safe" Linear Schedule (The "Anchor")
        # If we want 4 steps, standard is: 1000 -> 750 -> 500 -> 250 -> 0
        # We register this as a buffer so it's not a trainable parameter
        timesteps = np.linspace(1, 0, num_steps + 1)[:-1] 

        # 2. Square it (Quadratic curve) to push values lower
        # e.g., 0.5 becomes 0.25
        timesteps = timesteps ** 2 

        # 3. Scale back to 1000
        # Result: [1000, 562, 250, 62]
        linear_schedule = timesteps * 1000
        self.register_buffer("default_schedule", torch.tensor(linear_schedule, dtype=torch.float32))

        # 3. Learnable Scale
        # Controls how much the network is allowed to deviate from the default.
        # Initialize small so training starts stable!
        self.deviation_scale = nn.Parameter(torch.tensor(80.0))

    def forward(self, step_idx, latents):
        # A. Get the "Safe" Default Value for this step
        # e.g., if step_idx=0, base_t = 1000. If step_idx=1, base_t = 750.
        latents = latents.float()
        base_t = self.default_schedule[step_idx]

        # B. Get the Network Prediction (The "Nudge")
        # Extract stats: [Batch, 4]
        stats = extract_latent_stats(latents)  # [Batch, 4]

        # Embed step
        step_idx_tensor = torch.tensor([step_idx] * latents.shape[0], device=latents.device)
        emb = self.step_embed(step_idx_tensor)

        # Predict deviation (-1 to 1)
        inp = torch.cat([emb, stats], dim=1)
        nudge = self.net(inp)

        # C. Combine: t = Base + (Nudge * Scale)
        # e.g. t = 750 + (0.5 * 100) = 800
        t_pred = base_t + (nudge * self.deviation_scale)

        # D. Safety Clamps
        # 1. t must be > 0
        # 2. t must be < 1000
        # 3. CRITICAL: In a sequence, we enforce monotonicity in the training loop,
        #    but here we just ensure valid bounds.
        return t_pred.clamp(1.0, 1000.0)
