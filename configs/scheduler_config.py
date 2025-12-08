# configs/scheduler_config.py
import torch

class SchedulerConfig:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE_MODEL = torch.float16 if DEVICE == "cuda" else torch.float32
    LATENT_DIM = 4
    SCHEDULER_HIDDEN_DIM = 64
    SCHEDULER_EMBED_DIM = 32
    K_STEPS = 4 # Number of steps for the student scheduler
