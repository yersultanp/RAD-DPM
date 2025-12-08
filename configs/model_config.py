# configs/model_config.py
import torch

class ModelConfig:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE_MODEL = torch.float16 if DEVICE == "cuda" else torch.float32
    TEACHER_ID = "sd-legacy/stable-diffusion-v1-5"
    TEACHER_STEPS = 50
    NUM_PROMPTS = 5
