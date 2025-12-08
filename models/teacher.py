# models/teacher/teacher_loader.py
import sys
sys.path.append('/content/drive/MyDrive/Adversarial-Diffusion-Distillation')
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from configs.model_config import ModelConfig

def load_teacher_model(device):
    pipe = StableDiffusionPipeline.from_pretrained(
        ModelConfig.TEACHER_ID, 
        torch_dtype=torch.float16 if "cuda" in device else torch.float16
    )
    pipe = pipe.to(device)

    # 1. Freeze Weights (Parameters won't change)
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    # 2. Enable Checkpointing (Memory saving)
    pipe.unet.enable_gradient_checkpointing()
    
    # 3. Set mode to train (Required for checkpointing to function actively)
    pipe.unet.train() 
    
    return pipe