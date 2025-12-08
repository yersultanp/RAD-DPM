# configs/model_config.py

class ModelConfig:
    TEACHER_ID = "sd-legacy/stable-diffusion-v1-5"
    STUDENT_STEPS = 4   # K
    LATENT_DIM = 4
    SCHEDULER_HIDDEN_DIM = 64
    SCHEDULER_EMBED_DIM = 32