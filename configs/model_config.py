# configs/model_config.py

class ModelConfig:
    scheduler_hidden_dim = 128
    scheduler_num_layers = 2
    latent_stat_dim = 4  # mean, std, energy, etc.
    student_unet = False  # start without UNet student
