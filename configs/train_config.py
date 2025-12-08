# configs/train_config.py
import torch

class TrainConfig:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE_MODEL = torch.float16 if DEVICE == "cuda" else torch.float32
    EPOCHS = 25
    LEARNING_RATE = 2e-4
