# configs/train_config.py

class TrainConfig:
    learning_rate = 1e-4
    batch_size = 1
    num_steps = 5000
    K = 4  # number of student steps
    log_interval = 50
    save_interval = 500
    device = "cuda"
