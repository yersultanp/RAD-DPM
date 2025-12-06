# train/train_scheduler_only.py
from train.train_step import train_step
import torch

def train_scheduler(dataset, models, config):
    for step in range(config.num_steps):
        random_index = lambda: torch.randint(0, len(dataset), (1,)).item()
        batch = dataset[random_index()]
        loss = train_step(batch, models.scheduler, models.teacher, models.optimizer)
        if step % config.log_interval == 0:
            print(f"step {step}, loss {loss.item()}")
