# datasets/dataset_loader.py
import torch
from torch.utils.data import Dataset
import json
import os

class TeacherDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.prompts = json.load(open(os.path.join(data_dir, "prompts.json")))
        self.latents = sorted(os.listdir(os.path.join(data_dir, "teacher_latents")))
        self.images = sorted(os.listdir(os.path.join(data_dir, "teacher_images")))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        # load latent & image
        latent = torch.load(os.path.join(self.data_dir, "teacher_latents", self.latents[idx]))
        image = torch.load(os.path.join(self.data_dir, "teacher_images", self.images[idx]))
        prompt = self.prompts[idx]
        return latent, image, prompt
