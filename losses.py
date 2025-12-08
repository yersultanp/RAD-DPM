# losses/image_loss.py
import torch.nn.functional as F
import torch.nn as nn

def image_loss(student_img, teacher_img):
    return F.mse_loss(student_img, teacher_img)


class HybridLatentLoss(nn.Module):
    def __init__(self, alpha_mse=1.0, alpha_cos=0.5, alpha_stats=0.5):
        super().__init__()
        self.alpha_mse = alpha_mse     # Precision
        self.alpha_cos = alpha_cos     # Semantic Direction
        self.alpha_stats = alpha_stats # Texture/Contrast

    def forward(self, student_latents, teacher_latents):
        # 1. MSE (Standard Euclidean Distance)
        loss_mse = F.mse_loss(student_latents, teacher_latents)
        
        # 2. Cosine Similarity (Directional Alignment)
        # Flatten [B, 4, 64, 64] -> [B, 16384]
        flat_s = student_latents.view(student_latents.shape[0], -1)
        flat_t = teacher_latents.view(teacher_latents.shape[0], -1)
        
        # Cosine embedding loss expects target 1 (match) or -1 (oppose)
        # We want match (1.0). Loss is 1 - cos(theta)
        loss_cos = 1.0 - F.cosine_similarity(flat_s, flat_t, dim=1).mean()
        
        # 3. Statistical Matching (Cheap "Style" Loss)
        # Matches the mean (brightness) and std (contrast/texture)
        s_mean = student_latents.mean(dim=[1, 2, 3])
        t_mean = teacher_latents.mean(dim=[1, 2, 3])
        s_std = student_latents.std(dim=[1, 2, 3])
        t_std = teacher_latents.std(dim=[1, 2, 3])
        
        loss_stats = F.mse_loss(s_mean, t_mean) + F.mse_loss(s_std, t_std)
        
        # Combine
        total_loss = (self.alpha_mse * loss_mse) + \
                     (self.alpha_cos * loss_cos) + \
                     (self.alpha_stats * loss_stats)
                     
        return total_loss