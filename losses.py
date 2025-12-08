# losses/image_loss.py
import torch.nn.functional as F

def image_loss(student_img, teacher_img):
    return F.mse_loss(student_img, teacher_img)
