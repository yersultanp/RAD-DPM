# main_train.py
import torch
import sys
sys.path.append('/content/drive/MyDrive/Adversarial-Diffusion-Distillation')
from configs.model_config import ModelConfig
from configs.train_config import TrainConfig
from configs.scheduler_config import SchedulerConfig
from models.teacher import load_teacher_model
from models.student import RobustLearnedScheduler
from ddim_utils import DifferentiableDiffusionHandler
from train.train_step import train_one_step, generate_teacher_target
import torch.nn.functional as F
from losses.image_loss import image_loss
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from visualize.visualize_training import visualize_schedule

def main():
    student = RobustLearnedScheduler(SchedulerConfig.K_STEPS).to(TrainConfig.DEVICE)
    optimizer = torch.optim.AdamW(student.parameters(), lr=TrainConfig.LEARNING_RATE)
    pipe = load_teacher_model(TrainConfig.DEVICE)
    scaler = GradScaler() # Vital for float16 training stability
    diff_handler = DifferentiableDiffusionHandler(pipe)

    loss_history = []
    schedule_history = []

    print("Starting Training...")
    pbar = tqdm(range(TrainConfig.EPOCHS))

    # 3. Dummy Data (Replace with datasets/dataset_loader.py in real version)
    prompts = [
        "A futuristic city with flying cars",
        "A cute corgi running in a field"
    ]
    train_data = generate_teacher_target(pipe, prompts, TrainConfig.DEVICE)

    for epoch in pbar:
        epoch_loss = 0
        current_schedule = []

        for i, data in enumerate(train_data):
            text_emb = data["emb"]
            student_latents = data["noise"].clone().requires_grad_(True)
            target_latents = data["target"]

            optimizer.zero_grad()

            # === The Magic Wrapper for Float16 ===
            with autocast():
                # 1. Run Student Loop
                for k in range(SchedulerConfig.K_STEPS):
                    # Student MLP runs in fp32 inside, but accepts mixed inputs via autocast
                    t_curr = student(k, student_latents)

                    if i == 0: current_schedule.append(t_curr.item())

                    if k < SchedulerConfig.K_STEPS - 1:
                        t_next = student(k+1, student_latents)
                        t_next = torch.min(t_next, t_curr - 1).clamp(min=0)
                    else:
                        t_next = torch.zeros_like(t_curr)

                    student_latents = diff_handler.step(student_latents, t_curr, t_next, text_emb)

                # 2. Compute Loss (in fp16/fp32 automatically handled)
                loss = F.mse_loss(student_latents, target_latents)

            # 3. Backprop with Scaler
            scaler.scale(loss).backward()

            # Unscale grads to clip them
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_data)
        loss_history.append(avg_loss)
        schedule_history.append(current_schedule)
        pbar.set_description(f"Loss: {avg_loss:.4f}")

    print("Training Complete!")

    # 4. Visualize (Optional)
    visualize_schedule(student, diff_handler, pipe, train_data, loss_history, schedule_history, SchedulerConfig.K_STEPS)

    # 5. Save (Optional)
    torch.save(student.state_dict(), "student_scheduler.pth")
    print("Done!")

if __name__ == "__main__":
    main()
