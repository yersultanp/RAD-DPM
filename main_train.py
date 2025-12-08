# main_train.py
import torch
import sys
import random
sys.path.append('/content/drive/MyDrive/Adversarial-Diffusion-Distillation')
from configs.model_config import ModelConfig
from configs.train_config import TrainConfig
from configs.scheduler_config import SchedulerConfig
from models.teacher import load_teacher_model
from models.student import RobustLearnedScheduler
from ddim_utils import DifferentiableDiffusionHandler
from train.train_step import train_one_step, generate_teacher_target
import torch.nn.functional as F
from losses import image_loss
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from eval.visualize_schedule import visualize_scheduling_results
from eval.evaluate import evaluation_pipeline
from losses import HybridLatentLoss

def main():
    student = RobustLearnedScheduler(SchedulerConfig.K_STEPS).to(TrainConfig.DEVICE)
    optimizer = torch.optim.AdamW(student.parameters(), lr=TrainConfig.LEARNING_RATE)
    pipe = load_teacher_model(TrainConfig.DEVICE)
    scaler = GradScaler() # Vital for float16 training stability
    diff_handler = DifferentiableDiffusionHandler(pipe)
    loss_fn = HybridLatentLoss()

    loss_history = []
    schedule_history = []

    print("Starting Training...")
    pbar = tqdm(range(TrainConfig.EPOCHS))

    # 3. Dummy Data (Replace with datasets/dataset_loader.py in real version)
    TRAIN_PROMPTS = [
    # --- Animals & Textures (High Frequency Noise) ---
    "A macro photograph of a bumblebee on a yellow flower",
    "A close-up portrait of a snowy owl with detailed feathers",
    "A cute corgi running through a field of tall green grass",
    "A wet otter swimming in a river, 4k high resolution",
    "A detailed close-up of a dragon eye with scales",

    # --- Landscapes & Environments (Depth & Atmosphere) ---
    "A futuristic cyberpunk city at night with neon rain",
    "A serene misty mountain lake at sunrise, landscape photography",
    "A dusty desert road stretching into the horizon",
    "A dense tropical rainforest with sunlight filtering through leaves",
    "A snowy cabin in the woods during a winter storm",

    # --- Objects & Geometry (Structure & Edges) ---
    "A delicious pepperoni pizza with melting cheese",
    "A vintage typewriter sitting on an old wooden desk",
    "A shiny red sports car driving on a coastal highway",
    "A steaming cup of coffee with latte art on a saucer",
    "A stack of old leather-bound books in a library",

    # --- Portraits (Anatomy & Skin Tones) ---
    "A portrait of an old fisherman with a weathered face",
    "A professional headshot of a smiling woman in business attire",
    "A renaissance style oil painting of a young princess",
    "A cyberpunk android girl with glowing circuitry on her face",
    "A pencil sketch of a bearded man, rough charcoal style",

    # --- Art Styles (Non-Photorealistic Distributions) ---
    "An impressionist oil painting of a water lily pond",
    "A flat vector illustration of a rocket ship launching",
    "A watercolor painting of a cozy cafe in Paris",
    "A low poly 3d render of a deer in a forest",
    "A pop art style poster of a banana, warhol style",
    ]
    train_data = generate_teacher_target(pipe, TRAIN_PROMPTS, TrainConfig.DEVICE)

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
                loss = loss_fn(student_latents, target_latents)

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
    k = 8
    k_idx = random.sample([i for i in range(len(TRAIN_PROMPTS))], k)
    for i in k_idx:
      visualize_scheduling_results(student, diff_handler, pipe, train_data, loss_history, schedule_history, SchedulerConfig.K_STEPS, idx = i)


    # 5. Evaluate based on the LPIPS Score
    EVAL_PROMPTS = [
    # --- Category 1: Texture Stress Test ---
    # Low-step solvers often produce "plastic" or smooth textures. 
    # If your scheduler works, these should look fuzzy/detailed, not flat.
    "A extreme close-up of a fluffy cat, highly detailed fur texture",
    "A field of tall grass blowing in the wind, oil painting style",

    # --- Category 2: Lighting & Contrast ---
    # These require the model to resolve values (light vs dark) early in the schedule.
    "A cyberpunk city street at night with wet pavement and neon lights",
    "A silhouette of a cowboy riding against a bright sunset",
    
    # --- Category 3: Geometry & Structure ---
    # These test if the scheduler can "lock in" shapes quickly.
    # Bad schedules will make the cube look wobbly or the lines disconnected.
    "A transparent glass cube sitting on a wooden table, raytracing",
    "A modern architectural building with sharp concrete edges",

    # --- Category 4: Compositional Complexity ---
    # Multiple objects often confuse low-step solvers (objects blend together).
    "A bowl of fresh fruit containing apples, bananas, and grapes",
    "An astronaut riding a horse on Mars",

    # --- Category 5: Abstract/Artistic ---
    # Tests if the scheduler can handle non-photorealistic noise distributions.
    "Abstract geometric shapes, bauhaus style poster, red and blue",
    "A swirling galaxy in deep space, digital art"
    ]
    evaluation_pipeline(pipe, diff_handler, student, EVAL_PROMPTS, SchedulerConfig.K_STEPS, DEVICE=ModelConfig.DEVICE)

    # 5. Save (Optional)
    torch.save(student.state_dict(), "student_scheduler.pth")
    print("Done!")

if __name__ == "__main__":
    main()
