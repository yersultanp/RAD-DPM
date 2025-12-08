import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import matplotlib.pyplot as plt

# 1. Setup Evaluation Metrics
# LPIPS mimics human perception of similarity
lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to("cuda")

def evaluate_method(pipe, prompts, method_name="Method"):
    """
    Generates images for a list of prompts and returns the images.
    """
    images = []
    print(f"Running {method_name}...")
    for p in prompts:
        # Generate and convert to tensor [0, 1] for LPIPS
        img = pipe(p, output_type="pt").images[0]
        images.append(img)
    return torch.stack(images).to("cuda")

# 2. Define Experiment
prompts = [
    "A cyberpunk city street at night with neon lights",
    "A portrait of a cute cat in a garden",
    "A delicious bowl of ramen with chopsticks",
    "An astronaut riding a horse on Mars"
]

# A. Generate Ground Truth (Teacher 50 Steps)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
teacher_imgs = []
for p in prompts:
    # High quality reference
    img = pipe(p, num_inference_steps=50, output_type="pt").images[0]
    teacher_imgs.append(img)
teacher_imgs = torch.stack(teacher_imgs).to("cuda")

# B. Generate Baseline (DPM-Solver 4 Steps)
# This is currently the best "fixed" solver for low steps
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# Note: DPM-Solver often needs at least 10-20 steps to be good.
# At 4 steps, it usually struggles. This is your chance to win.
baseline_imgs = []
for p in prompts:
    img = pipe(p, num_inference_steps=4, output_type="pt").images[0]
    baseline_imgs.append(img)
baseline_imgs = torch.stack(baseline_imgs).to("cuda")

# C. Generate Your Student (4 Steps)
# (Assuming 'student' and 'diff_handler' are loaded from your training code)
student_imgs = []
with torch.no_grad():
    for p in prompts:
        # ... Insert your inference code here ...
        # Run the loop using student(k, latents)
        # For this example, I'll assume you have a function `run_student_inference(p)`
        # img = run_student_inference(p)

        # Placeholder for demo: Let's assume student is slightly better than baseline
        img = baseline_imgs[0] # REPLACE THIS with your actual student output
        student_imgs.append(img)
student_imgs = torch.stack(student_imgs).to("cuda")

# 3. Compute Scores
# Compare everyone to Teacher
score_baseline = lpips_metric(baseline_imgs, teacher_imgs).item()
score_student = lpips_metric(student_imgs, teacher_imgs).item()

print(f"LPIPS Score (Lower is Better):")
print(f"Baseline (DPM-Solver 4 steps): {score_baseline:.4f}")
print(f"Student  (Learned 4 steps):    {score_student:.4f}")

# 4. Visual Plot
fig, axs = plt.subplots(len(prompts), 3, figsize=(15, 5 * len(prompts)))
axs[0, 0].set_title("Teacher (50 Steps)")
axs[0, 1].set_title("DPM-Solver (4 Steps)")
axs[0, 2].set_title("Your Student (4 Steps)")

for i in range(len(prompts)):
    # Teacher
    axs[i, 0].imshow(teacher_imgs[i].cpu().permute(1, 2, 0))
    axs[i, 0].axis("off")

    # Baseline
    axs[i, 1].imshow(baseline_imgs[i].cpu().permute(1, 2, 0))
    axs[i, 1].axis("off")

    # Student
    axs[i, 2].imshow(student_imgs[i].cpu().permute(1, 2, 0))
    axs[i, 2].axis("off")

plt.tight_layout()
plt.show()
