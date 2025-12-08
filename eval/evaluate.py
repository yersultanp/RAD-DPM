import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch.cuda.amp import autocast

# Setup Metric
lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to("cuda")

def evaluation_pipeline(pipe, diff_handler, student, prompts, K_STEPS, DEVICE="cuda"):
    
    # Standardize Seed for fair comparison
    # We use a specific generator so Teacher, Baseline, and Student get the SAME noise
    SEED = 42
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    
    print("Generating Teacher (50 Steps) & Baseline (4 Steps)...")
    
    # ============================================
    # A. Generate Ground Truth (Teacher 50 Steps)
    # ============================================
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    teacher_imgs = []
    
    for p in prompts:
        # We pass generator to ensure reproducibility
        img = pipe(p, num_inference_steps=50, generator=generator, output_type="pt").images[0]
        teacher_imgs.append(img)
    teacher_imgs = torch.stack(teacher_imgs).to(DEVICE)

    # Reset Generator for Baseline so it gets the SAME noise as Teacher
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    # ============================================
    # B. Generate Baseline (DPM-Solver 4 Steps)
    # ============================================
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    baseline_imgs = []
    
    for p in prompts:
        img = pipe(p, num_inference_steps=4, generator=generator, output_type="pt").images[0]
        baseline_imgs.append(img)
    baseline_imgs = torch.stack(baseline_imgs).to(DEVICE)

    # Reset Generator for Student
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    # ============================================
    # C. Generate Your Student (4 Steps)
    # ============================================
    print("Generating Student (Learned Schedule)...")
    student_imgs = []
    
    # Encode all prompts first to save time/memory
    encoded_prompts = []
    with torch.no_grad():
        for p in prompts:
            inputs = pipe.tokenizer(p, return_tensors="pt", padding="max_length", truncation=True).to(DEVICE)
            encoded_prompts.append(pipe.text_encoder(inputs.input_ids)[0])

    with torch.no_grad():
        with autocast():
            for i, text_emb in enumerate(encoded_prompts):
                # 1. Create EXACT SAME starting noise as Teacher/Baseline
                # (Batch size 1, 4 channels, 64x64)
                latents = randn_tensor((1, 4, 64, 64), device=DEVICE, generator=generator, dtype=text_emb.dtype)
                
                # 2. Run the Student Loop
                for k in range(K_STEPS):
                    # Predict t (Make sure to pass latents, not student_latents which was undefined)
                    t_curr = student(k, latents)

                    # Lookahead
                    if k < K_STEPS - 1:
                        t_next = student(k+1, latents)
                        t_next = torch.min(t_next, t_curr - 1).clamp(min=0)
                    else:
                        t_next = torch.zeros_like(t_curr)

                    # Differentiable Step
                    latents = diff_handler.step(latents, t_curr, t_next, text_emb)

                # 3. Decode Logic (The missing part!)
                # Scale latents
                latents = 1 / 0.18215 * latents
                img = pipe.vae.decode(latents).sample
                
                # Normalize to [0, 1]
                img = (img / 2 + 0.5).clamp(0, 1)
                
                # Keep on GPU for LPIPS calculation
                student_imgs.append(img.squeeze(0))

    student_imgs = torch.stack(student_imgs).to(DEVICE)

    # ============================================
    # 3. Compute Scores
    # ============================================
    # LPIPS expects input in range [0, 1]
    score_baseline = lpips_metric(baseline_imgs, teacher_imgs).item()
    score_student = lpips_metric(student_imgs, teacher_imgs).item()

    print(f"\nResults over {len(prompts)} prompts:")
    print(f"LPIPS Score (Lower is Better):")
    print(f"Baseline (DPM-Solver 4 steps): {score_baseline:.4f}")
    print(f"Student  (Learned 4 steps):    {score_student:.4f}")

    # ============================================
    # 4. Visual Plot
    # ============================================
    # Move to CPU for plotting
    t_imgs_cpu = teacher_imgs.cpu().permute(0, 2, 3, 1).float().numpy()
    b_imgs_cpu = baseline_imgs.cpu().permute(0, 2, 3, 1).float().numpy()
    s_imgs_cpu = student_imgs.cpu().permute(0, 2, 3, 1).float().numpy()

    fig, axs = plt.subplots(len(prompts), 3, figsize=(12, 4 * len(prompts)))
    
    # Handle single row case
    if len(prompts) == 1:
        axs = axs.reshape(1, -1)

    # Set column titles
    axs[0, 0].set_title("Teacher (50 Steps)")
    axs[0, 1].set_title("DPM-Solver (4 Steps)")
    axs[0, 2].set_title("Student (4 Steps)")

    for i in range(len(prompts)):
        # Teacher
        axs[i, 0].imshow(t_imgs_cpu[i])
        axs[i, 0].axis("off")
        
        # Baseline
        axs[i, 1].imshow(b_imgs_cpu[i])
        axs[i, 1].axis("off")
        
        # Student
        axs[i, 2].imshow(s_imgs_cpu[i])
        axs[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("./results/final_evaluation_grid.png")
    plt.show()
