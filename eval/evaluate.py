import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch.cuda.amp import autocast

# Setup Metric
lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to("cuda")

def evaluation_pipeline(pipe, diff_handler, student, prompts, K_STEPS, DEVICE="cuda"):

    # Standardize Seed
    SEED = 42
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    print("Generating Teacher (50 Steps) & Baseline (4 Steps)...")

    # CRITICAL FIX 1: Ensure LoRA is DISABLED for Teacher and Baseline
    # We want the pure SD1.5 performance for comparison
    if hasattr(pipe.unet, "disable_adapter_layers"):
        pipe.unet.disable_adapter_layers()

    # ============================================
    # A. Generate Ground Truth (Teacher 50 Steps)
    # ============================================
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    teacher_imgs = []

    for p in prompts:
        img = pipe(p, num_inference_steps=50, generator=generator, output_type="pt").images[0]
        teacher_imgs.append(img)
    teacher_imgs = torch.stack(teacher_imgs).to(DEVICE)

    # Reset Generator
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

    # Reset Generator
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    # ============================================
    # C. Generate Your Student (4 Steps)
    # ============================================
    print("Generating Student (Learned Schedule + LoRA Refiner)...")
    student_imgs = []
    null_emb = pipe.text_encoder(
    pipe.tokenizer("", return_tensors="pt", padding="max_length", truncation=True).input_ids.to(DEVICE)
    )[0]
    encoded_prompts = []
    with torch.no_grad():
        for p in prompts:
            inputs = pipe.tokenizer(p, return_tensors="pt", padding="max_length", truncation=True).to(DEVICE)
            encoded_prompts.append(pipe.text_encoder(inputs.input_ids)[0])

    with torch.no_grad():
        with autocast():
            for i, text_emb in enumerate(encoded_prompts):
                cfg_text_emb = torch.cat([null_emb, text_emb])
                latents = randn_tensor((1, 4, 64, 64), device=DEVICE, generator=generator, dtype=text_emb.dtype)

                # 2. Run the Student Loop
                for k in range(K_STEPS):

                    # CRITICAL FIX 2: "The Switch"
                    # Steps 0 to K-2: Coarse Generation (LoRA OFF)
                    # Step K-1 (Final): Refinement (LoRA ON)
                    if k == K_STEPS - 1:
                        pipe.unet.enable_adapter_layers() # Activate Refiner
                    else:
                        pipe.unet.disable_adapter_layers() # Use Frozen Teacher logic

                    t_curr = student(k, latents)

                    if k < K_STEPS - 1:
                        t_next = student(k+1, latents)
                        t_next = torch.min(t_next, t_curr - 1).clamp(min=0)
                    else:
                        t_next = torch.zeros_like(t_curr)

                    # Differentiable Step (Uses pipe.unet, which now has adapters toggled correctly)
                    latents = diff_handler.step(latents, t_curr, t_next, cfg_text_emb, guidance_scale = 7.5)

                # 3. Decode Logic
                latents = 1 / 0.18215 * latents
                img = pipe.vae.decode(latents).sample
                img = (img / 2 + 0.5).clamp(0, 1)
                student_imgs.append(img.squeeze(0))

    student_imgs = torch.stack(student_imgs).to(DEVICE)

    # ============================================
    # 3. Compute Scores
    # ============================================
    score_baseline = lpips_metric(baseline_imgs, teacher_imgs).item()
    score_student = lpips_metric(student_imgs, teacher_imgs).item()

    print(f"\nResults over {len(prompts)} prompts:")
    print(f"LPIPS Score (Lower is Better):")
    print(f"Baseline (DPM-Solver 4 steps): {score_baseline:.4f}")
    print(f"Student  (Learned 4 steps):    {score_student:.4f}")

    # ============================================
    # 4. Visual Plot
    # ============================================
    t_imgs_cpu = teacher_imgs.cpu().permute(0, 2, 3, 1).float().numpy()
    b_imgs_cpu = baseline_imgs.cpu().permute(0, 2, 3, 1).float().numpy()
    s_imgs_cpu = student_imgs.cpu().permute(0, 2, 3, 1).float().numpy()

    fig, axs = plt.subplots(len(prompts), 3, figsize=(12, 4 * len(prompts)))
    if len(prompts) == 1: axs = axs.reshape(1, -1)

    axs[0, 0].set_title("Teacher (50 Steps)")
    axs[0, 1].set_title("DPM-Solver (4 Steps)")
    axs[0, 2].set_title("Student (4 Steps)")

    for i in range(len(prompts)):
        axs[i, 0].imshow(t_imgs_cpu[i]); axs[i, 0].axis("off")
        axs[i, 1].imshow(b_imgs_cpu[i]); axs[i, 1].axis("off")
        axs[i, 2].imshow(s_imgs_cpu[i]); axs[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("./results/final_evaluation_grid.png")
    plt.show()
