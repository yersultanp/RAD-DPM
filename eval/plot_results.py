import torch
import matplotlib.pyplot as plt
import numpy as np
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch.cuda.amp import autocast

# Initialize Metric
lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to("cuda")

def compare_methods_pipeline(pipe, dpm_handler, student, prompts, K_STEPS=4, DEVICE="cuda", save_dir="./final_results/full_comparison_grid.png"):
    """
    4-Way Comparison Pipeline:
    1. Teacher (50 Steps)
    2. Simple DPM-Solver (4 Natural Steps)
    3. Scheduled DPM (4 Learned Steps, No LoRA)
    4. Scheduled DPM + LoRA (4 Learned Steps + Refiner)
    """

    # Standardize Seed for fair comparison
    SEED = 42

    # Containers for results
    results = {
        "Teacher": [],
        "Simple DPM": [],
        "Scheduled": [],
        "Scheduled + LoRA": []
    }

    # Helper to decode latents to images
    def decode_latents(latents):
        latents = 1 / 0.18215 * latents
        img = pipe.vae.decode(latents).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        return img

    print(f"Running evaluation on {len(prompts)} prompts...")

    # Pre-encode prompts to save time
    encoded_prompts = []
    with torch.no_grad():
        for p in prompts:
            inputs = pipe.tokenizer(p, return_tensors="pt", padding="max_length", truncation=True).to(DEVICE)
            encoded_prompts.append(pipe.text_encoder(inputs.input_ids)[0])

    # Loop over prompts
    for i, (prompt, text_emb) in enumerate(zip(prompts, encoded_prompts)):
        print(f"  Prompt {i+1}: {prompt[:30]}...")

        # Generator for this specific prompt (ensures same noise across methods)
        g = torch.Generator(device=DEVICE).manual_seed(SEED + i)

        # 1. TEACHER (50 Steps) ------------------------------------------------
        # Ensure LoRA is OFF
        if hasattr(pipe.unet, "disable_adapter_layers"): pipe.unet.disable_adapter_layers()
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        with torch.no_grad():
            img_teacher = pipe(prompt, num_inference_steps=50, generator=g, output_type="pt").images[0]
        results["Teacher"].append(img_teacher)

        # 2. SIMPLE DPM SOLVER (4 Natural Steps) -------------------------------
        # Ensure LoRA is OFF
        if hasattr(pipe.unet, "disable_adapter_layers"): pipe.unet.disable_adapter_layers()
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        # Reset Generator
        g = torch.Generator(device=DEVICE).manual_seed(SEED + i)
        with torch.no_grad():
            img_dpm = pipe(prompt, num_inference_steps=K_STEPS, generator=g, output_type="pt").images[0]
            natural_steps = pipe.scheduler.timesteps.tolist()
        results["Simple DPM"].append(img_dpm)

        print(f"    Natural DPM Steps: {natural_steps}")
        # 3. & 4. SCHEDULED METHODS (Shared Initial Loop) ----------------------
        # We run the custom loop twice: once with LoRA OFF, once with LoRA ON (at end)

        # Prepare Inputs for Custom Loop
        g = torch.Generator(device=DEVICE).manual_seed(SEED + i)
        init_latents = randn_tensor((1, 4, 64, 64), device=DEVICE, generator=g, dtype=text_emb.dtype)

        # --- Run Config 3: Scheduled (No LoRA) ---
        if hasattr(pipe.unet, "disable_adapter_layers"): pipe.unet.disable_adapter_layers()

        with torch.no_grad(), autocast():
            # Reset Loop State
            schedule_history = []
            latents = init_latents.clone()
            t_curr = torch.full((1, 1), 1000.0, device=DEVICE)
            schedule_history.append(t_curr.item())
            hx = None; prev_noise = None; prev_lambda = None

            for k in range(K_STEPS):
                # RNN Schedule Prediction
                t_next, hx = student(latents, t_curr, hx)
                if k < K_STEPS - 1:
                    t_next = torch.min(t_next, t_curr - 1).clamp(min=0)
                else:
                    t_next = torch.zeros_like(t_curr) # Force land
                schedule_history.append(t_next.item())
                # DPM Integration Step
                latents, prev_noise, prev_lambda = dpm_handler.step(
                    latents, t_curr, t_next, text_emb,
                    prev_noise_pred=prev_noise, prev_lambda=prev_lambda,
                    guidance_scale=7.5 # Use CFG for eval!
                )
                t_curr = t_next

            results["Scheduled"].append(decode_latents(latents).squeeze(0))

        print(f"    Learned Schedule: {[int(s) for s in schedule_history]}")
        # --- Run Config 4: Scheduled + LoRA ---
        # Reset State for fresh run
        if hasattr(pipe.unet, "disable_adapter_layers"): pipe.unet.disable_adapter_layers()

        with torch.no_grad(), autocast():
            latents = init_latents.clone()
            t_curr = torch.full((1, 1), 1000.0, device=DEVICE)
            hx = None; prev_noise = None; prev_lambda = None

            for k in range(K_STEPS):
                # RNN Schedule Prediction
                t_next, hx = student(latents, t_curr, hx)
                if k < K_STEPS - 1:
                    t_next = torch.min(t_next, t_curr - 1).clamp(min=0)
                else:
                    t_next = torch.zeros_like(t_curr)

                # CRITICAL: Enable LoRA ONLY for the final step
                if k == K_STEPS - 1:
                    pipe.unet.enable_adapter_layers()
                else:
                    pipe.unet.disable_adapter_layers()

                # DPM Integration Step
                latents, prev_noise, prev_lambda = dpm_handler.step(
                    latents, t_curr, t_next, text_emb,
                    prev_noise_pred=prev_noise, prev_lambda=prev_lambda,
                    guidance_scale=7.5
                )
                t_curr = t_next

            results["Scheduled + LoRA"].append(decode_latents(latents).squeeze(0))

    # --- METRICS CALCULATION ---
    print("\nComputing LPIPS Scores...")

    # Convert lists to tensors
    for key in results:
        results[key] = torch.stack(results[key]).to(DEVICE)

    # Calculate vs Teacher
    scores = {}
    for method in ["Simple DPM", "Scheduled", "Scheduled + LoRA"]:
        score = lpips_metric(results[method], results["Teacher"]).item()
        scores[method] = score

    print(f"\n=== LPIPS Results (Lower is Better) ===")
    print(f"Simple DPM (Baseline):   {scores['Simple DPM']:.4f}")
    print(f"Scheduled (Ablation):    {scores['Scheduled']:.4f}")
    print(f"Scheduled + LoRA (Full): {scores['Scheduled + LoRA']:.4f}")

    # --- VISUALIZATION ---
    print("Generating Grid Plot...")
    n_prompts = len(prompts)
    fig, axs = plt.subplots(n_prompts, 4, figsize=(16, 4 * n_prompts))

    if n_prompts == 1: axs = axs.reshape(1, -1)

    # Column Titles
    cols = ["Teacher", "Simple DPM", "Scheduled", "Sched + LoRA"]
    for ax, col in zip(axs[0], cols):
        ax.set_title(col, fontsize=14, fontweight='bold')

    # Plot Images
    for i in range(n_prompts):
        for j, method in enumerate(cols):
            # Convert to numpy [H, W, C]
            img = results[method][i].cpu().permute(1, 2, 0).numpy()
            axs[i, j].imshow(img)
            axs[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(save_dir)
    plt.show()

    return scores
