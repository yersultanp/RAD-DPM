import torch
import matplotlib.pyplot as plt
import numpy as np
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch.cuda.amp import autocast

# Initialize Metric
lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to("cuda")

def comparison_pipeline(pipe, dpm_handler, student, prompts, K_STEPS=4, DEVICE="cuda", save_dir = "./final_results/comparison_img.png"):
    """
    Generates a 4-way comparison grid:
    1. Teacher (50 step DDIM)
    2. Simple DPM (4 step, fixed schedule)
    3. Scheduled (4 step, learned schedule, No LoRA)
    4. Scheduled + LoRA (4 step, learned schedule, Refiner at end)
    
    Prints natural vs. learned schedules for every prompt.
    """
    def decode_latents(pipe, latents):
        """Helper to decode latents into a displayable image."""
        latents = 1 / 0.18215 * latents
        img = pipe.vae.decode(latents).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.squeeze(0)
    
    SEED = 42
    results = {k: [] for k in ["Teacher", "Simple DPM", "Scheduled", "Scheduled + LoRA"]}

    print(f"\nStarting 4-Way Evaluation on {len(prompts)} prompts...")

    # 1. Get "Natural" DPM Schedule for comparison
    dpm_sched = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    dpm_sched.set_timesteps(K_STEPS)
    # Add t=0 implicitly for comparison with student's explicit end point
    natural_steps = dpm_sched.timesteps.tolist() + [0]
    print(f"Natural DPM Schedule (for {K_STEPS} steps): {natural_steps}\n")
    
    # Pre-compute Null Embedding for CFG
    null_emb = pipe.text_encoder(
        pipe.tokenizer("", return_tensors="pt", padding="max_length", truncation=True).input_ids.to(DEVICE)
    )[0]

    # --- MAIN EVALUATION LOOP ---
    for i, prompt in enumerate(prompts):
        print(f"Processing Prompt {i+1}/{len(prompts)}: '{prompt[:30]}...'")
        
        # Prepare embeddings and common noise generators
        g_seed = torch.Generator(device=DEVICE).manual_seed(SEED + i)
        text_input = pipe.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True).to(DEVICE)
        text_emb = pipe.text_encoder(text_input.input_ids)[0]
        cfg_text_emb = torch.cat([null_emb, text_emb])
        
        # Generate identical initial noise for all methods
        init_latents = randn_tensor((1, 4, 64, 64), device=DEVICE, generator=g_seed, dtype=text_emb.dtype)

        with torch.no_grad(), autocast():
            # ============================================
            # 1. Teacher (50 Steps DDIM)
            # ============================================
            if hasattr(pipe.unet, "disable_adapter_layers"): pipe.unet.disable_adapter_layers()
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            # Use a fresh generator for the pipeline call to ensure deterministic noise
            g_pipe = torch.Generator(device=DEVICE).manual_seed(SEED + i)
            img = pipe(prompt, num_inference_steps=50, generator=g_pipe, output_type="pt").images[0]
            results["Teacher"].append(img)

            # ============================================
            # 2. Simple DPM (4 Natural Steps)
            # ============================================
            if hasattr(pipe.unet, "disable_adapter_layers"): pipe.unet.disable_adapter_layers()
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=False
            )
            pipe.scheduler.set_timesteps(K_STEPS)
            natural_steps = pipe.scheduler.timesteps.tolist() + [0]
            print(f"Simple DPM Schedule (for {K_STEPS} steps): {natural_steps}\n")
            
            g_pipe = torch.Generator(device=DEVICE).manual_seed(SEED + i)
            img = pipe(prompt, num_inference_steps=K_STEPS, generator=g_pipe, output_type="pt").images[0]
            results["Simple DPM"].append(img)

            # ============================================
            # 3. Scheduled (Learned Steps, NO LoRA)
            # ============================================
            if hasattr(pipe.unet, "disable_adapter_layers"): pipe.unet.disable_adapter_layers()
            
            latents = init_latents.clone()
            t_curr = torch.full((1, 1), 1000.0, device=DEVICE)
            hx = None; prev_noise = None; prev_lambda = None
            learned_steps = [t_curr.item()]
            
            for k in range(K_STEPS):
                t_next, hx = student(latents, t_curr, hx)
                if k < K_STEPS - 1: t_next = torch.min(t_next, t_curr - 1).clamp(min=50)
                else: t_next = torch.zeros_like(t_curr)
                
                learned_steps.append(t_next.item())

                latents, prev_noise, prev_lambda = dpm_handler.step(
                    latents, t_curr, t_next, cfg_text_emb,
                    prev_noise_pred=prev_noise, prev_lambda=prev_lambda,
                    guidance_scale=5.0
                )
                t_curr = t_next
            
            results["Scheduled"].append(decode_latents(pipe, latents))
            print(f"  -> Learned Schedule (No LoRA): {[int(t) for t in learned_steps]}")

            # ============================================
            # 4. Scheduled + LoRA (Refiner at End)
            # ============================================
            # Reset state for a clean run
            latents = init_latents.clone()
            t_curr = torch.full((1, 1), 1000.0, device=DEVICE)
            hx = None; prev_noise = None; prev_lambda = None
            # Schedule might change slightly because LoRA changes latents at the last step
            learned_steps_lora = [t_curr.item()] 
            LORA_SCALE = 0.6
            for k in range(K_STEPS):
                # Toggle LoRA: ON only for the final step
                if k == K_STEPS - 1:
                    guidance = 1.5
                    if hasattr(pipe.unet, "enable_adapter_layers"): 
                        pipe.unet.enable_adapter_layers()
                        try:
                            pipe.unet.set_adapter("default", adapter_weights=[LORA_SCALE])
                        except:
                            pass # Fallback if set_adapter not supported
                else:
                    guidance = 4.0
                    if hasattr(pipe.unet, "disable_adapter_layers"): pipe.unet.disable_adapter_layers()

                t_next, hx = student(latents, t_curr, hx)
                if k < K_STEPS - 1: t_next = torch.min(t_next, t_curr - 1).clamp(min=50)
                else: t_next = torch.zeros_like(t_curr)
                
                learned_steps_lora.append(t_next.item())

                latents, prev_noise, prev_lambda = dpm_handler.step(
                    latents, t_curr, t_next, cfg_text_emb,
                    prev_noise_pred=prev_noise, prev_lambda=prev_lambda,
                    guidance_scale=guidance
                )
                t_curr = t_next
            
            results["Scheduled + LoRA"].append(decode_latents(pipe, latents))
            # print(f"  -> Learned Schedule (+ LoRA):  {[int(t) for t in learned_steps_lora]}")

    # --- Metrics & Visualization ---
    print("\nComputing LPIPS Scores...")
    for k, v in results.items(): results[k] = torch.stack(v).to(DEVICE)
    
    scores = {}
    for method in ["Simple DPM", "Scheduled", "Scheduled + LoRA"]:
        scores[method] = lpips_metric(results[method], results["Teacher"]).item()
        
    print(f"\n=== LPIPS Results (Lower is Better) ===")
    print(f"Simple DPM (Baseline):   {scores['Simple DPM']:.4f}")
    print(f"Scheduled (Ablation):    {scores['Scheduled']:.4f}")
    print(f"Scheduled + LoRA (Full): {scores['Scheduled + LoRA']:.4f}")

    print("Generating 4-Column Grid Plot...")
    cols = ["Teacher", "Simple DPM", "Scheduled", "Scheduled + LoRA"]
    n_prompts = len(prompts)
    fig, axs = plt.subplots(n_prompts, 4, figsize=(16, 4 * n_prompts))
    if n_prompts == 1: axs = axs.reshape(1, -1)

    for j, col in enumerate(cols): axs[0, j].set_title(col, fontsize=12, fontweight='bold')

    for i in range(n_prompts):
        for j, method in enumerate(cols):
            img = results[method][i].float().cpu().permute(1, 2, 0).numpy().astype(np.float32)
            axs[i, j].imshow(img)
            axs[i, j].axis("off")
            
    plt.tight_layout()
    plt.savefig(save_dir)
    plt.show()
    plt.close()


