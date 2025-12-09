import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda.amp import autocast

def visualize_scheduling_results(student, diff_handler, pipe, train_data, loss_history, schedule_history, K_STEPS, idx = 0):

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.title("MSE Loss (Mixed Precision)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    schedule_array = np.array(schedule_history)
    for step_k in range(K_STEPS):
        plt.plot(schedule_array[:, step_k], label=f"Step {step_k}")
    plt.title("Learned Timestep Evolution")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./results/scheduling_prompt_{idx}")

    print("Generating Final Comparison...")
    with torch.no_grad():
        with autocast():
            null_inputs = pipe.tokenizer("", return_tensors="pt", padding="max_length", truncation=True).to(TrainConfig.DEVICE)
            null_emb = pipe.text_encoder(null_inputs.input_ids)[0]
            data = train_data[idx]
            latents = data["noise"].clone()
            for k in range(K_STEPS):
                t_curr = student(k, latents)
                if k < K_STEPS - 1:
                    t_next = student(k+1, latents)
                    t_next = torch.min(t_next, t_curr - 1).clamp(min=0)
                else:
                    t_next = torch.zeros_like(t_curr)
                cfg_text_emb = torch.cat([null_emb, data["emb"]])
                latents = diff_handler.step(latents, t_curr, t_next, cfg_text_emb, guidance_scale = 7.5)

            # Decode (VAE expects fp16, handled by autocast)
            latents = 1 / 0.18215 * latents
            img_student = pipe.vae.decode(latents).sample
            img_student = (img_student / 2 + 0.5).clamp(0, 1).float().cpu().permute(0,2,3,1).numpy()[0]

            tgt_lat = 1 / 0.18215 * data["target"]
            img_teacher = pipe.vae.decode(tgt_lat).sample
            img_teacher = (img_teacher / 2 + 0.5).clamp(0, 1).float().cpu().permute(0,2,3,1).numpy()[0]

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plt.imshow(img_teacher); plt.title("Teacher"); plt.axis("off")
        plt.subplot(1, 2, 2); plt.imshow(img_student); plt.title("Student"); plt.axis("off")
    plt.savefig(f"./results/target_compare_prompt_{idx}")
