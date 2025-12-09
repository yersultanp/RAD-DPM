import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from torch.cuda.amp import autocast
from configs.train_config import TrainConfig

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

def plot_scheduler_training_history(loss_history, schedule_history, save_dir="./results"):
    """
    Plots the training loss and the evolution of the learned timesteps.

    Args:
        loss_history (list): List of average loss per epoch.
        schedule_history (list): List of lists, where each inner list contains
                                 the [t_0, t_1, ..., t_k] values for that epoch.
        save_dir (str): Directory to save the plot.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    epochs = range(1, len(loss_history) + 1)

    # Convert schedule history to numpy for easy slicing
    # Shape: [Num_Epochs, K_STEPS]
    schedule_array = np.array(schedule_history)
    K_STEPS = schedule_array.shape[1]

    plt.figure(figsize=(14, 6))

    # --- Plot 1: Loss Curve ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_history, 'b-', linewidth=2, label="Training Loss")
    plt.title("Scheduler Loss Convergence", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (Hybrid)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # --- Plot 2: Schedule Evolution ---
    plt.subplot(1, 2, 2)
    colors = plt.cm.viridis(np.linspace(0, 1, K_STEPS))

    for k in range(K_STEPS):
        plt.plot(epochs, schedule_array[:, k],
                 label=f"Step {k}",
                 color=colors[k],
                 linewidth=2)

    plt.title("Evolution of Learned Timesteps", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Timestep Value (0-1000)", fontsize=12)
    plt.ylim(-50, 1050) # Keep bounds visible
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    plt.tight_layout()

    # Save and Show
    save_path = os.path.join(save_dir, "scheduler_training_progress.png")
    plt.savefig(save_path, dpi=150)
    print(f"Training plots saved to {save_path}")
    plt.show()
