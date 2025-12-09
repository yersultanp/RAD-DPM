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

def plot_scheduler_training_history(loss_history, schedule_history, idx, save_dir="./results/"):
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
    save_path = os.path.join(save_dir, f"scheduler_training_progress_{idx}.png")
    plt.savefig(save_path, dpi=150)
    print(f"Training plots saved to {save_path}")
    plt.show()

import matplotlib.pyplot as plt
import torch
import numpy as np

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def analyze_schedule_variance(student, prompts, k_steps, device="cuda", save_dir="./final_results/"):
    # 1. Setup
    student.eval()
    trajectories = []
    
    # Ensure directory exists (Prevents FileNotFoundError)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print("Collecting trajectories...")
    
    # 2. Collection Loop
    # Note: We don't actually use the text prompts here, we just use the list 
    # to define 'how many' random noise samples to test. 
    # This correctly tests "Latent-based variance".
    for _ in range(len(prompts)):
        # Create random noise 
        latents = torch.randn(1, 4, 64, 64).to(device)
        
        # Initialize State
        t_curr = torch.full((1, 1), 1000.0, device=device)
        hx = None
        path = [1000]
        
        with torch.no_grad():
            for k in range(k_steps): 
                # RNN Prediction
                t_next, hx = student(latents, t_curr, hx)
                
                if k < k_steps - 1:
                    # Apply constraints (Must match training logic!)
                    max_allowed = t_curr - 10.0
                    t_next = torch.min(t_next, max_allowed).clamp(min=20.0)
                else:
                    # Force 0 at the end
                    t_next = torch.zeros_like(t_curr)
                
                path.append(t_next.item())
                t_curr = t_next
        
        trajectories.append(path)

    # 3. Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot individual lines (High transparency)
    trajectory_array = np.array(trajectories)
    
    # X-Axis is just 0, 1, 2... K
    steps_x = range(k_steps + 1)
    
    for i in range(len(trajectories)):
        plt.plot(steps_x, trajectory_array[i], color='blue', alpha=0.15)
        
    # Plot the Mean Path
    mean_path = np.mean(trajectory_array, axis=0)
    plt.plot(steps_x, mean_path, color='red', linewidth=3, marker='o', label="Mean Learned Schedule")
    
    # Plot Linear Baseline for comparison
    plt.plot(steps_x, np.linspace(1000, 0, k_steps + 1), 'k--', label="Linear Baseline")

    plt.title(f"Schedule Variance Analysis (K={k_steps})")
    plt.ylabel("Timestep (Noise Level)")
    plt.xlabel("Inference Step Index")
    plt.xticks(steps_x)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Save and Show
    save_path = os.path.join(save_dir, f"schedule_variance_analysis_k{k_steps}.png")
    plt.savefig(save_path, dpi=150)
    print(f"Schedule plots saved to {save_path}")
    plt.close() # Good practice to close figure to free memory
    
    return mean_path

# Append this to eval/visualize_schedule.py

def plot_refiner_history(loss_history, save_dir="./results"):
    """
    Plots the training loss for the LoRA Refiner.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    epochs = range(1, len(loss_history) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss_history, 'r-', linewidth=2, label="Refiner (MSE) Loss")
    
    plt.title("Refiner (LoRA) Convergence", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_path = os.path.join(save_dir, "refiner_training_loss.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Refiner plot saved to {save_path}")