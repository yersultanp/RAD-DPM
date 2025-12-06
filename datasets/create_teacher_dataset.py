import torch
import os
import json
from diffusers import StableDiffusionPipeline, DDPMScheduler
from PIL import Image
import argparse
from tqdm import tqdm
import sys
sys.path.append('/content/drive/MyDrive/Adversarial-Diffusion-Distillation')
from models.teacher import load_teacher, encode_image_to_latent, decode_latent_to_image
import numpy as np

def load_prompts(prompts_file):
    """Load prompts from a JSON file"""
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)
    return prompts

def generate_teacher_dataset(output_dir, prompts_file, num_images_per_prompt=1, image_size=512):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Teacher (SD v1.5) on {device}...")
    # Load Standard SD 1.5
    # Teacher already frozen in load_teacher
    unet, vae, tokenizer, text_encoder = load_teacher()

    # Load prompts
    prompts = load_prompts(prompts_file)

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "teacher_images")
    os.makedirs(images_dir, exist_ok=True)
    latents_dir = os.path.join(output_dir, "teacher_latents")
    os.makedirs(latents_dir, exist_ok=True)

    # Initialize scheduler for diffusion process
    scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    scheduler.set_timesteps(50)

    # Generate dataset
    dataset_info = []

    for i, prompt in enumerate(tqdm(prompts, desc="Generating images")):
        for j in range(num_images_per_prompt):
            # Generate image using proper diffusion process
            with torch.no_grad():
                # Encode text prompt
                text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, 
                                      return_tensors="pt").to(device)
                text_embeddings = text_encoder(text_inputs.input_ids).last_hidden_state
                
                # Create unconditional embeddings for classifier-free guidance
                uncond_input = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, 
                                       return_tensors="pt").to(device)
                uncond_embeddings = text_encoder(uncond_input.input_ids).last_hidden_state
                
                # Combine embeddings
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
                
                # Generate random latent
                latent_shape = (1, unet.config.in_channels, image_size // 8, image_size // 8)
                latents = torch.randn(latent_shape, device=device, dtype=torch.float16)
                latents = latents * scheduler.init_noise_sigma
                
                # Denoising loop
                for t in scheduler.timesteps:
                    # Expand latents for classifier-free guidance
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                    
                    # Predict noise
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                    
                    # Classifier-free guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
                    
                    # Compute previous noisy sample
                    latents = scheduler.step(noise_pred, t, latents).prev_sample
                
                # Scale and decode latent to image
                latents = 1 / 0.18215 * latents
                image_tensor = decode_latent_to_image(vae, latents)
                
                # Scale back the latent for saving
                latent = latents * 0.18215
                latent = latent.detach().cpu()

            # Save image
            image_filename = f"image_{i:05d}_{j}.png"
            image_path = os.path.join(images_dir, image_filename)
            image_tensor = image_tensor.cpu().permute(0, 2, 3, 1).float().numpy() # From (batch, channels, H, W) to (batch, H, W, channels) and numpy
            image = Image.fromarray((image_tensor[0] * 255).astype(np.uint8)) # Convert to uint8 and then PIL Image

            # Save image and latent
            image.save(os.path.join(images_dir, f"image_{i:04d}.png"))
            torch.save(latents, os.path.join(latents_dir, f"latent_{i:04d}.pt"))


            # Store metadata
            dataset_info.append({
                "image_path": image_path,
                "prompt": prompt,
                "prompt_id": i,
                "image_id": j
            })

    # Save dataset metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Dataset created successfully in {output_dir}")
    print(f"Total images generated: {len(dataset_info)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate teacher dataset using SD v1.5")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for dataset")
    parser.add_argument("--prompts_file", type=str, required=True, help="JSON file containing prompts")
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="Number of images per prompt")
    parser.add_argument("--image_size", type=int, default=512, help="Image size (height and width)")

    args = parser.parse_args()

    generate_teacher_dataset(
        args.output_dir,
        args.prompts_file,
        args.num_images_per_prompt,
        args.image_size
    )
