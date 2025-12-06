import torch
import os
import json
from diffusers import StableDiffusionPipeline
from PIL import Image
import argparse
from tqdm import tqdm

def load_prompts(prompts_file):
    """Load prompts from a JSON file"""
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)
    return prompts

def generate_teacher_dataset(output_dir, prompts_file, num_images_per_prompt=1, image_size=512):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading Teacher (SD v1.5) on {device}...")
    # Load Standard SD 1.5
    model_id = "sd-legacy/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    # Freeze Teacher
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    # Load prompts
    prompts = load_prompts(prompts_file)

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Generate dataset
    dataset_info = []

    for i, prompt in enumerate(tqdm(prompts, desc="Generating images")):
        for j in range(num_images_per_prompt):
            # Generate image
            with torch.no_grad():
                image = pipe(
                    prompt,
                    height=image_size,
                    width=image_size,
                    num_inference_steps=50,
                    guidance_scale=7.5
                ).images[0]

            # Save image
            image_filename = f"image_{i:05d}_{j}.png"
            image_path = os.path.join(images_dir, image_filename)
            image.save(image_path)

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
