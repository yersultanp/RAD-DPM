# models/teacher/teacher_loader.py
import torch
from diffusers import StableDiffusionPipeline

def load_teacher():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")

    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()

    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    return pipe.unet, pipe.vae, pipe.tokenizer, pipe.text_encoder

def teacher_noise_pred(unet, x_t, t, cond):
    with torch.no_grad():
        return unet(
            x_t,
            t,
            encoder_hidden_states=cond
        ).sample

def encode_image_to_latent(vae, image):
    return vae.encode(image).latent_dist.sample()

def decode_latent_to_image(vae, lat):
    return vae.decode(lat).sample()
