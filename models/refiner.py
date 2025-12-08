import peft
from peft import LoraConfig, get_peft_model

# ==========================================
# 1. Define the Refiner (LoRA)
# ==========================================
def attach_refiner_lora(pipe):
    # We add trainable LoRA weights to the FROZEN Teacher UNet
    unet = pipe.unet

    # Standard LoRA Config for Diffusion
    lora_config = LoraConfig(
        r=16, # Rank (Size of the adapter). 16 is small and fast.
        lora_alpha=16,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"], # Attention layers
        lora_dropout=0.1,
        bias="none",
    )

    # Wrap the UNet. Now unet has trainable LoRA parts!
    pipe.unet = get_peft_model(unet, lora_config)

    # IMPORTANT: Only train LoRA, freeze original UNet
    pipe.unet.print_trainable_parameters()
    return pipe
