import torch
import gc
import os
from PIL import Image
from torchvision import transforms
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel
import matplotlib.pyplot as plt
import pandas as pd
import math
import textwrap

def load_controlnet(
    model_path,
    controlnet_run,
    checkpoint,
    device="cuda",
    torch_dtype=torch.float16
):
    # --- Load Models ---
    controlnet_path = f"output/{controlnet_run}/{checkpoint}/controlnet"
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch_dtype)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        model_path,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.enable_attention_slicing()
    
    return pipeline
    
def get_control_image_processor(resolution):
    return transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ])

@torch.no_grad()
def run_controlnet_validation(
    pipeline,
    val_dataloader,
    save_dir,
    seed=42,
    num_images=1,
    num_steps=30,
    device="cuda",
    torch_dtype=torch.float16
):
    #TODO accomodate more than one image
    # --- Inference Loop ---
    for batch in val_dataloader:
        # Stack control images into a batch
        control_tensors = batch["conditioning_pixel_values"]
        prompts = batch['prompt']
        # Generate images (batched)
        generators = [
            torch.Generator(device=device).manual_seed(seed) 
            for i in range(len(prompts))
        ]
        
        with torch.autocast(device_type="cuda"):
            output_images = pipeline(
                prompt=prompts,
                image=control_tensors,
                num_inference_steps=num_steps,
                generator=generators,
            ).images        
            
        # --- Save each image individually ---
        for i, (name, output_img) in enumerate(zip(batch["name"], output_images)):
            output_path = os.path.join(save_dir, name)   
            os.makedirs('/'.join(output_path.split("/")[:-1]), exist_ok=True) 
            # Save the generated image
            output_img.save(output_path)

    # --- Cleanup ---
    torch.cuda.empty_cache()
