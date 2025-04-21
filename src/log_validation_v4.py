import argparse
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
from train_controlnet_v4 import import_model_class_from_model_name_or_path, CustomSDControlNetPipeline
from utils_validation import load_controlnet_v4

def get_control_image_processor(resolution):
    return transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ])

@torch.no_grad()
def run_controlnet_validation(
    model_path,
    controlnet_run,
    checkpoint,
    prompt,
    control_image_path,
    target_image_path,
    resolution=512,
    seed=42,
    device="cuda",
    torch_dtype=torch.float16,
    num_images=4,
    num_steps=30,
):
    #controlnet_path = f"output/{controlnet_run}/{checkpoint}/controlnet"
    #controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch_dtype)
    pipeline = load_controlnet_v4(model_path, controlnet_run, checkpoint)
    # pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    #     model_path,
    #     controlnet=controlnet,
    #     safety_checker=None,
    #     torch_dtype=torch_dtype,
    # )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.enable_attention_slicing()
    
    processor = get_control_image_processor(resolution)
    control_image = Image.open(control_image_path).convert("RGB")
    target_image = Image.open(target_image_path).convert("RGB")
    control_tensor = processor(control_image).unsqueeze(0).to(device, dtype=torch_dtype)

    generator = torch.Generator(device=device).manual_seed(seed)
    images = [control_image, target_image]

    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    tokenizer.model_max_length = 154
    tokens = tokenizer(prompt, return_tensors="pt", truncation=True)["input_ids"][0]
    chunks = [tokens[i:i+77] for i in range(0, len(tokens), 77)]
    
    chunk_embeds = []
    for chunk in chunks:
        padded = torch.nn.functional.pad(chunk, (0, 77 - chunk.size(0)), value=tokenizer.pad_token_id).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = text_encoder(padded)[0]  # shape: [1, seq_len, hidden_dim]
        chunk_embeds.append(emb)
    prompt_embeds = torch.mean(torch.stack(chunk_embeds), dim=0)

    for i in range(num_images):
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu"):
            image = pipeline(
                prompt_embeds=prompt_embeds,
                image=control_tensor,
                num_inference_steps=num_steps,
                generator=generator,
            ).images[0]
            images.append(image)

    # Calculate the grid size for subplots
    grid_size = math.ceil(math.sqrt(num_images+2))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10,10))

    # Flatten the axes array for easy iteration
    axes = axes.ravel()

    # Plot each image in a subplot
    for idx, (img, ax) in enumerate(zip(images, axes)):
        ax.imshow(img)
        ax.axis('off')

    # Hide any remaining empty subplots
    for ax in axes[num_images:]:
        ax.axis('off')
    wrapped_title = textwrap.fill(prompt, width=80)  # Adjust width as needed
    fig.suptitle(wrapped_title, y=1.02, fontsize=10)  # y>1.0 moves title up slightly

    plt.tight_layout()

    # Save the combined figure
    filename = "_".join(control_image_path.split('/')[-3:]) + checkpoint
    os.makedirs(os.path.join("output", controlnet_run, f"log_validation/"), exist_ok=True)

    out_path = os.path.join("output", controlnet_run, f"log_validation/{filename}.png")
    plt.savefig(out_path, bbox_inches='tight')

    plt.close()  # Close the figure to free memory
    print(f"Saved combined image to {out_path}")
    
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to base pretrained SD model")
    parser.add_argument("--run_id", type=str, required=True, help="Path to trained ControlNet run_id")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to validation set")
    parser.add_argument("--num_instances", type=int, required=True, help="Number of pairs to sample and generate")
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"])

    args = parser.parse_args()

    torch_dtype = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }[args.dtype]

    data = pd.read_csv(args.data_dir)
    if args.num_instances > 0:
        data = data.sample(args.num_instances)

    directory = f"output/{args.run_id}/"  

    # Get all folders starting with 'checkpoint'
    checkpoint_folders = [
        folder for folder in os.listdir(directory) 
        if folder.startswith('checkpoint') and os.path.isdir(os.path.join(directory, folder))
    ]
    checkpoint_folders = sorted(
        checkpoint_folders,
        key=lambda x: int(x.split('-')[-1])  # Extract the number and sort
    )
    print("Checkpoints found:", checkpoint_folders)

    for i, prompt, image_path, control_image_path in zip(range(args.num_instances), data['caption'], data['image_column'], data['conditioning_image_column']):
        print("Generating image...", i+1)
        for checkpoint_path in checkpoint_folders:
            run_controlnet_validation(
                model_path=args.model_path,
                controlnet_run=args.run_id,
                checkpoint=checkpoint_path,
                prompt=prompt,
                control_image_path=control_image_path,
                target_image_path=image_path,
                resolution=args.resolution,
                seed=args.seed,
                num_images=args.num_images,
                num_steps=args.num_steps,
                torch_dtype=torch_dtype,
            )


