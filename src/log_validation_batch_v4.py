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
from utils_data import make_dataset, collate_fn_validation
from utils_validation import load_controlnet_v4, run_controlnet_validation_v4

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to base pretrained SD model")
    parser.add_argument("--run_id", type=str, required=True, help="Path to trained ControlNet run_id")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--validation_data_dir", type=str, required=True, help="Path to validation set")
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"])
    parser.add_argument("--image_column", type=str, default="image", help="The column of the dataset containing the target image.")
    parser.add_argument("--conditioning_image_column", type=str, default="conditioning_image", help="The column of the dataset containing the controlnet conditioning image.")
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument("--checkpoints", type=str, default="")

    args = parser.parse_args()

    torch_dtype = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }[args.dtype]

    # --- Load Validation Data ---
    val_dataset = make_dataset(args, None, None, "validation")

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn_validation,
        batch_size=args.val_batch_size,
        num_workers=10
    )

    save_dir = f"output/{args.run_id}/"  

    # Get all folders starting with 'checkpoint'
    checkpoint_folders = [
        folder for folder in os.listdir(save_dir) 
        if folder.startswith('checkpoint') and os.path.isdir(os.path.join(save_dir, folder))
    ]
    checkpoint_folders = sorted(
        checkpoint_folders,
        key=lambda x: int(x.split('-')[-1])  # Extract the number and sort
    )
    print("Checkpoints found:", checkpoint_folders)
    if args.checkpoints != "":
        load_checkpoints = [int(x) for x in args.checkpoints.split(',')]
    else:
        load_checkpoints = []
    for checkpoint_path in checkpoint_folders:
        checkpoint = int(checkpoint_path.split('-')[-1])
        if load_checkpoints != []:
            if not (checkpoint in load_checkpoints):
                continue
        print("Validating checkpoint:", checkpoint_path)
        pipeline = load_controlnet_v4(args.model_path, args.run_id, checkpoint_path)
        run_controlnet_validation_v4(
            pipeline=pipeline,
            val_dataloader=val_dataloader,
            save_dir=os.path.join(save_dir,"validation",str(checkpoint_path.split('-')[-1])),
            seed=args.seed,
            num_images=args.num_images,
            num_steps=args.num_steps,
            torch_dtype=torch_dtype,
        )


