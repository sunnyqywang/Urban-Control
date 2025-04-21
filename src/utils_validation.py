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

# helper
def tokenize_captions(prompts, tokenizer):
    inputs = tokenizer(
        prompts, 
        max_length=154, #tokenizer.model_max_length, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )
    return inputs.input_ids


def process_long_prompt_batch(text_encoder, input_ids, weight_dtype_, tokenizer, max_length=77, chunk_strategy="average", device='cuda'):
    """
    Batch-compatible long prompt handling.
    Args:
        input_ids: shape [batch_size, seq_len]
    """
    device = input_ids.device
    # Use tokenizer's pad_token_id to detect actual sequence lengths
    seq_length = (input_ids != tokenizer.pad_token_id).sum(dim=1)  # [batch_size]
    
    # Short prompts: process as a batch
    short_mask = seq_length <= max_length
    if short_mask.any():
        short_ids = input_ids[short_mask]
        # Pad short sequences to max_length if needed
        padded_short_ids = []
        for seq in short_ids:
            seq_len = (seq != tokenizer.pad_token_id).sum()
            if seq_len < max_length:
                # Replace padding with actual pad_token_id
                padding = torch.full((max_length - seq_len,), tokenizer.pad_token_id, 
                                dtype=seq.dtype, device=device)
                padded_seq = torch.cat([seq[:seq_len], padding])
            else:
                padded_seq = seq[:max_length]
            padded_short_ids.append(padded_seq)
        
        short_ids = torch.stack(padded_short_ids)
        short_hidden = text_encoder(short_ids, return_dict=False)[0]
    
    # Long prompts: handle per strategy
    long_mask = ~short_mask
    if long_mask.any():
        long_ids = input_ids[long_mask]
        if chunk_strategy == "truncate":
            truncated = long_ids[:, :max_length]
            long_hidden = text_encoder(truncated, return_dict=False)[0]
        else:
            # Split long prompts into chunks
            chunks = []
            for seq in long_ids:
                # Remove padding and split
                non_pad = seq[seq != tokenizer.pad_token_id]
                seq_chunks = [non_pad[i:i+max_length] for i in range(0, len(non_pad), max_length)]
                # Pad chunks to max_length with pad_token_id
                padded_chunks = [
                    torch.cat([
                        chunk, 
                        torch.full((max_length - len(chunk),), tokenizer.pad_token_id, 
                                dtype=chunk.dtype, device=device)
                    ])
                    for chunk in seq_chunks
                ]
                chunks.extend(padded_chunks)
            
            # Process all chunks in parallel
            chunk_batch = torch.stack(chunks)
            chunk_hidden = text_encoder(chunk_batch, return_dict=False)[0]
            
            # Recombine chunks (average/pool)
            if chunk_strategy == "average":
                # Need to average by original sequence
                start_idx = 0
                long_hidden = []
                for seq in long_ids:
                    non_pad_len = (seq != tokenizer.pad_token_id).sum()
                    num_chunks = (non_pad_len + max_length - 1) // max_length
                    seq_hidden = chunk_hidden[start_idx:start_idx+num_chunks]
                    long_hidden.append(seq_hidden.mean(dim=0))
                    start_idx += num_chunks
                long_hidden = torch.stack(long_hidden)
            else:  # "pool"
                start_idx = 0
                long_hidden = []
                for seq in long_ids:
                    non_pad_len = (seq != tokenizer.pad_token_id).sum()
                    num_chunks = (non_pad_len + max_length - 1) // max_length
                    seq_hidden = chunk_hidden[start_idx:start_idx+num_chunks]
                    long_hidden.append(seq_hidden.max(dim=0).values)
                    start_idx += num_chunks
                long_hidden = torch.stack(long_hidden)
    
    # Combine results
    hidden_states = torch.zeros(
        input_ids.size(0), max_length, text_encoder.config.hidden_size, 
        device=device,
        dtype=weight_dtype_
    )
    if short_mask.any():
        hidden_states[short_mask] = short_hidden
    if long_mask.any():
        hidden_states[long_mask] = long_hidden
    
    return hidden_states

#######################################

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

def load_controlnet_v4(
    model_path,
    controlnet_run,
    checkpoint,
    device="cuda",
    torch_dtype=torch.float16
):
    # --- Load Models ---
    controlnet_path = f"output/{controlnet_run}/{checkpoint}/controlnet"
    print(controlnet_path)
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch_dtype)
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    
    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(model_path, None)
    text_encoder = text_encoder_cls.from_pretrained(
        model_path, subfolder="text_encoder", revision=None, variant=None
    )
    
    pipeline = CustomSDControlNetPipeline.from_pretrained(
        model_path,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        controlnet=controlnet,
        safety_checker=None,
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

@torch.no_grad()
def run_controlnet_validation_v4(
    pipeline,
    val_dataloader,
    save_dir,
    seed=42,
    num_images=1,
    num_steps=30,
    device="cuda",
    torch_dtype=torch.float16
):
    pipeline = pipeline.to(device)
    #TODO accomodate more than one image
    # --- Inference Loop ---
    for batch in val_dataloader:
        # Stack control images into a batch
        control_tensors = batch["conditioning_pixel_values"].to(device)
        #input_ids = torch.stack([example["input_ids"] for example in examples])
        prompts = batch['prompt']
        text_encoder = pipeline.text_encoder
        text_encoder.to(device, dtype=torch_dtype)
        # print(prompts)
        batch['input_ids'] = tokenize_captions(prompts, pipeline.tokenizer).to(device)
        # print(batch['input_ids'])
        prompts = process_long_prompt_batch(
            text_encoder=text_encoder, 
            input_ids=batch['input_ids'], 
            weight_dtype_=torch_dtype, 
            tokenizer=pipeline.tokenizer
        )
        
        # Generate images (batched)
        generators = [
            torch.Generator(device=device).manual_seed(seed) 
            for i in range(len(prompts))
        ]
        
        with torch.autocast(device_type="cuda"):
            output_images = pipeline(
                prompt_embeds=prompts,
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
