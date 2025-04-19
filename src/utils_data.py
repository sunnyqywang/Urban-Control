from datasets import Dataset
from datasets import load_dataset
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import functional as vision_functional
import torch
import torch.nn.functional as F
import random
from torchvision import transforms
from PIL import Image

def make_dataset(args, tokenizer, accelerator, split="train"):

    is_train = split == "train"

    # Load the dataset
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            data_dir=args.train_data_dir if is_train else args.validation_data_dir,
        )
    else:
        data_path = args.train_data_dir if is_train else args.validation_data_dir
        if data_path is not None:
            df = pd.read_csv(data_path)
            dataset = {split: Dataset.from_pandas(df)}
        else:
            raise ValueError("Data path not specified for the split.")

    column_names = dataset[split].column_names

    # Column resolution
    image_column = args.image_column or column_names[0]
    caption_column = args.caption_column or column_names[1]
    conditioning_image_column = args.conditioning_image_column or column_names[2]

    for col in [image_column, caption_column, conditioning_image_column]:
        if col not in column_names:
            raise ValueError(f"Column `{col}` not found in dataset columns: {column_names}")

    def tokenize_captions(examples):
        captions = []
        for caption in examples[caption_column]:
            if is_train and random.random() < args.proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(f"Caption column `{caption_column}` should contain strings or lists.")
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Transforms
    image_transforms = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    image_transforms_90 = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
        transforms.Lambda(lambda x: vision_functional.rotate(x, 90)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    conditioning_image_transforms = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
    ])

    conditioning_image_transforms_90 = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
        transforms.Lambda(lambda x: vision_functional.rotate(x, 90)),
        transforms.ToTensor(),
    ])

    def preprocess_fn(examples):
        if is_train:
            rotate_random_condition_type = np.array(['base' in condition_path for condition_path in examples[conditioning_image_column]])
            rotate_random_random_number = (np.random.rand(len(examples[image_column])) > 0.5)
            rotate_random = rotate_random_condition_type & rotate_random_random_number
        else:
            rotate_random = [False] * len(examples[image_column])

        images = [image_transforms_90(Image.open(path).convert("RGB")) 
                  if rotate 
                  else image_transforms(Image.open(path).convert("RGB")) 
                  for path, rotate in zip(examples[image_column], rotate_random)
                 ]

        cond_images = [conditioning_image_transforms_90(Image.open(path).convert("RGB")) 
                       if rotate 
                       else conditioning_image_transforms(Image.open(path).convert("RGB"))
                       for path, rotate in zip(examples[conditioning_image_column], rotate_random)
                      ]
        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = cond_images
        if is_train:
            examples["input_ids"] = tokenize_captions(examples)
        else:
            examples["prompt"]= examples[caption_column]
            examples["name"] = ["_".join(x.split('/')[-3:]) for x in examples[conditioning_image_column]]
        
        return examples

    # Shuffle and sample
    if is_train:
        with accelerator.main_process_first():
            if is_train and args.max_train_samples is not None:
                dataset[split] = dataset[split].shuffle(seed=args.seed).select(range(args.max_train_samples))
            elif not is_train and getattr(args, "max_val_samples", None) is not None:
                dataset[split] = dataset[split].select(range(args.max_val_samples))

            dataset_processed = dataset[split].with_transform(preprocess_fn)
    else:
        dataset_processed = dataset[split].with_transform(preprocess_fn)

    return dataset_processed

def collate_fn_train(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.stack([example["input_ids"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids
    }


def collate_fn_validation(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    prompt = [example["prompt"] for example in examples]
    names = [example["name"] for example in examples]

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "prompt": prompt,
        "name": names
    }