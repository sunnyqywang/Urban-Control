#!/bin/bash

python3 "/home/gridsan/qwang/urban-control/src/log_validation_batch_v4.py" \
    --model_path "models/stable-diffusion-v1-5" \
    --run_id "20250416_v4" \
    --validation_data_dir "data/train/20250416_v4_validation.csv" \
    --val_batch_size 8 \
    --image_column "image_column" \
    --conditioning_image_column "conditioning_image_column" \
    --caption "llm_caption" \
    --checkpoints "50000"
    