#!/bin/bash

python3 "/home/gridsan/qwang/urban-control/src/log_validation.py" \
    --model_path "models/stable-diffusion-v1-5" \
    --run_id "20250414_v3" \
    --data_dir "data/train/20250414_v0_validation.csv" \
    --num_instances 1 
