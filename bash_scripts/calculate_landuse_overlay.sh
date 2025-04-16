#!/bin/bash

# ------------ CONFIGURATION ------------

OUTPUT_DIR="/home/gridsan/qwang/urban-control/data/landuse_overlay"
SCRIPT="/home/gridsan/qwang/urban-control/src/calculate_landuse_overlay.py"
PROJ_ROOT_DIR="/home/gridsan/qwang/urban-control/"

mkdir -p "$OUTPUT_DIR"

# ------------ PROCESSING ------------

# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0"
# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0.5"
# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0"
# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0.5"
python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.7" "0.3"
python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.3" "0.7"

# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR"  "0" "0"
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR"  "0.5" "0.5"
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR"  "0.5" "0"
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR"  "0" "0.5"
python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.7" "0.3"
python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.3" "0.7"

# python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0"
# python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0.5"
python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.7" "0.3"
python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.3" "0.7"

