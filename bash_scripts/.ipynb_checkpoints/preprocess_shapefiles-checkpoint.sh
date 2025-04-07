#!/bin/bash

# ------------ CONFIGURATION ------------

INPUT_DIR="/home/gridsan/qwang/urban-control/data/shapefiles/socal-0915"
OUTPUT_DIR="/home/gridsan/qwang/urban-control/data/geojsons/la"
CONFIG_DIR="/home/gridsan/qwang/urban-control/configs/configs_la.yaml"
SCRIPT="/home/gridsan/qwang/urban-control/src/preprocess_shapefiles.py"

# INPUT_DIR="/home/gridsan/qwang/urban-control/data/shapefiles/illinois-0811"
# OUTPUT_DIR="/home/gridsan/qwang/urban-control/data/geojsons/chicago"
# CONFIG_DIR="/home/gridsan/qwang/urban-control/configs/configs_chicago.yaml"
# SCRIPT="/home/gridsan/qwang/urban-control/src/preprocess_shapefiles.py"


# INPUT_DIR="/home/gridsan/qwang/urban-control/data/shapefiles/texas-0915"
# OUTPUT_DIR="/home/gridsan/qwang/urban-control/data/geojsons/dallas"
# CONFIG_DIR="/home/gridsan/qwang/urban-control/configs/configs_dallas.yaml"
# SCRIPT="/home/gridsan/qwang/urban-control/src/preprocess_shapefiles.py"

mkdir -p "$OUTPUT_DIR"

# ------------ PROCESSING ------------

python3 "$SCRIPT" "$INPUT_DIR" "$OUTPUT_DIR" "$CONFIG_DIR"
    

