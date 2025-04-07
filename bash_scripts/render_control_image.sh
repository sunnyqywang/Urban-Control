#!/bin/bash

# ------------ CONFIGURATION ------------

OUTPUT_DIR="/home/gridsan/qwang/urban-control/data"
SCRIPT="/home/gridsan/qwang/urban-control/src/render_control_image.py"
PROJ_ROOT_DIR="/home/gridsan/qwang/urban-control/"

mkdir -p "$OUTPUT_DIR"

# ------------ PROCESSING ------------

# ------------ BASE - NO LANDUSE LAYERS ------------
# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0" ""
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0" ""
# python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0" ""

# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0.5" ""
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0.5" ""
# python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0.5" ""

python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0" ""
python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0.5" ""

# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0.5" ""
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0" ""


# ------------ FOREST ------------

# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0.5" "forest"
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0.5" "forest"
# python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0.5" "forest"

# ------------ RESIDENTIAL  ------------

# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0" "residential"
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0" "residential"
# python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0" "residential"
