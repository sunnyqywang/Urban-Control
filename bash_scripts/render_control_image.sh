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

# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0" ""
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0.5" ""

# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0.5" ""
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0" ""

python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.3" "0.7" ""
python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.3" "0.7" ""
python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.3" "0.7" ""

python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.7" "0.3" ""
python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.7" "0.3" ""
python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.7" "0.3" ""

# ------------ FOREST ------------

# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0" "forest"
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0" "forest"
# python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0" "forest"

# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0.5" "forest"
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0.5" "forest"
# python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0.5" "forest"

# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0" "forest"
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0.5" "forest"

# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0.5" "forest"
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0" "forest"

python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.3" "0.7" "forest"
python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.3" "0.7" "forest"
python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.3" "0.7" "forest"

python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.7" "0.3" "forest"
python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.7" "0.3" "forest"
python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.7" "0.3" "forest"

# ------------ RESIDENTIAL  ------------

# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0" "residential"
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0" "residential"
# python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0" "residential"

# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0.5" "residential"
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0.5" "residential"
# python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0.5" "residential"

# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0" "residential"
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0.5" "residential"

# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0.5" "residential"
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0" "residential"

python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.3" "0.7" "residential"
python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.3" "0.7" "residential"
python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.3" "0.7" "residential"

python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.7" "0.3" "residential"
python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.7" "0.3" "residential"
python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.7" "0.3" "residential"

# ------------ COMMERCIAL  ------------

# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0" "commercial"
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0" "commercial"
# python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0" "commercial"

# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0.5" "commercial"
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0.5" "commercial"
# python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0.5" "commercial"

# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0" "commercial"
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0.5" "commercial"

# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0.5" "commercial"
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0" "commercial"

python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.3" "0.7" "commercial"
python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.3" "0.7" "commercial"
python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.3" "0.7" "commercial"

python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.7" "0.3" "commercial"
python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.7" "0.3" "commercial"
python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.7" "0.3" "commercial"

# ------------ INDUSTRIAL  ------------

# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0" "industrial"
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0" "industrial"
# python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0" "industrial"

# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0.5" "industrial"
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0.5" "industrial"
# python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0.5" "industrial"

# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0" "industrial"
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0.5" "industrial"

# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0.5" "industrial"
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0" "industrial"

python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.3" "0.7" "commercial"
python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.3" "0.7" "commercial"
python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.3" "0.7" "commercial"

python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.7" "0.3" "commercial"
python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.7" "0.3" "commercial"
python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.7" "0.3" "commercial"


# ------------ FARMLAND  ------------

# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0" "farmland"
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0" "farmland"
# python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0" "farmland"

# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0.5" "farmland"
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0.5" "farmland"
# python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0.5" "farmland"

# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0" "farmland"
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0.5" "farmland"

# python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0" "0.5" "farmland"
# python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.5" "0" "farmland"

python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.3" "0.7" "farmland"
python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.3" "0.7" "farmland"
python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.3" "0.7" "farmland"

python3 "$SCRIPT" "dallas" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.7" "0.3" "farmland"
python3 "$SCRIPT" "la" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.7" "0.3" "farmland"
python3 "$SCRIPT" "chicago" "$PROJ_ROOT_DIR" "$OUTPUT_DIR" "0.7" "0.3" "farmland"