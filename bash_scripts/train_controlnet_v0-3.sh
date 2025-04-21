#!/bin/bash

python3 "/home/gridsan/qwang/urban-control/src/train_controlnet_v0-3.py" \
  --pretrained_model_name_or_path "models/stable-diffusion-v1-5" \
  --output_dir "output/20250416_v3" \
  --train_data_dir "./data/train/20250416_v3_train.csv" \
  --tracker_project_name "20250416_v3" \
  --resume_from_checkpoint "latest" \
  --seed 42 \
  --num_train_epochs 10 \
  --checkpoints_total_limit 10 \
  --checkpointing_steps 2000 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --lr_scheduler "cosine" \
  --lr_warmup_steps 500 \
  --mixed_precision "fp16" \
  --validation_steps 300 \
  --validation_image \
    "./data/validation/chicago_16803_24339.png" \
    "./data/validation/chicago_16812_24354.png" \
    "./data/validation/la_11199_26119.png" \
    "./data/validation/la_11207_26155.png" \
  --validation_prompt \
    "Satellite image in a city in chicago. Landuse include: 55% residential , parking (10%), commercial (10%), recreational (10%) . medium building density. Residential type is mainly apartment complexes." \
    "Satellite image in a city in la. Landuse include: 40% residential , commercial (25%), recreational (20%), forest (10%), parking (5%) . medium building density. Residential type is mainly single-family homes , with apartment complexes." \
    "Satellite image in a city in dallas.landuse include: 90% commercial , industrial (10%) . high building density." \
    "Satellite image in a city in dallas. Landuse include: 25% commercial , residential (25%), recreational (10%), industrial (5%), forest (5%) . high building density. Residential type is mainly apartment complexes." \
  --image_column "image_column" \
  --conditioning_image_column "conditioning_image_column" \
  --caption "caption" \
  --enable_xformers_memory_efficient_attention \

# v0 validation
# "Satellite image of a town in Chicago. Landuse parcels include 30 percent residential,
# 25 percent park, 25 percent nature reserve. Sparse building coverage." \
# "Satellite image of a city in Los Angeles. Landuse parcels include 65 percent residential,
#   10 percent park. Residential area consists entirely of houses. Medium building coverage." \
# "Satellite image of a town in Dallas. Landuse parcels include 20 percent residential,
#   40 percent industrial. Residential area consists entirely of houses. Medium building
#   coverage. " \
# "Satellite image of a town in Dallas. Landuse parcels include 65 percent residential,
#   30 percent industrial, 5 percent commercial. Residential area consists entirely
#   of houses. Medium building coverage." \


# v1 validation
  # --validation_image \
  #   "./data/validation/chicago_16803_24339.png" \
  #   "./data/validation/chicago_16812_24354.png" \
  #   "./data/validation/la_11199_26119.png" \
  #   "./data/validation/la_11207_26155.png" \
  # --validation_prompt \
  #   "This is a satellite image of la where the city forms the core. residential areas (50%) prevail here , with pockets of commercial (35%), recreational (5%) . Building density is high in this area. You'll find mostly apartment complexes here , alongside single-family homes dwellings." \
  #   "This is a satellite image of city in la. industrial areas (55%) prevail here , alongside some commercial (25%), residential (10%) . This area has a high building density. The residential buildings are mainly apartment complexes." \
  #   "This is a satellite image of city in chicago. Meanwhile, residential areas (30%) prevail here , with pockets of industrial (25%), parking (15%), forest (10%), recreational (10%) . Building density is medium in this area. In terms of settlement, single-family homes structures dominate the residential areas." \
  #   "This is a satellite image of city in dallas. In terms of settlement, this area is dominated by residential (50%) , with pockets of commercial (15%), industrial (5%), parking (5%) . Building density is high in this area. Meanwhile, you'll find mostly apartment complexes here , alongside townhouses dwellings." \

# v2 validation
  # --validation_image \
  #   "./data/validation/chicago_16803_24339.png" \
  #   "./data/validation/chicago_16812_24354.png" \
  #   "./data/validation/dallas_16+0+5_industrial_15029_26462.png" \
  #   "./data/validation/dallas_16+5+5_commercial_15147_26430.png" \
  #   "./data/validation/chicago_16+0+0_forest_16794_24425.png" \
  #   "./data/validation/la_16+5+0_residential_11287_26187.png" \
  # --validation_prompt \
  #   "This is a satellite image of la where the city forms the core. In terms of settlement, you'll find mostly residential (55%) in this zone , with pockets of recreational (30%) . Building density is low in this area. Furthermore, single-family homes structures dominate the residential areas." \
  #   "This is a satellite image of city in dallas. This area is dominated by residential (35%) , with pockets of commercial (20%), recreational (15%), parking (5%) . Building density is high in this area. single-family homes structures dominate the residential areas." \
  #   "The area shown in the satellite image of dallas falls within the city. Furthermore, you'll find mostly residential (60%) in this zone , complemented by industrial (30%) . This area has a medium building density. Furthermore, a industrial patch appears in the lower central region of the image in shaded purple. Additionally, single-family homes structures dominate the residential areas , mixed with apartment complexes residences." \
  #   "The area shown in the satellite image of dallas falls within the city. This area is dominated by residential (70%) , alongside some commercial (25%), parking (5%) . Building density is medium in this area. The main commercial zone is located toward the mid right in shaded blue. single-family homes structures dominate the residential areas." \
  #   "The area shown in the satellite image of chicago falls within the city. residential areas (60%) prevail here , complemented by forest (15%), industrial (25%). A forest patch appears in the lower right region of the image in shaded dark green." \
  #   "This is a satellite image of la where the city forms the core. Additionally, the landscape is primarily industrial (40%) , with pockets of residential (35%), commercial (5%) . This area has a high building density. The residential buildings are mainly apartment complexes. Meanwhile, the residential area is concentrated in the center in shaded orange." \
      



# v3 validation
# "Satellite image in a city in chicago. Landuse include: 55% residential , parking (10%), commercial (10%), recreational (10%) . medium building density. Residential type is mainly apartment complexes." \
# "Satellite image in a city in la. Landuse include: 40% residential , commercial (25%), recreational (20%), forest (10%), parking (5%) . medium building density. Residential type is mainly single-family homes , with apartment complexes." \
# "Satellite image in a city in dallas.landuse include: 90% commercial , industrial (10%) . high building density." \
# "Satellite image in a city in dallas. Landuse include: 25% commercial , residential (25%), recreational (10%), industrial (5%), forest (5%) . high building density. Residential type is mainly apartment complexes." \
