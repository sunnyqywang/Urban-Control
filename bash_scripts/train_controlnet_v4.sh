#!/bin/bash

python3 "/home/gridsan/qwang/urban-control/src/train_controlnet_v4.py" \
  --pretrained_model_name_or_path "models/stable-diffusion-v1-5" \
  --output_dir "output/20250414_v4" \
  --tracker_project_name "20250414_v4" \
  --seed 42 \
  --num_train_epochs 10 \
  --checkpoints_total_limit 10 \
  --checkpointing_steps 2000 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --lr_scheduler "cosine" \
  --lr_warmup_steps 500 \
  --mixed_precision "fp16" \
  --train_data_dir "./data/train/20250414_v4_train.csv" \
  --validation_steps 300 \
  --image_column "image_column" \
  --conditioning_image_column "conditioning_image_column" \
  --caption "llm_caption" \
  --enable_xformers_memory_efficient_attention \
  --validation_image \
    "./data/validation/chicago_16803_24339.png" \
    "./data/validation/chicago_16812_24354.png" \
    "./data/validation/la_11199_26119.png" \
    "./data/validation/la_11207_26155.png" \
  --validation_prompt \
    "This satellite view of Chicago showcases an urban landscape with predominantly residential areas comprising 85%. The remaining five percent comprises commercial spaces such as offices or shopping centers. Medium-density buildings are evident throughout the area, primarily consisting of multi-story apartment complexes that house residents." \
    "A satellite view of the Dallas cityscape showcases an area with predominantly suburban characteristics. The land use distribution reveals that approximately 45% of the region comprises recreational spaces such as parks or sports facilities; meanwhile, around 35% consists of residential zones where families reside primarily within detached houses. In comparison to other cities, there's relatively lower commercial development at just over 5%. This suggests a balanced mix between leisure activities, comfortable living environments for residents, and limited business infrastructure." \
    "This satellite view of an urban area showcases the landscape with various land uses including 35 percent for residential areas, primarily composed of single-family homes; 30 percent dedicated to industry; and 10 percent allocated towards water bodies such as lakes or rivers. The overall built environment exhibits moderate density characterized by mid-sized buildings that contribute to the visual character of the region." \
    "A detailed satellite view of the Dallas City reveals an impressive landscape dominated by commercial establishments accounting for 75 percent of land use. The area is characterized by dense buildings that create a bustling atmosphere with tall skyscrapers punctuating the skyline. High-rise structures are interspersed among smaller residential areas, creating a visually striking contrast between these two types of development."



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

# v2 validation
  # --validation_image \
  #   "./data/validation/chicago_16803_24339.png" \
  #   "./data/validation/chicago_16812_24354.png" \
  #   "./data/validation/la_11199_26119.png" \
  #   "./data/validation/la_11207_26155.png" \
  #   "./data/validation/chicago_16+0+0_forest_16794_24425.png" \
  #   "./data/validation/la_16+5+0_residential_11287_26187.png" \
  # --validation_prompt \
  #   "This is a satellite image of la where the city forms the core. In terms of settlement, you'll find mostly residential (55%) in this zone , with pockets of recreational (30%) . Building density is low in this area. Furthermore, single-family homes structures dominate the residential areas." \
  #   "This is a satellite image of city in dallas. This area is dominated by residential (35%) , with pockets of commercial (20%), recreational (15%), parking (5%) . Building density is high in this area. single-family homes structures dominate the residential areas." \
  #   "This is a satellite image of city in chicago. In terms of settlement, residential areas (50%) prevail here , with pockets of industrial (15%), recreational (15%) . This area has a medium building density. Furthermore, housing consists primarily of single-family homes." \
  #   "This is a satellite image of city in la. Additionally, the landscape is primarily residential (40%) , complemented by commercial (30%), industrial (15%), parking (5%) . This area has a high building density. Furthermore, apartment complexes structures dominate the residential areas , complemented by single-family homes." \
  #   "The area shown in the satellite image of chicago falls within the city. residential areas (60%) prevail here , complemented by forest (15%), industrial (25%). A forest patch appears in the upper right region of the image in shaded dark green." \
  #   "This is a satellite image of la where the city forms the core. Additionally, the landscape is primarily industrial (40%) , with pockets of residential (35%), commercial (5%) . This area has a high building density. The residential buildings are mainly apartment complexes. Meanwhile, the residential area is concentrated in the center in shaded orange." \



# v3 validation
# "Satellite image in a city in chicago. Landuse include: 55% residential , parking (10%), commercial (10%), recreational (10%) . medium building density. Residential type is mainly apartment complexes." \
# "Satellite image in a city in la. Landuse include: 40% residential , commercial (25%), recreational (20%), forest (10%), parking (5%) . medium building density. Residential type is mainly single-family homes , with apartment complexes." \
# "Satellite image in a city in dallas.landuse include: 90% commercial , industrial (10%) . high building density." \
# "Satellite image in a city in dallas. Landuse include: 25% commercial , residential (25%), recreational (10%), industrial (5%), forest (5%) . high building density. Residential type is mainly apartment complexes." \
