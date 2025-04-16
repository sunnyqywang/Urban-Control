import numpy as np
import pandas as pd

import geopandas as gpd
import mercantile
from shapely.geometry import box

import os

from utils_io import *
from utils_geo import *

from multiprocessing import Pool

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render control images")
    parser.add_argument("city", help="City name (dallas/la/chicago)")
    parser.add_argument("proj_root_dir", help="Path to project_root")
    parser.add_argument("output_dir", help="Path to output csv")
    parser.add_argument("offset_right", default=0)
    parser.add_argument("offset_down", default=0)
    parser.add_argument("additional_layer", default="")
    
    args = parser.parse_args()
    
    
    city = args.city
    offset_right = float(args.offset_right)
    offset_down = float(args.offset_down)
    
    params_file = f"{args.proj_root_dir}/configs/configs_{city}.yaml"
    params = load_config(params_file)
    io_file = f"{args.proj_root_dir}/configs/io_{city}.yaml"
    io_config = load_config(io_file)
    osm_params = f"{args.proj_root_dir}/configs/configs_osm_layers.yaml"
    osm_params = load_config(osm_params)
    
    if args.additional_layer == "":
        tile_save_dir = f"{args.output_dir}/satellite_tiles_control_base/{city}/"
    else:
        tile_save_dir = f"{args.output_dir}/satellite_tiles_control_{args.additional_layer}/{city}/"
    os.makedirs(tile_save_dir, exist_ok=True)
    
    # Create tile grid
    bbox = [float(x) for x in params['bounding_box'].split(',')]
    tile_grid = create_tile_grid(bbox, params['zoom'], offset_right, offset_down)

    # Calculate tile area
    tile_area = calculate_tile_area(params['zoom'], params['central_x'], params['central_y'])
    tile_area = np.round(tile_area, -2)
    NUM_WORKERS = 2

    layer_render = list(osm_params['layers_render'].keys())

    # Read relevant shapefiles
    load_files = {}
    for name in layer_render:
        load_files[name] = io_config['geojsons'][name]
    layers = load_all(load_files, params={'bbox':bbox})
    
    grouped_layers = {}
    print("📦 Loading and assigning layers...")
    if args.additional_layer != "":        
        layer = layers['landuse_complement'].to_crs("EPSG:3857")
        layer = layer[layer['fclass']==args.additional_layer]
        assigned = assign_features_to_tiles(layer, tile_grid)
        grouped_layers[args.additional_layer] = assigned
        print(f"  ✅ {args.additional_layer}: {len(assigned)} features assigned to tiles")
        # if saving landuse, only save tiles with that landuse present
        tile_grid = tile_grid.merge(grouped_layers[args.additional_layer][['z','x','y']].drop_duplicates(), on=['z','x','y'])
        
    layer_render.remove('landuse_complement')
    for name in layer_render:
        layer = layers[name].to_crs("EPSG:3857")
        if 'filters' in osm_params['layers_render'][name]:
            for c, f in osm_params['layers_render'][name]['filters'].items():
                if isinstance(f, list):
                    layer = layer[layer[c].isin(f)]
                elif isinstance(f, dict):
                    if 'geq' in f.keys():
                        layer = layer[layer[c] >= float(f['geq'])]
                else:
                    raise ValueError("unknown filter condition")
        assigned = assign_features_to_tiles(layer, tile_grid)
        grouped_layers[name] = assigned
        print(f"  ✅ {name}: {len(assigned)} features assigned to tiles")
        

    print("🎨 Rendering tiles in parallel...")
    tile_jobs = [
        (row["z"], row["x"], row["y"], row["geometry"], grouped_layers, osm_params['layers_render'], tile_save_dir, offset_right, offset_down)
        for _, row in tile_grid.iterrows()
    ]

    with Pool(NUM_WORKERS) as pool:
        pool.map(render_tile, tile_jobs)

    print(f"✅ Done! {len(tile_grid)} tiles saved to:", tile_save_dir, offset_right, offset_down)