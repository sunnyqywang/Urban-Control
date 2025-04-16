import numpy as np
import pandas as pd

import geopandas as gpd
import mercantile
from shapely.geometry import box
from shapely.ops import unary_union

import os


from utils_io import *
from utils_geo import *


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate landuse overlay with tiles")
    parser.add_argument("city", help="City name (dallas/la/chicago)")
    parser.add_argument("proj_root_dir", help="Path to project_root")
    parser.add_argument("output_dir", help="Path to output csv")
    parser.add_argument("offset_right", default=0)
    parser.add_argument("offset_down", default=0)
    args = parser.parse_args()
    
    city = args.city
    offset_right = float(args.offset_right)
    offset_down = float(args.offset_down)
    
    params_file = f"{args.proj_root_dir}/configs/configs_{city}.yaml"
    params = load_config(params_file)
    io_file = f"{args.proj_root_dir}/configs/io_{city}.yaml"
    io_config = load_config(io_file)
    osm_mapping_file = f"{args.proj_root_dir}/configs/configs_osm_layers.yaml"
    osm_mapping = load_config(osm_mapping_file)
    
    # Create tile grid
    bbox = [float(x) for x in params['bounding_box'].split(',')]
    tile_grid = create_tile_grid(bbox, params['zoom'], offset_right, offset_down)

    # Calculate tile area
    tile_area = calculate_tile_area(params['zoom'], params['central_x'], params['central_y'])
    tile_area = np.round(tile_area, -2)
    
    # Read relevant shapefiles
    layers = load_all(io_config['geojsons'], params={'bbox':bbox})
    
    if 'landuse_complement' not in params['layers_calculate']:
        # Complement land use layer with building layer
        buildings = layers['buildings']
        landuse = layers['landuse']
        # Only take buildings with type information
        buildings = buildings[~buildings['type'].isna()]
        # Keep buildings when land use layer is missing
        landuse_complement = gpd.sjoin(buildings, landuse, how='left', predicate="intersects")
        landuse_complement = landuse_complement[landuse_complement["index_right"].isna()]
        # Cast building types to landuse types
        landuse_complement['fclass'] = landuse_complement['type'].map(osm_mapping['building_landuse_mapping'])
        landuse_complement = landuse_complement.to_crs(3857)
        landuse_complement_dissolved = dissolve_by_type_and_distance(landuse_complement,"fclass",50)

    landuse_df = None

    for layer_name in params['layers_calculate']:
        print("Processing", layer_name)

        layer = layers[layer_name].to_crs(epsg=3857)
        if layer_name == 'water':
            layer['fclass'] = 'water'
        elif layer_name in ['landuse','traffic']:
            layer['fclass'] = layer['fclass'].map(osm_mapping['landuse_mapping'])   

        if layer_name == 'landuse':
            layer = pd.concat([layer[['fclass','geometry']].to_crs("EPSG: 4326"), 
                               landuse_complement_dissolved[['fclass','geometry']].to_crs("EPSG: 4326")])
            layer = layer.to_crs(epsg=3857)
            layer.to_file(f"{args.proj_root_dir}/data/geojsons/{city}/landuse_complement.geojson", driver="GeoJSON")
            # parking is handled by traffic layer
            layer = layer[layer['fclass']!='parking']
            
        if layer_name == 'landuse_complement':
            # parking is handled by traffic layer
            layer = layer[layer['fclass']!='parking']
            
        intersections = gpd.overlay(tile_grid, layer, how='intersection')

        if layer_name == 'buildings':
            residential_intersections = intersections[intersections['type'].isin(['house','apartments','terrace','residential','detached'])]
            residential_intersections['fclass'] = residential_intersections['type'].map(osm_mapping['building_residential_mapping'])
            residential_intersections = residential_intersections.groupby(['z','x','y','fclass'], as_index=False).agg(area_m2=('geometry', lambda x: unary_union(x).area))
            residential_intersections = residential_intersections.pivot(index=['z','x','y'], columns='fclass', values='area_m2')
            residential_intersections = residential_intersections.reset_index()
            
            intersections = intersections.groupby(['z','x','y','fclass'], as_index=False).agg(area_m2=('geometry', lambda x: unary_union(x).area))
            intersections = intersections.groupby(['z','x','y'], as_index=False).agg(
                avg_building_footprint=pd.NamedAgg(column='area_m2', aggfunc="mean"),
                tot_building_footprint=pd.NamedAgg(column='area_m2', aggfunc="sum")
            )
            intersections = intersections.reset_index()
            intersections = intersections.merge(residential_intersections, on=['z','x','y'], how='left')

        elif (layer_name == 'landuse') | (layer_name == 'landuse_complement'):
            intersections = intersections.groupby(['z','x','y','xmin','ymin','xmax','ymax','fclass'], as_index=False).agg(
                convex_hull=('geometry', lambda x: unary_union(x).convex_hull),
                convex_hull_area=('geometry', lambda x: unary_union(x).convex_hull.area),
                centroid_x=('geometry', lambda x: unary_union(x).centroid.x),
                centroid_y=('geometry', lambda x: unary_union(x).centroid.y),
                area_m2=('geometry', lambda x: unary_union(x).area)
            )
            intersections['centroid_position_x'] = (intersections['centroid_x']-intersections['xmin']) / (intersections['xmax']-intersections['xmin'])
            intersections['centroid_position_y'] = (intersections['centroid_y']-intersections['ymin']) / (intersections['ymax']-intersections['ymin'])
            intersections['concentrated'] = intersections['area_m2'] / intersections['convex_hull_area']

            intersections = intersections.set_index(['z','x','y','fclass'])
            intersections = intersections[['area_m2','convex_hull_area','concentrated','centroid_position_x','centroid_position_y']].unstack('fclass')
            intersections.columns = [f"{col[0]}_{col[1]}" for col in intersections.columns]

            intersections = intersections.reset_index()

        else:
            intersections = intersections.groupby(['z','x','y','fclass'], as_index=False).agg(area_m2=('geometry', lambda x: unary_union(x).area))
            intersections = intersections.pivot(index=['z','x','y'], columns='fclass', values='area_m2')
            intersections = intersections.reset_index()            

        if landuse_df is not None:
            landuse_df = landuse_df.merge(intersections, on=['z','x','y'], how='left')
        else:
            landuse_df = intersections

    cols = [c for c in landuse_df.columns if (c not in ['z','x','y','index']) & ('centroid' not in c) & ('concentrated' not in c)]
    landuse_df.loc[:, cols] = landuse_df[cols]/tile_area

    landuse_df.to_csv(f"{args.output_dir}/{city}_{int(offset_right*10)}_{int(offset_down*10)}.csv", index=False)
    
    print("✅ Done! Calculations saved to:", f"{args.output_dir}/{city}_{int(offset_right*10)}_{int(offset_down*10)}.csv")