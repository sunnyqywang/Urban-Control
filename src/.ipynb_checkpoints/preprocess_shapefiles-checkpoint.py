from utils_geo import batch_clip_shapefiles
from utils_io import *

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Clip shapefile to bbox and save as GeoJSON.")
    parser.add_argument("input", help="Path to input shapefile (.shp)")
    parser.add_argument("output", help="Path to output GeoJSON")
    parser.add_argument("config", help="Path to config files")
    # parser.add_argument("--bbox", nargs=4, type=float, required=True,
    #                     metavar=('MINX', 'MINY', 'MAXX', 'MAXY'),
    #                     help="Bounding box in format: minx miny maxx maxy")
    parser.add_argument("--bboxcrs", default="EPSG:4326", help="CRS of bbox (default: EPSG:4326)")
    args = parser.parse_args()
    
    params = load_config(args.config)
    bbox = [float(x) for x in params['bounding_box'].split(',')]

    batch_clip_shapefiles(shapefile_dir=args.input, 
                      bbox=bbox,
                      output_dir=args.output, 
                      bbox_crs=args.bboxcrs)
    