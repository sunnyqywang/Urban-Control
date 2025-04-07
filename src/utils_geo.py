import geopandas as gpd
import mercantile
from shapely import box
from shapely.geometry import shape
import os
import matplotlib.pyplot as plt
import fiona
from pyproj import Transformer

def dissolve_by_type_and_distance(gdf, attr, distance):
    """
    Dissolve geometries within a certain distance, grouped by attribute.
    
    Parameters:
        gdf (GeoDataFrame): Input features
        attr (str): Attribute column to group by (e.g., 'type')
        distance (float): Distance threshold (in CRS units)
    
    Returns:
        GeoDataFrame: Dissolved features
    """
    
    dissolved_parts = []
    for value, group in gdf.groupby(attr):
        # Buffer each geometry outward by half the distance (to allow merge)
        buffered = group.buffer(distance / 2)
        # Merge overlapping areas (produces a single or multi geometry)
        merged = buffered.unary_union
        # Sometimes unary_union returns a MultiPolygon; split them
        if hasattr(merged, "geoms"):
            for geom in merged.geoms:
                dissolved_parts.append({attr: value, "geometry": geom})
        else:
            dissolved_parts.append({attr: value, "geometry": merged})
    return gpd.GeoDataFrame(dissolved_parts, crs=gdf.crs)


def read_shapefile_clipped(shapefile_path, bbox, bbox_crs="EPSG:4326"):
    features = []
    with fiona.open(shapefile_path) as src:
        shp_crs = src.crs
        if shp_crs is None:
            raise ValueError(f"Missing CRS in {shapefile_path}")

        # Reproject bbox if needed
        if bbox_crs != shp_crs:
            transformer = Transformer.from_crs(bbox_crs, shp_crs, always_xy=True)
            minx, miny = transformer.transform(bbox[0], bbox[1])
            maxx, maxy = transformer.transform(bbox[2], bbox[3])
            bbox_proj = (minx, miny, maxx, maxy)
        else:
            bbox_proj = bbox
        for f in src.filter(bbox=bbox_proj):
            features.append({
                **f,
                "geometry": shape(f["geometry"])
            })

        gdf = gpd.GeoDataFrame.from_features(features, crs=src.crs)
        
    return gdf

def batch_clip_shapefiles(shapefile_dir, bbox, output_dir="clipped_geojson", bbox_crs="EPSG:4326"):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(shapefile_dir):
        if file.endswith(".shp"):
            base = os.path.splitext(file)[0]
            shp_path = os.path.join(shapefile_dir, file)
            out_path = os.path.join(output_dir, f"{base}.geojson")

            try:
                print(f"📍 Clipping {file}...")
                clipped = read_shapefile_clipped(shp_path, bbox, bbox_crs=bbox_crs)

                if not clipped.empty:
                    clipped.to_file(out_path, driver="GeoJSON")
                    print(f"✅ Saved {out_path}")
                else:
                    print(f"⚠️ No features within bbox for {file}")
            except Exception as e:
                print(f"❌ Error processing {file}: {e}")
        

def create_tile_grid(bbox_latlon, zoom, offset_right=0.0, offset_down=0.0):
    """ 
    Create tile grid GeoDataFrame with optional offset.

    Parameters:
        bbox_latlon (tuple): (min_lon, min_lat, max_lon, max_lat)
        zoom (int): Slippy map zoom level
        offset_down (float): % of tile height to shift down (0.5 = 50%)
        offset_right (float): % of tile width to shift right (0.5 = 50%)

    Returns:
        GeoDataFrame with columns z, x, y, geometry (EPSG:3857)
    """

    tiles = list(mercantile.tiles(*bbox_latlon, zoom))
    tile_geoms = []

    for tile in tiles:
        b = mercantile.xy_bounds(tile)
        xmin, ymin, xmax, ymax = b.left, b.bottom, b.right, b.top

        # Calculate tile size
        width = xmax - xmin
        height = ymax - ymin

        # Apply offsets
        dx = width * offset_right
        dy = height * offset_down

        xmin += dx
        xmax += dx
        ymin -= dy
        ymax -= dy

        tile_geoms.append({
            'z': tile.z,
            'x': tile.x,
            'y': tile.y,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'geometry': box(xmin, ymin, xmax, ymax)
        })

    return gpd.GeoDataFrame(tile_geoms, crs='EPSG:3857')

def calculate_tile_area(z,x,y):
    # Get tile bounds in Web Mercator (meters)
    bounds = mercantile.xy_bounds(x, y, z)  # left, bottom, right, top
    # Create bounding box geometry
    tile_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    # Area in m² (since Web Mercator is in meters)
    area_m2 = tile_box.area
    
    return area_m2


def assign_features_to_tiles(layer_gdf, tile_grid):
    bbox_union = tile_grid.unary_union
    clipped = gpd.clip(layer_gdf, bbox_union)
    joined = gpd.sjoin(clipped, tile_grid, how="inner", predicate="intersects")
    return joined


def render_tile(tile_info):
    z, x, y, geom, grouped_layers, layer_config, OUTPUT_DIR, offset_right, offset_down = tile_info
    bounds = geom.bounds
    
    fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.axis("off")
    ax.set_facecolor("white")


    for name, df in grouped_layers.items():
        features = df.query(f'z=={z} and x=={x} and y=={y}')
        if not features.empty:
            if name in layer_config.keys():
                stroke = layer_config[name]['stroke_color']
                fill = layer_config[name]['fill_color']
                linewidth = float(layer_config[name]['linewidth'])
            else:
                stroke = layer_config['landuse_complement'][name]['stroke_color']
                fill = layer_config['landuse_complement'][name]['fill_color']
                linewidth = float(layer_config['landuse_complement'][name]['linewidth'])
                
            if fill != 'None':
                features.plot(ax=ax, facecolor=fill, edgecolor=stroke, linewidth=linewidth)
            else:
                features.plot(ax=ax, color=stroke, linewidth=linewidth)

    out_dir = os.path.join(OUTPUT_DIR, str(z)+"+"+str(int(offset_right*10))+"+"+str(int(offset_down*10)), str(x))
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{y}.png"), bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()
    