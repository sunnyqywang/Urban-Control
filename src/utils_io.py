import os
from dotenv import load_dotenv
import re
import yaml


import pandas as pd
import geopandas as gpd
import fiona
from shapely.geometry import shape
from argparse import Namespace

def save_args_to_yaml(input_args, file_path):
    """
    Save argparse arguments to a YAML file.
    
    Args:
        input_args (Namespace): Parsed arguments from argparse
        file_path (str): Path to the YAML file to save
    """
    # Convert Namespace to dictionary
    if isinstance(input_args, Namespace):
        args_dict = vars(input_args)
    else:
        args_dict = input_args  # in case it's already a dictionary

    save_path = os.path.join(input_args.output_dir, file_path)
    os.makedirs(input_args.output_dir, exist_ok=True)

    # Write to YAML file
    with open(save_path, 'w') as f:
        yaml.dump(args_dict, f, default_flow_style=False)

def load_env_variables():
    """
    Load environment variables from `.env` in local development.
    """
    
    # Check if running locally by checking for the presence of `.env`
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    dotenv_path = os.path.join(parent_directory, ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
    else:
        print("ERROR")
        return None
        
    # Retrieve all environment variables
    env_variables = dict(os.environ)

    # Return all environment variables as a dictionary
    return env_variables


def _yaml_parse_environ(yaml_module, env_variables):
    """
    Parse expressions of the form << ENVIRON_VARIABLE >> in YAML files using
    the provided environment variables.
    """
    
    pattern = re.compile(r"^(.*)\<\<(.*)\>\>(.*)$")
    yaml_module.add_implicit_resolver("!pathex", pattern)


    def pathex_constructor(loader, node):
        value = loader.construct_scalar(node)
        left, env_var, right = pattern.match(value).groups()
        env_var = env_var.strip()
        if env_var not in env_variables:
            msg = f"Environment variable {env_var} not defined"
            raise ValueError(msg)
        return left + env_variables[env_var] + right

    yaml_module.add_constructor("!pathex", pathex_constructor)

    return yaml_module


def load_config(config_path):
    """
    Load a YAML configuration file with support for dynamic environment variable substitution.
    """
    
    # Load all environment variables using the unified function
    env_variables = load_env_variables()

    with open(config_path, "r") as ymlfile:
        full_cfg = _yaml_parse_environ(yaml, env_variables).load(
           ymlfile, Loader=_yaml_parse_environ(yaml, env_variables).FullLoader
        )

    return full_cfg


def load_all(files, params=None):
    all_data = {}
    
    for name, filepath in files.items():
        print("Loading ... ", name)
        df = load_generic(filepath, params)
        all_data[name] = df

    return all_data

def load_generic(filepath, params=None):
    if params is None:
        params = {}
        
    _, file_extension = os.path.splitext(filepath)

    if file_extension == ".csv":
        df = pd.read_csv(filepath, **params)

    elif file_extension == ".parquet":
        df = pd.read_parquet(filepath, **params)

    elif file_extension == ".pkl" or file_extension == ".pickle":
        with open(filepath, "rb") as handle:
            df = pickle.load(filepath, **params)

    elif file_extension == ".xlsx":
        df = pd.read_excel(filepath, **params)

    elif file_extension == ".shp":
        # Spatial filter (lon/lat bounding box)
        with fiona.open(filepath) as src:
            filtered = [
                {**f, "geometry": shape(f["geometry"])}
                for f in src.filter(bbox=params['bbox'])
            ]
        df = gpd.GeoDataFrame.from_features(filtered, crs=src.crs)
    
    elif file_extension == '.geojson':
        df = gpd.read_file(filepath)
        
    else:
        msg = f"Reading a {file_extension} to pandas is not yet supported"
        raise ValueError(msg)

    return df



