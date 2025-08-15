from pathlib import Path
import pandas as pd
import yaml

def load_config(path_root: Path, config_path: str = "config/config.yaml"):
    full_config_path = path_root / config_path
    try:
        with open(full_config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: The configuration file was not found")
        return None

def load_raw_data(path_root: Path, config: dict):
    raw_path = path_root / config["paths"]["raw"]
    try:
        data_base = pd.read_csv(raw_path)
        return data_base
    except FileNotFoundError:
        print("Error: The raw data file was not found")
        return None
