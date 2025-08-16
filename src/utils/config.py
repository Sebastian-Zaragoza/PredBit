from pathlib import Path
import yaml

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

def load_config(path = "config/config.yaml"):
    project_root = get_project_root()
    full_config_path = project_root/path
    try:
        with open(full_config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print("Error: The configuration file was not found.")
        return None