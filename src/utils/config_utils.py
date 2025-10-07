# ==========================
# Config Utils
# ==========================
import os
import yaml

def load_config(config_path="config.yaml"):
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to the config file (default: "config.yaml")

    Returns:
        dict: Parsed configuration dictionary
    """
    # Ensure absolute path (so it works no matter where script is run from)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    full_path = os.path.join(root_dir, config_path)

    with open(full_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


# Example usage
if __name__ == "__main__":
    cfg = load_config()
    print("✅ Config loaded successfully!")
    print(cfg.keys())