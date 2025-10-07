import os
import yaml

class ConfigLoader:
    def __init__(self, config_filename="config.yaml"):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        config_path = os.path.join(root_dir, config_filename)
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get_paths(self):
        return self.config.get("paths", {})

    def get_processed_data_paths(self):
        return self.config.get("paths", {}).get("processed_data", {})

    def get_saved_model_paths(self):
        return self.config.get("paths", {}).get("saved_models", {})

    def get_training_params(self):
        return self.config.get("training", {})

    def get_logging_config(self):
        return self.config.get("logging", {})