# src/utils/config_loader.py
import os
import yaml

class ConfigLoader:
    def __init__(self, config_path="config.yaml"):
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        with open(os.path.join(self.root_dir, config_path), "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def _resolve_path(self, value):
        """Resolve relative paths to absolute, handle nested dicts."""
        if isinstance(value, str):
            if not os.path.isabs(value):
                return os.path.join(self.root_dir, value)
            return value
        elif isinstance(value, dict):
            return {k: self._resolve_path(v) for k, v in value.items()}
        else:
            return value

    def get_paths(self):
        return self._resolve_path(self.config.get("paths", {}))

    def get_processed_data_paths(self):
        return self._resolve_path(self.config.get("processed_data", {}))

    def get_saved_model_paths(self):
        return self._resolve_path(self.config.get("saved_models", {}))

    def get_training_params(self):
        return self.config.get("training", {})

    def get_labels(self):
        return self.config.get("labels", {})