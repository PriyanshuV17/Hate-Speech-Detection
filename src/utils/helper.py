# ==========================
# Helper Functions
# ==========================
from src.utils.config_loader import ConfigLoader

# Load labels from config.yaml
_config = ConfigLoader()
_label_map = _config.get_labels()

def map_label(label_id: int) -> str:
    """
    Map numeric label_id to its corresponding class name.

    Args:
        label_id (int): Numeric label ID from dataset

    Returns:
        str: Class name (e.g., "bullying", "offensive", etc.)
    """
    return _label_map.get(label_id, "unknown")