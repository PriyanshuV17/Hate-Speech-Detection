def map_label(label):
    label_map = {0: "hate_speech", 1: "offensive_language", 2: "neither"}
    return label_map.get(label, "unknown")