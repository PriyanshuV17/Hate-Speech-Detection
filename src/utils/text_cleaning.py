# ==========================
# Text Cleaning Utilities
# ==========================
import re
import string
import emoji

def clean_text(text: str, aggressive: bool = False) -> str:
    """
    Clean text for preprocessing.

    Args:
        text (str): Input text
        aggressive (bool): If True, applies heavy cleaning (URLs, mentions, punctuation, digits).
                           If False, only trims whitespace and normalizes spacing.

    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""

    if aggressive:
        # Lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        # Remove mentions and hashtags
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#\w+", "", text)
        # Remove emojis
        text = emoji.replace_emoji(text, replace='')
        # Remove punctuation and digits
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r"\d+", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text