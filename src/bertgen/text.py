"""Text normalization shared by deduplication sites."""

import re
import unicodedata


def normalize(text: str) -> str:
    """NFKC, casefold and whitespace-collapse `text` to build the dedup key."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", folded).strip()
