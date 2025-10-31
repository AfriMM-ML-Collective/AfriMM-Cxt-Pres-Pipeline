"""
Configuration constants and baseline model definitions.

This module defines baseline models (which are not re-executed by default)
and provides utilities for baseline detection.
"""

from typing import Set

# Baseline model definitions
BASELINES = {
    "m2m100": "facebook/m2m100_418M",
    "nllb200": "facebook/nllb-200-distilled-600M",
}

# Keywords that identify baseline models in model IDs
BASELINE_KEYWORDS: Set[str] = {"m2m100", "nllb"}

# Non-language columns in the dataset
NON_LANG_COLS = {"id", "image_id", "eng", "_audit"}

# Default scoring model
DEFAULT_SCORER = "sentence-transformers/LaBSE"

# Default output directory
DEFAULT_OUT_DIR = "outputs"


def is_baseline(model_id: str) -> bool:
    """
    Check if a model ID corresponds to a baseline model.
    
    Args:
        model_id: HuggingFace model identifier or path
        
    Returns:
        True if the model is a baseline (M2M-100 or NLLB-200), False otherwise
        
    Examples:
        >>> is_baseline("facebook/nllb-200-distilled-600M")
        True
        >>> is_baseline("my-custom-model")
        False
    """
    if not model_id:
        return False
    mid = model_id.lower()
    return (model_id in BASELINES.values()) or any(k in mid for k in BASELINE_KEYWORDS)

