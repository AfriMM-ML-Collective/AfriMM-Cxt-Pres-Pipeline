"""
AfriCaption: Context-preserving translation pipeline for AfriMMD.

This package provides a modular pipeline for translating multilingual datasets
while preserving context and maintaining translation quality through semantic
similarity scoring.
"""

__version__ = "1.0.0"
__author__ = "AfriCaption Team"

from africaption.config import BASELINES, BASELINE_KEYWORDS, is_baseline, NON_LANG_COLS
from africaption.translators import TranslatorBase, MarianPerLang, NLLBLike
from africaption.scorer import LaBSEScorer
from africaption.dataset import load_afrimmd_dataset, language_columns, cosine_similarity
from africaption.pipeline import process_dataset
from africaption.adapter_loader import load_adapters

__all__ = [
    "BASELINES",
    "BASELINE_KEYWORDS",
    "is_baseline",
    "TranslatorBase",
    "MarianPerLang",
    "NLLBLike",
    "LaBSEScorer",
    "load_afrimmd_dataset",
    "language_columns",
    "cosine_similarity",
    "process_dataset",
]

