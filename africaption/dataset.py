"""
Dataset loading and utility functions.

This module handles dataset loading from HuggingFace, language column
identification, and semantic similarity computations.
"""

from typing import List, Set

from datasets import load_dataset, concatenate_datasets, Dataset
from sentence_transformers import util as sbert_util

from africaption.config import NON_LANG_COLS


def load_afrimmd_dataset() -> Dataset:
    """
    Load and concatenate the AfriMMD dataset from HuggingFace.
    
    Automatically downloads the dataset if not cached and concatenates
    all available splits (train, validation, test) into a single dataset.
    
    Returns:
        Concatenated dataset containing all splits
        
    Raises:
        ValueError: If dataset is empty or cannot be loaded
        
    Examples:
        >>> dataset = load_afrimmd_dataset()
        >>> print(f"Loaded {len(dataset)} samples")
    """
    print("Loading AfriMMD dataset...")
    dataset_dict = load_dataset("AfriMM/AfriMMD")
    
    # Concatenate all splits into one dataset
    splits = list(dataset_dict.keys())
    print(f"Found splits: {splits}")
    
    if len(splits) == 1:
        ds = dataset_dict[splits[0]]
    else:
        ds = concatenate_datasets([dataset_dict[split] for split in splits])
        print(f"Concatenated {len(splits)} splits into single dataset")
    
    if len(ds) == 0:
        raise ValueError("Loaded dataset is empty")
    
    print(f"Total samples loaded: {len(ds)}")
    return ds


def language_columns(example_keys: List[str]) -> List[str]:
    """
    Extract language column names from dataset column names.
    
    Filters out non-language columns (id, image_id, eng, _audit) to identify
    which columns contain translations in different languages.
    
    Args:
        example_keys: List of all column names from the dataset
        
    Returns:
        List of language column identifiers
        
    Examples:
        >>> cols = ["id", "eng", "afr", "amh", "image_id"]
        >>> language_columns(cols)
        ['afr', 'amh']
    """
    return [k for k in example_keys if k not in NON_LANG_COLS]


def cosine_similarity(emb_a, emb_b) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        emb_a: First embedding tensor
        emb_b: Second embedding tensor
        
    Returns:
        Cosine similarity score as a float
        
    Examples:
        >>> emb1 = scorer.embed_norm(["Hello"])[0]
        >>> emb2 = scorer.embed_norm(["Bonjour"])[0]
        >>> score = cosine_similarity(emb1, emb2)
    """
    return float(sbert_util.cos_sim(emb_a, emb_b).item())

