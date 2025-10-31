"""
Semantic similarity scoring module.

This module provides the LaBSE-based scorer for evaluating translation quality
through semantic similarity between source and translated text.
"""

from typing import List

import torch
from sentence_transformers import SentenceTransformer

from africaption.config import DEFAULT_SCORER


class LaBSEScorer:
    """
    LaBSE-based semantic similarity scorer.
    
    Uses Language-agnostic BERT Sentence Embedding (LaBSE) to compute
    cosine similarity between source and translated texts for quality assessment.
    
    Attributes:
        device: Computation device (cuda/cpu)
        model: Loaded SentenceTransformer model
        
    Examples:
        >>> scorer = LaBSEScorer()
        >>> embeddings = scorer.embed_norm(["Hello world", "Bonjour monde"])
        >>> similarity = cosine_similarity(embeddings[0], embeddings[1])
    """
    
    def __init__(self, model_name: str = DEFAULT_SCORER, device: str = None):
        """
        Initialize the LaBSE scorer.
        
        Args:
            model_name: HuggingFace model identifier for the scorer
            device: Computation device (auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
    
    def embed_norm(self, texts: List[str]):
        """
        Generate normalized embeddings for input texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Tensor of normalized embeddings
        """
        return self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)

