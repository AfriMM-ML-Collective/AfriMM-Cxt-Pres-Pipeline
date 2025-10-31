"""
Translation adapters for different model architectures.

This module provides translator classes that abstract over different
multilingual translation model types (Marian per-language and NLLB-like).
"""

from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from africaption.config import is_baseline


class TranslatorBase:
    """
    Abstract base class for translation adapters.
    
    All translator implementations must provide:
    - supported_langs(): List of language codes supported
    - translate(text, lang): Translate text to target language
    - name: Property identifying the translator type
    """
    
    def supported_langs(self) -> List[str]:
        """Return list of supported language codes."""
        raise NotImplementedError
    
    def translate(self, text: str, lang: str) -> str:
        """
        Translate text from English to target language.
        
        Args:
            text: English source text
            lang: Target language code
            
        Returns:
            Translated text string
        """
        raise NotImplementedError
    
    @property
    def name(self) -> str:
        """Return identifier for this translator type."""
        raise NotImplementedError


class MarianPerLang(TranslatorBase):
    """
    Per-language Marian MT translator.
    
    Uses separate Helsinki-NLP Marian models for each target language.
    Configured via YAML with language-to-model mapping.
    
    YAML Configuration:
        models:
          afr: "Helsinki-NLP/opus-mt-en-af"
          amh: "Helsinki-NLP/opus-mt-en-am"
          ...
    
    Attributes:
        device: Computation device (cuda/cpu)
        mapping: Dictionary mapping language codes to model IDs
        _cache: Internal cache for loaded tokenizer/model pairs
    """
    
    def __init__(self, mapping: Dict[str, str], device: Optional[str] = None):
        """
        Initialize Marian per-language translator.
        
        Args:
            mapping: Dictionary mapping language codes to HuggingFace model IDs
            device: Computation device (auto-detected if None)
            
        Note:
            Baseline check is handled by load_adapters() before instantiation.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mapping = mapping
        self._cache: Dict[str, Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]] = {}
    
    @property
    def name(self) -> str:
        """Return translator identifier."""
        return "MarianPerLang"
    
    def supported_langs(self) -> List[str]:
        """Return list of supported language codes."""
        return list(self.mapping.keys())
    
    def _get(self, lang: str) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
        """
        Get or load tokenizer and model for a language.
        
        Models are cached after first load to avoid redundant loading.
        
        Args:
            lang: Language code
            
        Returns:
            Tuple of (tokenizer, model) for the language
        """
        if lang not in self._cache:
            model_id = self.mapping[lang]
            tok = AutoTokenizer.from_pretrained(model_id)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)
            self._cache[lang] = (tok, mdl)
        return self._cache[lang]
    
    def translate(self, text: str, lang: str) -> str:
        """
        Translate English text to target language.
        
        Args:
            text: English source text
            lang: Target language code
            
        Returns:
            Translated text string
            
        Raises:
            KeyError: If language is not in mapping
        """
        tokenizer, model = self._get(lang)
        enc = tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        gen = model.generate(
            **enc, 
            max_new_tokens=96, 
            num_beams=4, 
            no_repeat_ngram_size=3
        )
        return tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()


class NLLBLike(TranslatorBase):
    """
    NLLB-style multilingual translator.
    
    Uses a single multilingual model with language-specific tokens to force
    translation direction. Supports models like NLLB, M2M-100, etc.
    
    YAML Configuration:
        type: "nllb_like"
        model: "facebook/nllb-200-distilled-600M"
        lang_tokens:
          afr: "afr_Latn"
          amh: "amh_Ethi"
          ...
    
    Attributes:
        device: Computation device (cuda/cpu)
        tokenizer: Pre-loaded tokenizer
        model: Pre-loaded translation model
        lang_tokens: Mapping from language codes to model-specific language tokens
    """
    
    def __init__(
        self, 
        model_id: str, 
        lang_tokens: Dict[str, str], 
        device: Optional[str] = None
    ):
        """
        Initialize NLLB-like multilingual translator.
        
        Args:
            model_id: HuggingFace model identifier
            lang_tokens: Dictionary mapping language codes to model language tokens
            device: Computation device (auto-detected if None)
            
        Raises:
            ValueError: If model cannot be loaded (invalid ID, authentication required, etc.)
            
        Note:
            Baseline check is handled by load_adapters() before instantiation.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lang_tokens = lang_tokens
        
        try:
            self.tok = AutoTokenizer.from_pretrained(model_id)
            self.mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)
        except (OSError, Exception) as e:
            raise ValueError(
                f"Failed to load model '{model_id}'. "
                f"Is it a valid HuggingFace model ID or local path? "
                f"If it's a private/gated repo, ensure you're authenticated with `huggingface-cli login`. "
                f"Original error: {e}"
            ) from e
    
    @property
    def name(self) -> str:
        """Return translator identifier."""
        return "NLLBLike"
    
    def supported_langs(self) -> List[str]:
        """Return list of supported language codes."""
        return list(self.lang_tokens.keys())
    
    def translate(self, text: str, lang: str) -> str:
        """
        Translate English text to target language using forced language token.
        
        Args:
            text: English source text
            lang: Target language code
            
        Returns:
            Translated text string
            
        Raises:
            ValueError: If language is not mapped in lang_tokens
        """
        if lang not in self.lang_tokens:
            raise ValueError(f"Language {lang} not mapped in lang_tokens.")
        
        forced_token_id = self.tok.convert_tokens_to_ids(self.lang_tokens[lang])
        enc = self.tok(text, return_tensors="pt").to(self.device)
        gen = self.mdl.generate(
            **enc,
            forced_bos_token_id=forced_token_id,
            max_new_tokens=96,
            num_beams=4,
            no_repeat_ngram_size=3
        )
        return self.tok.batch_decode(gen, skip_special_tokens=True)[0].strip()

