"""
Adapter loader for translation models from YAML configuration.

This module handles loading and instantiating translator adapters from
YAML configuration files, supporting both Marian per-language and
NLLB-like multilingual model configurations.
"""

import sys
from pathlib import Path
from typing import List, Optional

import yaml

from africaption.config import is_baseline
from africaption.translators import TranslatorBase, MarianPerLang, NLLBLike


def load_adapters(
    models_yaml: Optional[str],
    allow_baselines: bool = False
) -> List[TranslatorBase]:
    """
    Load translation adapters from YAML configuration file.
    
    Supports two configuration styles:
    1. Marian per-language: Dictionary mapping languages to model IDs
    2. NLLB-like: Single multilingual model with language tokens
    
    Args:
        models_yaml: Path to YAML configuration file
        allow_baselines: Whether to allow baseline models (default: False)
        
    Returns:
        List of initialized translator adapters
        
    Raises:
        FileNotFoundError: If YAML file cannot be located
        SystemExit: If no valid adapters are found
        
    Examples:
        >>> adapters = load_adapters("config/models_multilingual.yaml")
        >>> len(adapters)
        1
    """
    if not models_yaml:
        print("""
            =====================================
            Did not detect a new model
            =====================================
        """)
        sys.exit(0)
    
    # Resolve YAML path
    yaml_path = _resolve_yaml_path(models_yaml)
    if not yaml_path:
        raise FileNotFoundError(f"Could not locate models YAML: {models_yaml}")
    
    # Load and parse configuration
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    adapters: List[TranslatorBase] = []
    
    # Style A: Per-language Marian models
    if isinstance(cfg.get("models"), dict) and cfg["models"]:
        mapping = _filter_baselines(cfg["models"], allow_baselines)
        if mapping:
            adapters.append(MarianPerLang(mapping))
    
    # Style B: Multilingual NLLB-like model
    if cfg.get("type") == "nllb_like":
        model_id = cfg.get("model")
        lang_tokens = cfg.get("lang_tokens") or {}
        
        if model_id and lang_tokens:
            if allow_baselines or not is_baseline(model_id):
                adapters.append(NLLBLike(model_id, lang_tokens))
    
    # Validate that at least one adapter was loaded
    if not adapters:
        print("""
            =====================================
            Did not detect a new model
            =====================================
        """)
        sys.exit(0)
    
    return adapters


def _resolve_yaml_path(models_yaml: str) -> Optional[Path]:
    """
    Resolve YAML file path from various sources.
    
    Checks:
    1. Direct path (absolute or relative to CWD)
    2. Relative to script directory
    
    Args:
        models_yaml: Path string to resolve
        
    Returns:
        Resolved Path object, or None if not found
    """
    yaml_path = Path(models_yaml)
    if yaml_path.is_file():
        return yaml_path
    
    # Try relative to script directory
    script_dir = Path(__file__).parent.parent
    yaml_path = script_dir / models_yaml
    if yaml_path.is_file():
        return yaml_path
    
    return None


def _filter_baselines(
    mapping: dict,
    allow_baselines: bool
) -> dict:
    """
    Filter baseline models from language-to-model mapping.
    
    Args:
        mapping: Dictionary mapping language codes to model IDs
        allow_baselines: Whether to allow baseline models
        
    Returns:
        Filtered mapping dictionary
    """
    if allow_baselines:
        return mapping
    return {k: v for k, v in mapping.items() if not is_baseline(v)}

