"""
Command-line interface for the AfriCaption pipeline.

Provides auto-detection of configuration files, baseline model detection,
and simplified command-line arguments.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import yaml

from africaption.config import BASELINE_KEYWORDS, is_baseline
from africaption.pipeline import process_dataset


def find_models_yaml(custom_path: Optional[str] = None) -> Optional[str]:
    """
    Auto-detect models YAML configuration file.
    
    Searches for common configuration file names in the project directory:
    - models_multilingual.yaml/yml
    - models.yaml/yml
    - models_marian.yaml/yml
    
    Args:
        custom_path: Optional custom path to override auto-detection
        
    Returns:
        Path to found YAML file, or None if not found
        
    Examples:
        >>> yaml_path = find_models_yaml()
        >>> print(yaml_path)
        models_multilingual.yaml
    """
    if custom_path:
        return custom_path
    
    script_dir = Path(__file__).parent.parent
    common_names = [
        "models_multilingual.yaml",
        "models_multilingual.yml",
        "models.yaml",
        "models.yml",
        "models_marian.yaml",
        "models_marian.yml"
    ]
    
    # Check in script directory
    for name in common_names:
        path = script_dir / name
        if path.is_file():
            return str(path)
    
    # Check in config subdirectory
    config_dir = script_dir / "config"
    if config_dir.is_dir():
        for name in common_names:
            path = config_dir / name
            if path.is_file():
                return str(path)
    
    return None


def auto_detect_baseline_need(models_yaml: str) -> bool:
    """
    Detect if configuration requires baseline models to be enabled.
    
    Checks the YAML configuration for baseline model IDs (NLLB, M2M-100)
    and determines if --allow-baselines flag should be automatically enabled.
    
    Args:
        models_yaml: Path to YAML configuration file
        
    Returns:
        True if baseline models are detected, False otherwise
    """
    try:
        yaml_path = Path(models_yaml)
        if not yaml_path.is_file():
            script_dir = Path(__file__).parent.parent
            yaml_path = script_dir / models_yaml
            config_dir = script_dir / "config" / models_yaml
            if config_dir.is_file():
                yaml_path = config_dir
        
        if yaml_path.is_file():
            cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            
            # Check NLLB-like model
            if cfg.get("type") == "nllb_like":
                model_id = cfg.get("model", "").lower()
                if any(keyword in model_id for keyword in BASELINE_KEYWORDS):
                    return True
            
            # Check Marian models
            if isinstance(cfg.get("models"), dict):
                for model_id in cfg["models"].values():
                    if is_baseline(model_id):
                        return True
    except Exception:
        pass
    
    return False


def parse_args():
    """
    Parse command-line arguments with sensible defaults.
    
    Returns:
        Parsed arguments namespace
    """
    ap = argparse.ArgumentParser(
        "AfriCaption: Context-preserving translation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Process AfriMMD dataset with context-preserving translation updates.
Automatically detects model configurations and handles baseline models.
        """,
        epilog="""
Examples:
  python main.py                    # Auto-detect config, process all data
  python main.py --test             # Test with 5 samples
  python main.py --models config/models.yaml --test
        """
    )
    
    # Auto-detect or require models YAML
    default_yaml = find_models_yaml()
    ap.add_argument(
        "--models-yaml",
        default=default_yaml,
        help="YAML describing model(s) to use (default: auto-detect)"
    )
    
    # Simple test mode
    ap.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only 5 samples"
    )
    
    # Other options
    ap.add_argument(
        "--out-dir",
        default="outputs",
        help="Output directory (default: outputs)"
    )
    ap.add_argument(
        "--scorer",
        default="sentence-transformers/LaBSE",
        help="Similarity model (default: LaBSE)"
    )
    ap.add_argument(
        "--allow-baselines",
        action="store_true",
        help="Allow baseline models (NLLB/M2M) - auto-detected if needed"
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples (overrides --test)"
    )
    
    return ap.parse_args()


def main():
    """
    Main CLI entry point.
    
    Handles argument parsing, baseline auto-detection, and invokes
    the main processing pipeline.
    """
    args = parse_args()
    
    # Handle test mode
    max_samples = 5 if args.test else args.max_samples
    
    # Auto-detect if baselines are needed
    if not args.allow_baselines and args.models_yaml:
        if auto_detect_baseline_need(args.models_yaml):
            print("[INFO] Detected baseline model(s). Enabling --allow-baselines automatically.")
            args.allow_baselines = True
    
    # Validate configuration file exists
    if not args.models_yaml:
        print("ERROR: No models YAML found. Please specify --models-yaml")
        script_dir = Path(__file__).parent.parent
        available = list(script_dir.glob("models*.yaml")) + list(script_dir.glob("config/models*.yaml"))
        if available:
            print("Available files:", [f.name for f in available])
        sys.exit(1)
    
    # Run processing pipeline
    process_dataset(
        args.models_yaml,
        args.out_dir,
        scorer_name=args.scorer,
        allow_baselines=args.allow_baselines,
        max_samples=max_samples
    )

