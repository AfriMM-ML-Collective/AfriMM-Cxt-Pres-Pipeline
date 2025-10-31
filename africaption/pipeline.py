"""
Main processing pipeline for context-preserving translation updates.

This module orchestrates the complete pipeline: loading datasets, matching
languages with available translators, scoring translations, and maintaining
translation quality through semantic similarity comparison.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from africaption.config import DEFAULT_SCORER, DEFAULT_OUT_DIR
from africaption.dataset import load_afrimmd_dataset, language_columns, cosine_similarity
from africaption.scorer import LaBSEScorer
from africaption.translators import TranslatorBase
from africaption.adapter_loader import load_adapters


def process_dataset(
    models_yaml: str,
    out_dir: str = DEFAULT_OUT_DIR,
    scorer_name: str = DEFAULT_SCORER,
    allow_baselines: bool = False,
    max_samples: Optional[int] = None
) -> None:
    """
    Process the AfriMMD dataset with context-preserving translation updates.
    
    This is the main pipeline that:
    1. Loads and optionally limits the dataset
    2. Loads translation adapters from YAML configuration
    3. Identifies language overlaps between dataset and models
    4. For each sample, compares existing and new translations via semantic similarity
    5. Retains the higher-scoring translation (old or new)
    6. Writes outputs in JSONL and Parquet formats with audit trail
    
    Args:
        models_yaml: Path to YAML file describing translation models
        out_dir: Output directory for processed files
        scorer_name: HuggingFace model identifier for semantic scorer
        allow_baselines: Whether to allow baseline models (NLLB/M2M)
        max_samples: Limit number of samples for testing (None = all)
        
    Returns:
        None (writes output files to disk)
        
    Examples:
        >>> process_dataset("config/models_multilingual.yaml", max_samples=5)
    """
    # Load and prepare dataset
    ds = load_afrimmd_dataset()
    
    # Limit dataset for testing
    if max_samples is not None and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))
        print(f"[TEST MODE] Processing only {len(ds)} samples (requested: {max_samples})")
    
    # Load adapters and initialize scorer
    adapters = load_adapters(models_yaml, allow_baselines=allow_baselines)
    scorer = LaBSEScorer(model_name=scorer_name)
    lang_cols = language_columns(ds.column_names)
    
    # Build language-adapter mapping
    adapter_for_lang, candidate_langs = _build_language_mapping(adapters, lang_cols)
    
    # Report language overlap
    _report_language_overlap(lang_cols, adapters, candidate_langs)
    
    if not candidate_langs:
        print("No overlap between dataset languages and provided new model(s).")
        return
    
    # Prepare output paths
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_jsonl = Path(out_dir) / "afrimmd_updated.jsonl"
    out_parq = Path(out_dir) / "afrimmd_updated.parquet"
    
    # Process each sample
    rows = []
    for example in tqdm(ds, desc="Processing dataset"):
        row = dict(example)
        eng: str = (row.get("eng") or "").strip()
        if not eng:
            continue
        
        # Embed English once for efficiency
        emb_en = scorer.embed_norm([eng])[0]
        audit = {}
        
        # Process each candidate language
        for lang in candidate_langs:
            old_txt = (row.get(lang) or "").strip()
            old_score = None
            
            # Score existing translation if present
            if old_txt:
                old_emb = scorer.embed_norm([old_txt])[0]
                old_score = cosine_similarity(emb_en, old_emb)
            
            # Generate and score new translations from all adapters
            best_new_txt, best_new_score, best_model = _find_best_translation(
                adapters=adapter_for_lang[lang],
                scorer=scorer,
                emb_en=emb_en,
                eng_text=eng,
                lang=lang
            )
            
            # Decide which translation to keep
            kept = "old"
            if best_new_txt is not None and (old_score is None or best_new_score > old_score):
                row[lang] = best_new_txt
                kept = "new"
            
            # Record audit information
            audit[lang] = {
                "old": None if old_score is None else round(old_score, 4),
                "new": None if best_new_txt is None else round(best_new_score, 4),
                "kept": kept,
                "model": best_model if kept == "new" else "existing",
            }
        
        row["_audit"] = audit
        rows.append(row)
    
    # Write output files
    _write_outputs(out_jsonl, out_parq, rows)
    print(f"Saved:\n- {out_jsonl}\n- {out_parq}\nDone.")


def _build_language_mapping(
    adapters: List[TranslatorBase],
    lang_cols: List[str]
) -> Tuple[Dict[str, List[TranslatorBase]], List[str]]:
    """
    Build mapping of languages to available adapters.
    
    Args:
        adapters: List of loaded translator adapters
        lang_cols: List of language columns in the dataset
        
    Returns:
        Tuple of (adapter_for_lang dict, candidate_langs list)
    """
    adapter_for_lang: Dict[str, List[TranslatorBase]] = {lc: [] for lc in lang_cols}
    
    for adapter in adapters:
        for lang in adapter.supported_langs():
            if lang in adapter_for_lang:
                adapter_for_lang[lang].append(adapter)
    
    # Only process languages with at least one adapter
    candidate_langs = [lc for lc in lang_cols if adapter_for_lang[lc]]
    
    return adapter_for_lang, candidate_langs


def _report_language_overlap(
    lang_cols: List[str],
    adapters: List[TranslatorBase],
    candidate_langs: List[str]
) -> None:
    """
    Print detailed language overlap analysis.
    
    Args:
        lang_cols: All language columns in dataset
        adapters: Loaded translator adapters
        candidate_langs: Languages that will be processed
    """
    all_model_langs = set()
    for adapter in adapters:
        all_model_langs.update(adapter.supported_langs())
    
    print("\n" + "="*60)
    print("LANGUAGE OVERLAP ANALYSIS")
    print("="*60)
    print(f"Dataset languages ({len(lang_cols)}): {sorted(lang_cols)}")
    print(f"Model-supported languages ({len(all_model_langs)}): {sorted(all_model_langs)}")
    print(f"\n✓ OVERLAPPING languages to process ({len(candidate_langs)}): {sorted(candidate_langs)}")
    
    if len(lang_cols) > len(candidate_langs):
        missing = sorted(set(lang_cols) - set(candidate_langs))
        print(f"✗ Languages in dataset NOT supported by model ({len(missing)}): {missing}")
    
    print("="*60 + "\n")


def _find_best_translation(
    adapters: List[TranslatorBase],
    scorer: LaBSEScorer,
    emb_en,
    eng_text: str,
    lang: str
) -> Tuple[Optional[str], float, Optional[str]]:
    """
    Find the best translation across all adapters for a language.
    
    Args:
        adapters: List of adapters supporting this language
        scorer: Semantic similarity scorer
        emb_en: Pre-computed English embedding
        eng_text: English source text
        lang: Target language code
        
    Returns:
        Tuple of (best_text, best_score, best_model_name)
    """
    best_new_txt = None
    best_new_score = -math.inf
    best_model = None
    
    for adapter in adapters:
        try:
            candidate = adapter.translate(eng_text, lang)
            if not candidate:
                continue
            
            cand_emb = scorer.embed_norm([candidate])[0]
            score = cosine_similarity(emb_en, cand_emb)
            
            if score > best_new_score:
                best_new_txt = candidate
                best_new_score = score
                best_model = adapter.name
                
        except Exception as e:
            print(f"[WARN] {adapter.name} failed on {lang}: {e}")
    
    return best_new_txt, best_new_score, best_model


def _write_outputs(out_jsonl: Path, out_parq: Path, rows: List[Dict]) -> None:
    """
    Write processed results to JSONL and Parquet formats.
    
    Args:
        out_jsonl: Path for JSONL output file
        out_parq: Path for Parquet output file
        rows: List of processed row dictionaries
    """
    # Write JSONL
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    # Write Parquet
    pd.DataFrame(rows).to_parquet(out_parq, index=False)

