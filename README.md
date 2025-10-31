## AFRICAPTION

### Context Preserving Pipeline

[![arXiv](https://img.shields.io/badge/arXiv-2510.17405-red)](https://arxiv.org/abs/2510.17405)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hugging Face Dataset](https://img.shields.io/badge/HuggingFace-AfriMM%2FAfriMMD-blue)](https://huggingface.co/datasets/AfriMM/AfriMMD)

This repository presents the AfriCaption pipeline, a robust system designed to enhance data quality and ensure continuous improvement in African NLP through advanced translation and evaluation mechanisms. It addresses the critical need for high-quality data in African NLP, where existing machine translation models often exhibit inconsistent performance across languages.

<p align="center">
  <img src="external_assets/pipeline_shot.png" alt="Context Preserving Pipeline overview" width="820" />
</p>


### Project Structure

```
Cxt-pres-pipeline/
├── africaption/          # Main package module
│   ├── __init__.py      # Package initialization and exports
│   ├── config.py        # Configuration constants and baseline definitions
│   ├── translators.py   # Translation adapter classes (Marian, NLLB-like)
│   ├── scorer.py        # Semantic similarity scoring (LaBSE)
│   ├── dataset.py       # Dataset loading and utilities
│   ├── adapter_loader.py # YAML configuration loader
│   ├── pipeline.py      # Main processing pipeline
│   └── cli.py           # Command-line interface
├── config/              # Configuration files
│   ├── models_multilingual.yaml
│   └── models_marian.yaml
├── outputs/             # Generated output files
├── main.py              # Entry point script
├── test.py              # Quick test script (5 samples)
├── run.py               # Full dataset processing script
└── README.md
```

### Quick Start

**Test with 5 samples:**
```bash
python test.py
```

**Process entire dataset:**
```bash
python run.py
```

**Custom usage:**
```bash
python main.py --test                                    # Test mode
python main.py --models-yaml config/models.yaml         # Custom config
python main.py --out-dir custom_outputs                 # Custom output
```

### Usage

The pipeline automatically:
- Detects model configuration files (`config/models_multilingual.yaml` by default)
- Identifies language overlaps between dataset and models
- Compares translation quality via semantic similarity (LaBSE)
- Retains higher-scoring translations (old or new)
- Generates outputs in JSONL and Parquet formats with audit trails

### Configuration

Model configurations are defined in YAML files under `config/`:

**Multilingual NLLB-like model:**
```yaml
type: "nllb_like"
model: "facebook/nllb-200-distilled-600M"
lang_tokens:
  afr: "afr_Latn"
  amh: "amh_Ethi"
  ...
```

**Per-language Marian models:**
```yaml
models:
  afr: "Helsinki-NLP/opus-mt-en-af"
  amh: "Helsinki-NLP/opus-mt-en-am"
  ...
```

### Output

The pipeline generates:
- `outputs/afrimmd_updated.jsonl` - JSONL format with audit trail
- `outputs/afrimmd_updated.parquet` - Parquet format for analysis

Each record includes an `_audit` field with translation quality scores and retention decisions.

### Citations

If you use this pipeline or dataset, please cite:

```bibtex
@misc{oduwole2025africaptionestablishingnewparadigm,
  title={AFRICAPTION: Establishing a New Paradigm for Image Captioning in African Languages}, 
  author={Mardiyyah Oduwole and Prince Mireku and Fatimo Adebanjo and Oluwatosin Olajide and Mahi Aminu Aliyu and Jekaterina Novikova},
  year={2025},
  eprint={2510.17405},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2510.17405}
}
```
