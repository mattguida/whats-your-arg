# What’s Your Argument? A Detailed Investigation on Argument Detection and Understanding with LLMs

This repository contains the code and data for the paper "What’s Your Argument? A Detailed Investigation on Argument Detection and Understanding with LLMs" (ACL 2025).

## Overview

We present a comprehensive analysis of large language models' capabilities in argument mining tasks using a dataset of over 2,000 opinion comments on polarizing topics. The study focuses on three main tasks:
1. Argument Detection
2. Argument Span Extraction
3. Argument Support/Attack Classification

## Repository Structure

```
├── data/
│   ├── raw/                # Original comment datasets
│   ├── processed/          # Preprocessed datasets
│   └── annotations/        # Human annotations for arguments
├── models/
│   ├── roberta/           # Fine-tuned RoBERTa models
│   └── llm/               # LLM prompts and configurations
├── src/
│   ├── preprocessing/     # Data preprocessing scripts
│   ├── training/          # Model training code
│   ├── evaluation/        # Evaluation scripts
│   └── utils/            # Helper functions
├── experiments/
│   ├── configs/          # Experiment configurations
│   └── results/          # Experimental results and logs
├── notebooks/            # Analysis notebooks
└── scripts/              # Utility scripts
```

## Requirements

```
python>=3.8
torch>=1.9.0
transformers>=4.21.0
datasets>=2.3.0
scikit-learn>=0.24.2
rouge-score>=0.1.2
accelerate>=0.12.0
```

## Dataset

The dataset consists of over 2,000 opinion comments covering six polarizing topics:
- Gay Marriage
- Marijuana Legalization
- [Other topics]

Each comment is annotated with:
- Presence of pre-defined arguments
- Argument spans
- Support/attack labels

The original datasets are from:
- Boltužić and Šnajder (2014)
- Hasan and Ng (2014)

## Models

We evaluate several models:
- Fine-tuned RoBERTa (baseline)
- GPT4o
- GPT4o-mini
- Gemini1.5-flash
- Llama3-8b

## Usage

### Data Preprocessing

```bash
# Preprocess raw data
python src/preprocessing/prepare_data.py

# Generate train/val/test splits
python src/preprocessing/create_splits.py
```

### Training

```bash
# Fine-tune RoBERTa
python src/training/train_roberta.py --config experiments/configs/roberta_config.json

# Run LLM experiments
python src/training/run_llm_experiments.py --model [model_name] --task [task_number]
```

### Evaluation

```bash
# Evaluate models
python src/evaluation/evaluate.py --model [model_name] --task [task_number]

# Generate result tables
python src/evaluation/generate_tables.py
```

## Results

Our main findings show:
- LLMs excel at support/attack classification
- Limited reliability in argument detection
- Inconsistent improvements with in-context learning
- Fine-tuned LLama outperforms prompt-based approaches

Detailed results can be found in the `experiments/results/` directory.

## Citation

```bibtex
[Paper citation will go here]
```


## Contact



## Acknowledgments

