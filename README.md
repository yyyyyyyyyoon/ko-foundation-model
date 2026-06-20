# Ko-LLM

Korean-focused language model pretraining and summary fine-tuning codebase.

This repository contains the model architecture, tokenizer-related files, pretraining script, summary supervised fine-tuning script, evaluation script, and utility scripts for Korean language model experiments.

The project builds a decoder-only autoregressive language model and extends it to a Korean document summarization model through supervised fine-tuning.

The original training datasets and model checkpoints are not included in this repository.

---
## Model Overview
| Item | Setting |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters | 1.2B |
| Layers | 24 |
| Hidden Size | 2,048 |
| FFN Intermediate Size | 5,632 |
| Attention Heads | 16 |
| Key/Value Heads | 4 (GQA) |
| Context Length | 4,096 tokens |
| Vocabulary Size | 64,000 |
---
## Initial Setup

### Installation

Create a Python virtual environment and install the required packages.

```bash
# 1. Create a conda virtual environment
conda create -n ko-llm python=3.11 -y
conda activate ko-llm

# 2. Install required Python packages
python -m pip install -r requirements.txt
```


---

## Project Structure

```text
ko-llm/
├── README.md
├── requirements.txt
├── .gitignore
│
├── base_model.py
├── project_paths.py
│
├── tokenizer_bpe_64k/
│   ├── merges.txt
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.json
│
└── scripts/
    ├── data/
    │   ├── build_tokenizer_train_data.py
    │   └── build_packed_dataset.py
    │
    ├── tokenizer/
    │   ├── build_tokenizer.py
    │   ├── evaluate_tokenizer.py
    │   └── create_tokenizer_test_split.py
    │
    ├── train/
    │   ├── train_base_model.py
    │   └── train_summary_model.py
    │
    ├── eval/
    │   ├── eval_base_model.py
    │   └── eval_summary_model.py
    │
    └── plot/
        ├── plot_base_loss.py
        └── plot_summary_loss.py
```

---

## Path Configuration

Common project paths are defined in `project_paths.py`.

By default, paths are resolved relative to the project root.

```python
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_ROOT = PROJECT_ROOT / "dataset"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

TOKENIZER_DIR = PROJECT_ROOT / "tokenizer_bpe_64k"
PACKED_DATA_DIR = DATA_ROOT / "packed_corpus_4k"
```

The training and evaluation scripts import common paths from `project_paths.py`.

```python
from project_paths import PACKED_DATA_DIR, TOKENIZER_DIR, OUTPUT_ROOT
```

---

## Tokenizer

Tokenizer files may be committed if they are required for running the training or evaluation scripts and are small enough to store in Git.

Expected tokenizer directory:

```text
tokenizer_bpe_64k/
├── merges.txt
├── tokenizer.json
├── tokenizer_config.json
└── vocab.json
```

If the tokenizer is stored under the dataset directory, make sure `TOKENIZER_DIR` in `project_paths.py` points to the correct location.

---

## Dataset Preparation

### Packed Corpus for Base Model Pretraining

The base model pretraining script expects packed token blocks under:

```text
dataset/
└── packed_corpus_4k/
    ├── train_*.pt
    └── eval.pt
```

Each packed file should contain tokenized blocks with sequence length matching `MAX_LENGTH`.

Default block size:

```text
MAX_LENGTH = 4096
```

### Summary SFT Dataset

The summary fine-tuning script expects JSONL files under:

```text
dataset/
└── sft/
    └── summary/
        ├── train.jsonl
        ├── valid.jsonl
        └── test.jsonl
```

Each JSONL sample should contain at least:

```json
{
  "prompt": "input text",
  "response": "target summary"
}
```

---

## Train Base Model

Run base model pretraining from the project root:

```bash
PYTHONPATH=. python3 scripts/train/train_base_model.py
```

Main training settings, such as sequence length, batch size, gradient accumulation, learning rate, evaluation interval, and checkpoint interval, are defined at the top of `scripts/train/train_base_model.py`.

The script expects packed pretraining data under the local dataset directory defined in `project_paths.py`.

To train from scratch:
```python
RESUME_CHECKPOINT = None
```

To resume from a previous checkpoint:

```python
RESUME_CHECKPOINT = OUTPUT_DIR / "checkpoints" / "step_XXXXX"
```

---

## Train Summary SFT Model

Run summary supervised fine-tuning from the project root:

```bash
PYTHONPATH=. python3 scripts/train/train_summary_model.py
```

Before running, check the base checkpoint path and summary dataset paths at the top of `scripts/train/train_summary_model.py`.

```python
BASE_CHECKPOINT_DIR = Path("...")
TRAIN_JSONL = SUMMARY_TRAIN_JSONL
VALID_JSONL = SUMMARY_VALID_JSONL
OUTPUT_DIR = OUTPUT_ROOT / "summary_model"
```

The SFT script loads a base model checkpoint and fine-tunes the model using summary instruction-response pairs.

---

## Evaluate Summary Model

Run summary model evaluation from the project root:

```bash
PYTHONPATH=. python3 scripts/eval/eval_summary_model.py
```

Before running, check the checkpoint path, test data path, and output directory at the top of `scripts/eval/eval_summary_model.py`.

```python
CHECKPOINT_DIR = OUTPUT_ROOT / "summary_model" / "checkpoints" / "step_XXXXX"
TEST_JSONL = DATA_ROOT / "test_sample.jsonl"
OUTPUT_DIR = OUTPUT_ROOT / "summary_model_eval" / CHECKPOINT_STEP

MAX_NEW_TOKENS = 128
TEMPERATURE = 0.0
TOP_P = 0.9
MAX_ROUGE_SAMPLES = None
ROUGE_TOKENIZER = "char"
```

The evaluation script calculates loss, perplexity, and ROUGE scores.

---

## Outputs

Training and evaluation results are saved under `outputs/`.

Example output structure:

```text
outputs/
├── base_model/
│   ├── logs/
│   │   └── train_log.csv
│   └── checkpoints/
│       └── step_XXXXX/
│           ├── pytorch_model.pt
│           ├── optimizer.pt
│           ├── scheduler.pt
│           └── training_config.json
│
├── summary_model/
│   └── checkpoints/
│       └── step_XXXXX/
│
└── summary_model_eval/
    └── step_XXXXX/
        ├──summary_eval_results.json
        └── summary_predictions.jsonl
```

---
