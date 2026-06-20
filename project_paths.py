import os
from pathlib import Path

# project_paths.py가 있는 위치를 프로젝트 루트로 봄
PROJECT_ROOT = Path(__file__).resolve().parent

# 기본 경로
# 필요하면 환경변수로 덮어쓸 수 있음
DATA_ROOT = Path(
    os.environ.get("KOLLM_DATA_ROOT", PROJECT_ROOT / "dataset")
).resolve()

OUTPUT_ROOT = Path(
    os.environ.get("KOLLM_OUTPUT_ROOT", PROJECT_ROOT / "outputs")
).resolve()


# Tokenizer / Pretraining data
TOKENIZER_DIR = DATA_ROOT / "tokenizer_bpe_64k"
PACKED_DATA_DIR = DATA_ROOT / "packed_corpus_4k"


# SFT summary data
SFT_DIR = DATA_ROOT / "sft" / "summary"
SUMMARY_TRAIN_JSONL = SFT_DIR / "train.jsonl"
SUMMARY_VALID_JSONL = SFT_DIR / "valid.jsonl"
SUMMARY_TEST_JSONL = SFT_DIR / "test.jsonl"