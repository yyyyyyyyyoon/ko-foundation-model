import argparse
import json
import random
from pathlib import Path


DATA_ROOT = Path(r"C:\Users\dbstj\dataset")
PROCESSED_DIR = DATA_ROOT / "processed"
OUTPUT_MANIFEST = Path("outputs") / "tokenizer_test_split_manifest.json"
GB = 1024 ** 3
TOTAL_TARGET_GB = 5

TEXT_SOURCES = {
    "kowiki": PROCESSED_DIR / "kowiki_train.jsonl",
    "ko_aihub": PROCESSED_DIR / "ko_aihub_train.jsonl",
    "enwiki": PROCESSED_DIR / "enwiki_train.jsonl",
}

CODE_SOURCE = PROCESSED_DIR / "code_train.txt"

TRAIN_BUDGETS = {
    "ko": int(TOTAL_TARGET_GB * GB * 0.6),
    "en": int(TOTAL_TARGET_GB * GB * 0.3),
    "code": int(TOTAL_TARGET_GB * GB * 0.1),
}

KO_SOURCE_ORDER = ["kowiki", "ko_aihub"]
EN_SOURCE_ORDER = ["enwiki"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create tokenizer test splits using only data that was not consumed by the "
            "existing tokenizer training build."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic reservoir sampling.",
    )
    parser.add_argument(
        "--text-test-docs",
        type=int,
        default=5000,
        help="Number of holdout documents to sample per text source.",
    )
    parser.add_argument(
        "--code-test-blocks",
        type=int,
        default=5000,
        help="Number of holdout code blocks to sample.",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=50,
        help="Minimum character length for text documents.",
    )
    parser.add_argument(
        "--min-code-chars",
        type=int,
        default=30,
        help="Minimum character length for code blocks.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=OUTPUT_MANIFEST,
        help="Path to write split metadata.",
    )
    return parser.parse_args()


def byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


def stream_jsonl_records(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_number, line in enumerate(f, 1):
            raw_line = line.rstrip("\n")
            if not raw_line.strip():
                continue

            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            text = str(obj.get("text", "")).strip() if isinstance(obj, dict) else ""
            if not text:
                continue

            block = text + "\n\n"
            yield {
                "line_number": line_number,
                "raw_line": raw_line,
                "text": text,
                "train_block_bytes": byte_len(block),
            }


def stream_code_blocks(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    for block_index, block in enumerate(content.split("\n\n"), 1):
        stripped = block.strip()
        if stripped:
            yield {
                "block_index": block_index,
                "text": stripped,
                "block_bytes": byte_len(stripped + "\n\n"),
            }


def consume_text_budget(source_names, budget_bytes: int):
    usage = {}
    remaining = budget_bytes

    for source_name in source_names:
        path = TEXT_SOURCES[source_name]
        consumed_docs = 0
        consumed_bytes = 0

        for record in stream_jsonl_records(path):
            size = record["train_block_bytes"]
            if consumed_bytes + size > remaining:
                break

            consumed_docs += 1
            consumed_bytes += size

        usage[source_name] = {
            "path": str(path),
            "consumed_docs": consumed_docs,
            "consumed_bytes": consumed_bytes,
        }
        remaining -= consumed_bytes
        if remaining <= 0:
            break

    for source_name in source_names:
        usage.setdefault(
            source_name,
            {
                "path": str(TEXT_SOURCES[source_name]),
                "consumed_docs": 0,
                "consumed_bytes": 0,
            },
        )

    return usage


def consume_code_budget(path: Path, budget_bytes: int):
    if not path.exists():
        return {
            "path": str(path),
            "source_bytes": 0,
            "consumed_bytes": 0,
            "fully_consumed": False,
        }

    source_bytes = path.stat().st_size
    consumed_bytes = min(source_bytes + 2, budget_bytes)
    return {
        "path": str(path),
        "source_bytes": source_bytes,
        "consumed_bytes": consumed_bytes,
        "fully_consumed": consumed_bytes >= source_bytes,
    }


def reservoir_sample(items, sample_size: int, rng: random.Random):
    sample = []
    seen = 0

    for item in items:
        seen += 1
        if len(sample) < sample_size:
            sample.append(item)
            continue

        replace_index = rng.randint(0, seen - 1)
        if replace_index < sample_size:
            sample[replace_index] = item

    return sample, seen


def sample_unused_text_records(path: Path, skip_docs: int, sample_size: int, min_chars: int, seed: int):
    rng = random.Random(seed)

    def eligible_records():
        seen_docs = 0
        for record in stream_jsonl_records(path):
            if seen_docs < skip_docs:
                seen_docs += 1
                continue
            if len(record["text"]) < min_chars:
                continue
            yield record

    sample, eligible_count = reservoir_sample(eligible_records(), sample_size, rng)
    return sample, eligible_count


def sample_unused_code_blocks(path: Path, consumed_bytes: int, sample_size: int, min_chars: int, seed: int):
    rng = random.Random(seed)
    skipped_bytes = 0

    def eligible_blocks():
        nonlocal skipped_bytes
        for block in stream_code_blocks(path):
            if skipped_bytes < consumed_bytes:
                skipped_bytes += block["block_bytes"]
                continue
            if len(block["text"]) < min_chars:
                continue
            yield block

    sample, eligible_count = reservoir_sample(eligible_blocks(), sample_size, rng)
    return sample, eligible_count


def write_jsonl_output(path: Path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record["raw_line"] + "\n")


def write_code_output(path: Path, blocks):
    with path.open("w", encoding="utf-8") as f:
        for index, block in enumerate(blocks):
            if index:
                f.write("\n\n")
            f.write(block["text"])


def sort_text_sample(records):
    return sorted(records, key=lambda x: x["line_number"])


def sort_code_sample(blocks):
    return sorted(blocks, key=lambda x: x["block_index"])


def main():
    args = parse_args()

    ko_usage = consume_text_budget(KO_SOURCE_ORDER, TRAIN_BUDGETS["ko"])
    en_usage = consume_text_budget(EN_SOURCE_ORDER, TRAIN_BUDGETS["en"])
    code_usage = consume_code_budget(CODE_SOURCE, TRAIN_BUDGETS["code"])

    manifest = {
        "seed": args.seed,
        "rules": {
            "text_test_docs_per_source": args.text_test_docs,
            "code_test_blocks": args.code_test_blocks,
            "min_text_chars": args.min_text_chars,
            "min_code_chars": args.min_code_chars,
            "train_reconstruction": "Replays the original tokenizer-train prefix consumption and samples only from the unused suffix.",
        },
        "train_usage": {
            "ko": ko_usage,
            "en": en_usage,
            "code": code_usage,
        },
        "outputs": {},
    }

    for source_name, info in {**ko_usage, **en_usage}.items():
        path = Path(info["path"])
        if not path.exists():
            print(f"[WARN] Source not found: {path}")
            continue

        sample, eligible_count = sample_unused_text_records(
            path=path,
            skip_docs=info["consumed_docs"],
            sample_size=args.text_test_docs,
            min_chars=args.min_text_chars,
            seed=args.seed,
        )
        sample = sort_text_sample(sample)

        output_path = path.with_name(path.name.replace("_train", "_test"))
        if sample:
            write_jsonl_output(output_path, sample)
            print(f"[DONE] {source_name}: wrote {len(sample)} docs to {output_path}")
        else:
            print(f"[WARN] {source_name}: no eligible holdout documents left")

        manifest["outputs"][source_name] = {
            "output_path": str(output_path),
            "sampled_docs": len(sample),
            "eligible_unused_docs": eligible_count,
            "consumed_docs": info["consumed_docs"],
            "consumed_bytes": info["consumed_bytes"],
        }

    code_output_path = CODE_SOURCE.with_name(CODE_SOURCE.name.replace("_train", "_test"))
    if CODE_SOURCE.exists():
        blocks, eligible_count = sample_unused_code_blocks(
            path=CODE_SOURCE,
            consumed_bytes=code_usage["consumed_bytes"],
            sample_size=args.code_test_blocks,
            min_chars=args.min_code_chars,
            seed=args.seed,
        )
        blocks = sort_code_sample(blocks)

        if blocks:
            write_code_output(code_output_path, blocks)
            print(f"[DONE] code: wrote {len(blocks)} blocks to {code_output_path}")
        else:
            print("[WARN] code: no eligible holdout blocks left")

        manifest["outputs"]["code"] = {
            "output_path": str(code_output_path),
            "sampled_blocks": len(blocks),
            "eligible_unused_blocks": eligible_count,
            "consumed_bytes": code_usage["consumed_bytes"],
            "fully_consumed_by_train": code_usage["fully_consumed"],
        }

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[DONE] Manifest saved to: {args.output_manifest}")


if __name__ == "__main__":
    main()
