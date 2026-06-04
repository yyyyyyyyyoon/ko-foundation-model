import json
from pathlib import Path


DATA_ROOT = Path(r"C:\Users\dbstj\dataset")
PROCESSED_DIR = DATA_ROOT / "processed"

KOREAN_JSONL_FILES = [
    PROCESSED_DIR / "kowiki_train.jsonl",
    PROCESSED_DIR / "ko_aihub_train.jsonl",
]

ENGLISH_JSONL_FILES = [
    PROCESSED_DIR / "enwiki_train.jsonl",
]

CODE_TRAIN_FILE = PROCESSED_DIR / "code_tokenizer_train.txt"
CODE_FALLBACK_FILE = PROCESSED_DIR / "code_train.txt"

OUTPUT_DIR = DATA_ROOT / "tokenizer_train"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "tokenizer_train_data.txt"

TOTAL_TARGET_GB = 5
GB = 1024 ** 3

TARGET_BYTES = {
    "ko": int(TOTAL_TARGET_GB * GB * 0.6),
    "en": int(TOTAL_TARGET_GB * GB * 0.3),
    "code": int(TOTAL_TARGET_GB * GB * 0.1),
}


def byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


def iter_jsonl_text(file_path: Path):
    if not file_path.exists():
        print(f"[WARN] File not found: {file_path}")
        return

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = str(obj.get("text", "")).strip() if isinstance(obj, dict) else ""
            if text:
                yield text


def write_jsonl_texts(out, files, byte_budget: int, label: str):
    written = 0
    docs = 0

    for file_path in files:
        print(f"[{label}] reading: {file_path}")

        for text in iter_jsonl_text(file_path):
            block = text + "\n\n"
            size = byte_len(block)

            if written + size > byte_budget:
                print(f"[{label}] budget reached")
                print(f"[{label}] docs: {docs}, bytes: {written / GB:.3f} GB")
                return written, docs

            out.write(block)
            written += size
            docs += 1

    print(f"[{label}] finished")
    print(f"[{label}] docs: {docs}, bytes: {written / GB:.3f} GB")
    return written, docs


def write_text_files(out, files, byte_budget: int, label: str):
    written = 0

    for file_path in files:
        if not file_path.exists():
            print(f"[WARN] File not found: {file_path}")
            continue

        print(f"[{label}] reading: {file_path}")

        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break

                size = byte_len(chunk)
                if written + size > byte_budget:
                    remain = byte_budget - written
                    partial = chunk.encode("utf-8")[:remain].decode("utf-8", errors="ignore")
                    out.write(partial)
                    written += byte_len(partial)

                    print(f"[{label}] budget reached")
                    print(f"[{label}] bytes: {written / GB:.3f} GB")
                    return written

                out.write(chunk)
                written += size

        out.write("\n\n")
        written += 2

    print(f"[{label}] finished")
    print(f"[{label}] bytes: {written / GB:.3f} GB")
    return written


def resolve_code_files():
    if CODE_TRAIN_FILE.exists():
        print(f"[code] using split train file: {CODE_TRAIN_FILE}")
        return [CODE_TRAIN_FILE]

    print(f"[code] split train file not found, falling back to: {CODE_FALLBACK_FILE}")
    return [CODE_FALLBACK_FILE]


def main():
    code_files = resolve_code_files()

    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        ko_bytes, ko_docs = write_jsonl_texts(
            out,
            KOREAN_JSONL_FILES,
            TARGET_BYTES["ko"],
            "ko",
        )

        en_bytes, en_docs = write_jsonl_texts(
            out,
            ENGLISH_JSONL_FILES,
            TARGET_BYTES["en"],
            "en",
        )

        code_bytes = write_text_files(
            out,
            code_files,
            TARGET_BYTES["code"],
            "code",
        )

    print("\n[DONE]")
    print(f"Output: {OUTPUT_FILE}")
    print(f"ko: {ko_bytes / GB:.3f} GB, docs: {ko_docs}")
    print(f"en: {en_bytes / GB:.3f} GB, docs: {en_docs}")
    print(f"code: {code_bytes / GB:.3f} GB")


if __name__ == "__main__":
    main()
