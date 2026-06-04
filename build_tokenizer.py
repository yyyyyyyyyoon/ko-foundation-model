import argparse
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

DATA_ROOT = Path(r"C:\Users\dbstj\dataset")
TRAIN_FILE = DATA_ROOT / "tokenizer_train" / "tokenizer_train_data.txt"
DEFAULT_REPORT_DIR = Path("outputs")
SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<system>",
    "<user>",
    "<assistant>",
    "<NAME>",
    "<EMAIL>",
    "<PHONE>",
    "<ADDRESS>",
    "<URL>",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=64000,
        help="Vocabulary size for tokenizer training.",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=TRAIN_FILE,
        help="Training text file path.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory to save the trained tokenizer. Defaults to dataset/tokenizer_bpe_<vocab-size>.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="UTF-8 debug report path. Defaults to outputs/tokenizer_debug_<vocab-size>.txt.",
    )
    return parser.parse_args()


def safe_token_preview(tokens):
    return [token.encode("unicode_escape").decode("ascii") for token in tokens]


def main():
    args = parse_args()
    save_dir = args.save_dir or (DATA_ROOT / f"tokenizer_bpe_{args.vocab_size // 1000}k")
    report_path = args.report_path or (DEFAULT_REPORT_DIR / f"tokenizer_debug_{args.vocab_size // 1000}k.txt")

    save_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(args.train_file)],
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
    )

    tokenizer.save_model(str(save_dir))
    tokenizer_json_path = save_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json_path))

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json_path),
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        additional_special_tokens=SPECIAL_TOKENS[4:],
    )
    fast_tokenizer.save_pretrained(str(save_dir))

    print("Tokenizer saved to:", save_dir)
    print("Vocab size:", fast_tokenizer.vocab_size)

    samples = [
        "한국어 언어모델을 구축하기 위한 사전학습 데이터를 정제한다.",
        "The model is trained on Korean, English, and code corpora.",
        "def preprocess_text(text):\n    return text.strip()",
    ]

    report_lines = [
        f"Tokenizer saved to: {save_dir}",
        f"Vocab size: {fast_tokenizer.vocab_size}",
    ]

    for sample in samples:
        ids = fast_tokenizer.encode(sample)
        tokens = fast_tokenizer.convert_ids_to_tokens(ids)
        decoded = fast_tokenizer.decode(ids)
        token_preview = safe_token_preview(tokens[:50])

        print("\nInput:", sample)
        print("Token count:", len(ids))
        print("Tokens:", token_preview)
        print("Decoded:", decoded)
        print("Roundtrip equal:", sample == decoded)

        report_lines.extend(
            [
                "",
                f"Input: {sample}",
                f"Token count: {len(ids)}",
                f"Token ids: {ids[:50]}",
                f"Tokens: {tokens[:50]}",
                f"Tokens escaped: {token_preview}",
                f"Decoded: {decoded}",
                f"Roundtrip equal: {sample == decoded}",
            ]
        )

    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("UTF-8 report saved to:", report_path)


if __name__ == "__main__":
    main()
