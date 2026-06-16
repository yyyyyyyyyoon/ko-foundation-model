import math
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from base_model import KLLMConfig, KLLMForCausalLM


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PackedEvalDataset(Dataset):
    def __init__(self, eval_path: Path):
        if not eval_path.exists():
            raise FileNotFoundError(f"eval.pt not found: {eval_path}")

        self.data = torch.load(eval_path, map_location="cpu")

        if len(self.data.shape) != 2:
            raise ValueError(f"Expected eval data shape [num_blocks, block_size], got {self.data.shape}")

        print(f"[EVAL DATA] path: {eval_path}")
        print(f"[EVAL DATA] blocks: {self.data.shape[0]}")
        print(f"[EVAL DATA] block size: {self.data.shape[1]}")
        print(f"[EVAL DATA] tokens: {self.data.numel()}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        input_ids = self.data[idx].long()
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def collate_fn(batch):
    input_ids = torch.stack([x["input_ids"] for x in batch], dim=0)
    labels = torch.stack([x["labels"] for x in batch], dim=0)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def build_model(vocab_size: int):
    config = KLLMConfig(
        vocab_size=vocab_size,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    return KLLMForCausalLM(config)


@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()

    total_loss = 0.0
    total_steps = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]

        if loss is None or torch.isnan(loss) or torch.isinf(loss):
            continue

        total_loss += loss.item()
        total_steps += 1

    eval_loss = total_loss / total_steps
    perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")

    return eval_loss, perplexity


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Final base checkpoint directory, e.g. outputs/base_model_v2/checkpoints/step_163766",
    )
    parser.add_argument(
        "--eval-pt",
        type=str,
        default="/home/aiselab/workspace/ko-llm/dataset/packed_corpus_4k/eval.pt",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default="/home/aiselab/workspace/ko-llm/dataset/tokenizer_bpe_64k",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint)
    ckpt_path = checkpoint_dir / "pytorch_model.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {ckpt_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    vocab_size = len(tokenizer)

    model = build_model(vocab_size)

    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)

    model = model.to(DEVICE)

    eval_dataset = PackedEvalDataset(Path(args.eval_pt))
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    eval_loss, perplexity = evaluate(model, eval_loader)

    print("=" * 50)
    print(f"Final checkpoint: {checkpoint_dir.name}")
    print(f"Eval loss: {eval_loss:.6f}")
    print(f"Perplexity: {perplexity:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()