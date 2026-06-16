import csv
import json
import math
import random
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from base_model import KLLMConfig, KLLMForCausalLM


# Defaults
DATA_ROOT = Path("/home/aiselab/workspace/ko-llm/dataset")
TOKENIZER_DIR = DATA_ROOT / "tokenizer_bpe_64k"

DEFAULT_SFT_DIR = DATA_ROOT / "sft" / "summary"
DEFAULT_TRAIN_JSONL = DEFAULT_SFT_DIR / "train.jsonl"
DEFAULT_VALID_JSONL = DEFAULT_SFT_DIR / "valid.jsonl"
DEFAULT_TEST_JSONL = DEFAULT_SFT_DIR / "test.jsonl"
DEFAULT_OUTPUT_DIR = Path("/home/aiselab/workspace/ko-llm/outputs/summary_model")

MAX_LENGTH = 4096
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
NUM_EPOCHS = 1

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03

LOG_STEPS = 10
EVAL_STEPS = 500
SAVE_STEPS = 500
EVAL_RATIO = 0.02
KEEP_LAST_N_CHECKPOINTS = 2

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = torch.cuda.is_available()

# Tokenizer / Model
def load_tokenizer(tokenizer_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must have eos_token_id.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

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


def load_base_checkpoint(model, checkpoint_dir: Path):
    ckpt_path = checkpoint_dir / "pytorch_model.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)

    print(f"[BASE] Loaded base checkpoint from {ckpt_path}")

# SFT Dataset
def truncate_prompt_keep_suffix(
    prompt_ids: List[int],
    max_prompt_len: int,
    suffix_len: int = 32,
) -> List[int]:
    if len(prompt_ids) <= max_prompt_len:
        return prompt_ids

    if max_prompt_len <= suffix_len:
        return prompt_ids[-max_prompt_len:]

    head_len = max_prompt_len - suffix_len
    return prompt_ids[:head_len] + prompt_ids[-suffix_len:]


class SummarySFTDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, max_length: int):
        if not jsonl_path.exists():
            raise FileNotFoundError(f"SFT jsonl not found: {jsonl_path}")

        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        bad = 0

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    ex = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue

                prompt = ex.get("prompt")
                response = ex.get("response")

                if not prompt or not response:
                    bad += 1
                    continue

                self.samples.append(
                    {
                        "prompt": prompt,
                        "response": response,
                    }
                )

        if not self.samples:
            raise ValueError(f"No valid SFT samples found in {jsonl_path}")

        print(f"[SFT DATA] path: {jsonl_path}")
        print(f"[SFT DATA] valid samples: {len(self.samples)}")
        print(f"[SFT DATA] skipped bad samples: {bad}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]
        prompt = ex["prompt"]
        response = ex["response"]

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]

        response_ids = self.tokenizer(
            response,
            add_special_tokens=False,
        )["input_ids"]

        # response 끝에 EOS 추가
        response_ids = response_ids + [self.tokenizer.eos_token_id]

        # response가 너무 길면 response부터 잘라냄
        if len(response_ids) >= self.max_length:
            response_ids = response_ids[: self.max_length - 1] + [self.tokenizer.eos_token_id]
            prompt_ids = []

        max_prompt_len = self.max_length - len(response_ids)
        prompt_ids = truncate_prompt_keep_suffix(prompt_ids, max_prompt_len)

        input_ids = prompt_ids + response_ids

        labels = [-100] * len(prompt_ids) + response_ids.copy()

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch, pad_token_id: int):
    max_len = max(x["input_ids"].size(0) for x in batch)

    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for x in batch:
        input_ids = x["input_ids"]
        labels = x["labels"]

        pad_len = max_len - input_ids.size(0)

        input_ids = torch.cat(
            [
                input_ids,
                torch.full((pad_len,), pad_token_id, dtype=torch.long),
            ],
            dim=0,
        )

        labels = torch.cat(
            [
                labels,
                torch.full((pad_len,), -100, dtype=torch.long),
            ],
            dim=0,
        )

        attention_mask = (input_ids != pad_token_id).long()

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)

    return {
        "input_ids": torch.stack(input_ids_list, dim=0),
        "labels": torch.stack(labels_list, dim=0),
        "attention_mask": torch.stack(attention_mask_list, dim=0),
    }

# Eval / Save
@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()

    total_loss = 0.0
    total_steps = 0
    skipped_steps = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        valid_label_count = (labels[:, 1:] != -100).sum().item()
        if valid_label_count == 0:
            skipped_steps += 1
            continue

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]

        if loss is None or torch.isnan(loss) or torch.isinf(loss):
            skipped_steps += 1
            continue

        total_loss += loss.item()
        total_steps += 1

    if total_steps == 0:
        eval_loss = float("nan")
        perplexity = float("inf")
    else:
        eval_loss = total_loss / total_steps
        perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")

    model.train()

    print(
        f"[EVAL] eval_loss={eval_loss:.4f}, "
        f"ppl={perplexity:.2f}, "
        f"valid_steps={total_steps}, "
        f"skipped_steps={skipped_steps}"
    )

    return eval_loss, perplexity


def cleanup_old_checkpoints(checkpoint_dir: Path, keep_last_n: int):
    if keep_last_n is None or keep_last_n <= 0:
        return

    ckpts = []

    for path in checkpoint_dir.glob("step_*"):
        if not path.is_dir():
            continue

        try:
            step = int(path.name.split("_")[-1])
        except ValueError:
            continue

        ckpts.append((step, path))

    ckpts.sort(key=lambda x: x[0])

    if len(ckpts) <= keep_last_n:
        return

    for step, path in ckpts[:-keep_last_n]:
        print(f"[CLEANUP] Removing old checkpoint: {path}")
        import shutil
        shutil.rmtree(path)


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    step: int,
    output_dir: Path,
    config_to_save: Dict[str, Any],
):
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_dir = checkpoint_dir / f"step_{step}"
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), save_dir / "pytorch_model.pt")
    torch.save(optimizer.state_dict(), save_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), save_dir / "scheduler.pt")

    with open(save_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] {save_dir}")

    cleanup_old_checkpoints(checkpoint_dir, KEEP_LAST_N_CHECKPOINTS)

# Train
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-jsonl",
        type=str,
        default=str(DEFAULT_TRAIN_JSONL),
    )
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        required=True,
        help="Base model checkpoint dir, e.g. outputs/base_model_v2/checkpoints/step_163766",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_LENGTH,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=EVAL_RATIO,
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--valid-jsonl",
        type=str,
        default=str(DEFAULT_VALID_JSONL),
    )

    parser.add_argument(
        "--test-jsonl",
        type=str,
        default=str(DEFAULT_TEST_JSONL),
    )

    args = parser.parse_args()

    torch.manual_seed(SEED)
    random.seed(SEED)

    train_jsonl = Path(args.train_jsonl)
    valid_jsonl = Path(args.valid_jsonl)
    test_jsonl = Path(args.test_jsonl)
    base_checkpoint = Path(args.base_checkpoint)
    output_dir = Path(args.output_dir)
    log_dir = output_dir / "logs"
    checkpoint_dir = output_dir / "checkpoints"

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / "train_log.csv"

    print(f"Device: {DEVICE}")
    print(f"Use AMP: {USE_AMP}")
    print(f"Train jsonl: {train_jsonl}")
    print(f"Valid jsonl: {valid_jsonl}")
    print(f"Test jsonl: {test_jsonl}")
    print(f"Base checkpoint: {base_checkpoint}")
    print(f"Output dir: {output_dir}")

    tokenizer = load_tokenizer(TOKENIZER_DIR)
    vocab_size = len(tokenizer)

    print(f"Tokenizer vocab size: {vocab_size}")

    model = build_model(vocab_size)
    load_base_checkpoint(model, base_checkpoint)
    model = model.to(DEVICE)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e9:.3f}B")

    train_dataset = SummarySFTDataset(
        jsonl_path=train_jsonl,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    eval_dataset = SummarySFTDataset(
        jsonl_path=valid_jsonl,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    print(f"[SPLIT] train samples: {len(train_dataset)}")
    print(f"[SPLIT] valid samples: {len(eval_dataset)}")
    print(f"[SPLIT] test jsonl is reserved for final evaluation: {test_jsonl}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    total_update_steps = max(
        1,
        (len(train_loader) * args.epochs) // GRAD_ACCUM_STEPS,
    )

    if args.max_train_steps is not None:
        total_update_steps = min(total_update_steps, args.max_train_steps)

    warmup_steps = int(total_update_steps * WARMUP_RATIO)

    print(f"Total update steps: {total_update_steps}")
    print(f"Warmup steps: {warmup_steps}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    config_to_save = {
        "task": "summary_sft",
        "train_jsonl": str(train_jsonl),
        "valid_jsonl": str(valid_jsonl),
        "test_jsonl": str(test_jsonl),
        "tokenizer_dir": str(TOKENIZER_DIR),
        "base_checkpoint": str(base_checkpoint),
        "output_dir": str(output_dir),
        "max_length": args.max_length,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "total_update_steps": total_update_steps,
        "split": "predefined_train_valid_test",
        "log_steps": LOG_STEPS,
        "eval_steps": EVAL_STEPS,
        "save_steps": SAVE_STEPS,
        "loss_masking": "prompt=-100, response=labels",
    }

    with open(output_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, ensure_ascii=False, indent=2)

    log_exists = log_path.exists()
    log_f = open(log_path, "a", encoding="utf-8", newline="")
    writer = csv.writer(log_f)

    if not log_exists:
        writer.writerow(
            [
                "global_step",
                "optimizer_step",
                "epoch",
                "train_loss",
                "lr",
                "eval_loss",
                "eval_ppl",
            ]
        )
        log_f.flush()

    global_step = 0
    optimizer_step = 0
    running_loss = 0.0
    running_count = 0

    optimizer.zero_grad(set_to_none=True)

    try:
        for epoch in range(args.epochs):
            print(f"[EPOCH] {epoch + 1}/{args.epochs}")

            for batch in train_loader:
                global_step += 1

                input_ids = batch["input_ids"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs["loss"]

                    if loss is None or torch.isnan(loss) or torch.isinf(loss):
                        print(f"[WARN] invalid loss at global_step={global_step}, skip")
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    loss_for_backward = loss / GRAD_ACCUM_STEPS

                scaler.scale(loss_for_backward).backward()

                running_loss += loss.item()
                running_count += 1

                if global_step % GRAD_ACCUM_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    scaler.step(optimizer)
                    scaler.update()

                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    optimizer_step += 1
                    lr = scheduler.get_last_lr()[0]

                    if optimizer_step % LOG_STEPS == 0:
                        avg_loss = running_loss / max(1, running_count)
                        print(
                            f"[TRAIN] epoch={epoch + 1} "
                            f"global_step={global_step} "
                            f"optimizer_step={optimizer_step} "
                            f"loss={avg_loss:.4f} "
                            f"lr={lr:.6e}"
                        )

                        writer.writerow(
                            [
                                global_step,
                                optimizer_step,
                                epoch + 1,
                                avg_loss,
                                lr,
                                "",
                                "",
                            ]
                        )
                        log_f.flush()

                        running_loss = 0.0
                        running_count = 0

                    if optimizer_step % EVAL_STEPS == 0:
                        eval_loss, eval_ppl = evaluate(model, eval_loader)

                        writer.writerow(
                            [
                                global_step,
                                optimizer_step,
                                epoch + 1,
                                "",
                                lr,
                                eval_loss,
                                eval_ppl,
                            ]
                        )
                        log_f.flush()

                    if optimizer_step % SAVE_STEPS == 0:
                        save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            step=optimizer_step,
                            output_dir=output_dir,
                            config_to_save=config_to_save,
                        )

                    if optimizer_step >= total_update_steps:
                        print("[TRAIN] Reached total_update_steps.")
                        break

            if optimizer_step >= total_update_steps:
                break

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=optimizer_step,
            output_dir=output_dir,
            config_to_save=config_to_save,
        )

        print("[DONE] SFT training finished.")

    finally:
        log_f.close()


if __name__ == "__main__":
    main()