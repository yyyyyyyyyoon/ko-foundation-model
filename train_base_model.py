import csv
import json
import math
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from base_model import KLLMConfig, KLLMForCausalLM


DATA_ROOT = Path(r"C:\Users\dbstj\dataset")

TRAIN_FILES = [
    DATA_ROOT / "processed" / "kowiki_train.jsonl",
    DATA_ROOT / "processed" / "ko_aihub_train.jsonl",
    DATA_ROOT / "processed" / "enwiki_train.jsonl",
    DATA_ROOT / "processed" / "code_train.jsonl",
]

TOKENIZER_DIR = DATA_ROOT / "tokenizer_bpe_64k"

OUTPUT_DIR = Path("outputs/base_model")
LOG_DIR = OUTPUT_DIR / "logs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / "train_log.csv"

MAX_LENGTH = 1024
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
NUM_EPOCHS = 1
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.03

LOG_STEPS = 10
EVAL_STEPS = 100
SAVE_STEPS = 500

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = torch.cuda.is_available()


class TextJsonlDataset(Dataset):
    def __init__(self, files: List[Path], tokenizer, max_length: int):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for file_path in files:
            if not file_path.exists():
                print(f"[WARN] File not found: {file_path}")
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    text = obj.get("text", "").strip()
                    if text:
                        self.samples.append(text)

        print(f"Loaded samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def load_tokenizer(tokenizer_dir: Path):
    return AutoTokenizer.from_pretrained(str(tokenizer_dir))


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

        total_loss += loss.item()
        total_steps += 1

    eval_loss = total_loss / max(total_steps, 1)

    if eval_loss < 20:
        perplexity = math.exp(eval_loss)
    else:
        perplexity = float("inf")

    model.train()

    return eval_loss, perplexity


def save_checkpoint(model, optimizer, scheduler, step: int):
    save_dir = CHECKPOINT_DIR / f"step_{step}"
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), save_dir / "pytorch_model.pt")
    torch.save(optimizer.state_dict(), save_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), save_dir / "scheduler.pt")

    print(f"[SAVE] {save_dir}")


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

    model = KLLMForCausalLM(config)

    return model


def build_small_test_model(vocab_size: int):
    config = KLLMConfig(
        vocab_size=vocab_size,
        hidden_size=512,
        intermediate_size=1408,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )

    model = KLLMForCausalLM(config)

    return model


def main(use_small_test_model: bool = False):
    torch.manual_seed(SEED)

    print(f"Device: {DEVICE}")

    tokenizer = load_tokenizer(TOKENIZER_DIR)
    vocab_size = len(tokenizer)

    print(f"Tokenizer vocab size: {vocab_size}")

    dataset = TextJsonlDataset(TRAIN_FILES, tokenizer, MAX_LENGTH)

    if len(dataset) == 0:
        raise ValueError("No training samples found.")

    eval_size = max(1, int(len(dataset) * 0.05))
    train_size = len(dataset) - eval_size

    train_dataset, eval_dataset = random_split(
        dataset,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    if use_small_test_model:
        model = build_small_test_model(vocab_size)
    else:
        model = build_model(vocab_size)

    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e9:.3f}B")

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    total_update_steps = max(1, (len(train_loader) * NUM_EPOCHS) // GRAD_ACCUM_STEPS)
    warmup_steps = int(total_update_steps * WARMUP_RATIO)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    global_step = 0
    optimizer_step = 0
    running_loss = 0.0
    running_count = 0

    with open(LOG_PATH, "w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(
            [
                "global_step",
                "optimizer_step",
                "epoch",
                "train_loss",
                "eval_loss",
                "perplexity",
                "learning_rate",
            ]
        )

        model.train()

        for epoch in range(NUM_EPOCHS):
            for batch in train_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs["loss"]
                    loss_for_backward = loss / GRAD_ACCUM_STEPS

                scaler.scale(loss_for_backward).backward()

                running_loss += loss.item()
                running_count += 1
                global_step += 1

                if global_step % GRAD_ACCUM_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    scaler.step(optimizer)
                    scaler.update()

                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                    optimizer_step += 1
                    lr = scheduler.get_last_lr()[0]

                    if optimizer_step % LOG_STEPS == 0:
                        train_loss = running_loss / max(running_count, 1)
                        running_loss = 0.0
                        running_count = 0

                        writer.writerow(
                            [
                                global_step,
                                optimizer_step,
                                epoch,
                                train_loss,
                                "",
                                "",
                                lr,
                            ]
                        )
                        log_file.flush()

                        print(
                            f"[TRAIN] "
                            f"global_step={global_step} "
                            f"optimizer_step={optimizer_step} "
                            f"epoch={epoch} "
                            f"loss={train_loss:.4f} "
                            f"lr={lr:.6e}"
                        )

                    if optimizer_step % EVAL_STEPS == 0:
                        eval_loss, perplexity = evaluate(model, eval_loader)

                        writer.writerow(
                            [
                                global_step,
                                optimizer_step,
                                epoch,
                                "",
                                eval_loss,
                                perplexity,
                                lr,
                            ]
                        )
                        log_file.flush()

                        print(
                            f"[EVAL] "
                            f"global_step={global_step} "
                            f"optimizer_step={optimizer_step} "
                            f"eval_loss={eval_loss:.4f} "
                            f"ppl={perplexity:.2f}"
                        )

                    if optimizer_step % SAVE_STEPS == 0:
                        save_checkpoint(model, optimizer, scheduler, optimizer_step)

        save_checkpoint(model, optimizer, scheduler, optimizer_step)

    print("Training finished.")
    print(f"Log saved to: {LOG_PATH}")


if __name__ == "__main__":
    main(use_small_test_model=True) # True: 소형 config test, False: base model train
