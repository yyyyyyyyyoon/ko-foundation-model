import bisect
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from base_model import KLLMConfig, KLLMForCausalLM
from torch.utils.data import Dataset, DataLoader, Subset
import shutil

# Pahts
DATA_ROOT = Path("/home/aiselab/workspace/ko-llm/dataset")

TOKENIZER_DIR = DATA_ROOT / "tokenizer_bpe_64k"
PACKED_DATA_DIR = DATA_ROOT / "packed_corpus_4k"

OUTPUT_DIR = Path("/home/aiselab/workspace/ko-llm/outputs/base_model_v2")
LOG_DIR = OUTPUT_DIR / "logs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / "train_log.csv"

# Training config
MAX_LENGTH = 4096
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
NUM_EPOCHS = 1

MAX_TRAIN_STEPS: Optional[int] = None
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.03

LOG_STEPS = 10
EVAL_STEPS = 2000
SAVE_STEPS = 2000
KEEP_LAST_N_CHECKPOINTS = 1
MAIN_EVAL_DOCS = 256
SMALL_TEST_EVAL_DOCS = 64

RESUME_CHECKPOINT: Optional[str] = "/home/aiselab/workspace/ko-llm/outputs/base_model_v2/checkpoints/step_30000" # checkpoint 이어서 학습할 경우

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = torch.cuda.is_available()

# Dataset
class PackedBlockDataset(Dataset):
    # build_packed_data.py가 생성한 train_*.pt 파일을 학습용 Dataset으로 로드
    def __init__(self, packed_dir: Path, block_size: int):
        self.packed_dir = packed_dir
        self.block_size = block_size

        self.files = sorted(packed_dir.glob("train_*.pt"))

        if not self.files:
            raise ValueError(f"No packed train files found in {packed_dir}")

        self.file_infos = []
        self.cumulative_blocks = []
        self.total_blocks = 0

        for file_path in self.files:
            data = torch.load(file_path, map_location="cpu")

            if data.dim() != 2:
                raise ValueError(
                    f"Invalid packed file shape: {file_path}, shape={tuple(data.shape)}"
                )

            if data.size(1) != block_size:
                raise ValueError(
                    f"Block size mismatch in {file_path}: "
                    f"expected={block_size}, got={data.size(1)}"
                )

            num_blocks = data.size(0)
            self.file_infos.append(
                {
                    "path": file_path,
                    "start": self.total_blocks,
                    "num_blocks": num_blocks,
                }
            )

            self.total_blocks += num_blocks
            self.cumulative_blocks.append(self.total_blocks)

        self._cache_file = None
        self._cache_data = None

        print(f"[TRAIN DATA] packed dir: {packed_dir}")
        print(f"[TRAIN DATA] packed files: {len(self.files)}")
        print(f"[TRAIN DATA] total blocks: {self.total_blocks}")
        print(f"[TRAIN DATA] block size: {block_size}")

    def __len__(self):
        return self.total_blocks

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_blocks:
            raise IndexError(idx)

        file_idx = bisect.bisect_right(self.cumulative_blocks, idx)
        info = self.file_infos[file_idx]

        local_idx = idx - info["start"]
        file_path = info["path"]

        if self._cache_file != file_path:
            self._cache_data = torch.load(file_path, map_location="cpu").long()
            self._cache_file = file_path

        input_ids = self._cache_data[local_idx]
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class PackedEvalDataset(Dataset):
    # build_packed_data.py가 생성한 eval.pt 파일을 평가용 Dataset으로 로드
    def __init__(self, eval_path: Path, block_size: int):
        if not eval_path.exists():
            raise ValueError(f"Eval file not found: {eval_path}")

        self.data = torch.load(eval_path, map_location="cpu").long()

        if self.data.dim() != 2:
            raise ValueError(
                f"Invalid eval file shape: {eval_path}, shape={tuple(self.data.shape)}"
            )

        if self.data.size(1) != block_size:
            raise ValueError(
                f"Eval block size mismatch: expected={block_size}, got={self.data.size(1)}"
            )

        print(f"[EVAL DATA] eval path: {eval_path}")
        print(f"[EVAL DATA] blocks: {self.data.size(0)}")
        print(f"[EVAL DATA] block size: {self.data.size(1)}")

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        input_ids = self.data[idx]
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

# Tokenizer / Model
def load_tokenizer(tokenizer_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must have eos_token_id for packed pretraining.")

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
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )

    model = KLLMForCausalLM(config)

    return model

def load_resume_checkpoint_if_needed(model):
    if RESUME_CHECKPOINT is None:
        print("[RESUME] Training from scratch.")
        return

    ckpt_path = Path(RESUME_CHECKPOINT) / "pytorch_model.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)

    print(f"[RESUME] Loaded model checkpoint from {ckpt_path}")

def get_resume_steps():
    if RESUME_CHECKPOINT is None:
        return 0, 0

    ckpt_name = Path(RESUME_CHECKPOINT).name  # step_30000
    resume_optimizer_step = int(ckpt_name.split("_")[-1])
    resume_global_step = resume_optimizer_step * GRAD_ACCUM_STEPS

    return resume_global_step, resume_optimizer_step

# Eval / Save
@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()

    total_loss = 0.0
    total_steps = 0
    skipped_steps = 0
    nan_steps = 0

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
            nan_steps += 1
            continue

        total_loss += loss.item()
        total_steps += 1

    if total_steps == 0:
        eval_loss = float("nan")
        perplexity = float("inf")
    else:
        eval_loss = total_loss / total_steps
        if eval_loss < 20:
            perplexity = math.exp(eval_loss)
        else:
            perplexity = float("inf")

    model.train()

    print(
        f"[EVAL] valid_steps={total_steps}, "
        f"skipped_steps={skipped_steps}, "
        f"nan_steps={nan_steps}"
    )

    return eval_loss, perplexity

def save_checkpoint(model, optimizer, scheduler, step: int):
    save_dir = CHECKPOINT_DIR / f"step_{step}"
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), save_dir / "pytorch_model.pt")
    torch.save(optimizer.state_dict(), save_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), save_dir / "scheduler.pt")

    config_to_save: Dict[str, Any] = {
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "max_train_steps": MAX_TRAIN_STEPS,
        "log_steps": LOG_STEPS,
        "eval_steps": EVAL_STEPS,
        "save_steps": SAVE_STEPS,
        "data_ratio": "ko:en:code = 6:3:1",
        "packed_data_dir": str(PACKED_DATA_DIR),
        "tokenizer_dir": str(TOKENIZER_DIR),
        "resume_checkpoint": RESUME_CHECKPOINT,
    }

    with open(save_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] {save_dir}")
    cleanup_old_checkpoints(KEEP_LAST_N_CHECKPOINTS)

def cleanup_old_checkpoints(keep_last_n: int = 1):
    checkpoint_items = []

    for ckpt_dir in CHECKPOINT_DIR.glob("step_*"):
        if not ckpt_dir.is_dir():
            continue

        try:
            step = int(ckpt_dir.name.split("_")[-1])
        except ValueError:
            continue

        checkpoint_items.append((step, ckpt_dir))

    checkpoint_items.sort(key=lambda x: x[0])

    if len(checkpoint_items) <= keep_last_n:
        return

    to_delete = checkpoint_items[:-keep_last_n]

    for step, ckpt_dir in to_delete:
        print(f"[CLEANUP] Removing old checkpoint: {ckpt_dir}")
        shutil.rmtree(ckpt_dir)

# Train
def main(use_small_test_model: bool = False):
    torch.manual_seed(SEED)

    print(f"Device: {DEVICE}")
    print(f"Use AMP: {USE_AMP}")
    print(f"Packed data dir: {PACKED_DATA_DIR}")

    tokenizer = load_tokenizer(TOKENIZER_DIR)
    vocab_size = len(tokenizer)

    print(f"Tokenizer vocab size: {vocab_size}")

    # 1. 모델 생성
    if use_small_test_model:
        model = build_small_test_model(vocab_size)
    else:
        model = build_model(vocab_size)

    # 2. resume checkpoint가 있으면 모델 가중치 로드
    load_resume_checkpoint_if_needed(model)

    # 3. GPU로 이동
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e9:.3f}B")

    # 4. 미리 생성한 packed dataset 로드
    train_dataset = PackedBlockDataset(
        packed_dir=PACKED_DATA_DIR,
        block_size=MAX_LENGTH,
    )

    eval_dataset = PackedEvalDataset(
        eval_path=PACKED_DATA_DIR / "eval.pt",
        block_size=MAX_LENGTH,
    )

    # 5. optimizer / scheduler 설정
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    if MAX_TRAIN_STEPS is not None:
        total_update_steps = int(MAX_TRAIN_STEPS)
    else:
        total_update_steps = max(
            1,
            (len(train_dataset) * NUM_EPOCHS) // GRAD_ACCUM_STEPS,
        )

    warmup_steps = int(total_update_steps * WARMUP_RATIO)

    print(f"Total update steps: {total_update_steps}")
    print(f"Warmup steps: {warmup_steps}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    resume_global_step, resume_optimizer_step = get_resume_steps()

    if resume_global_step > 0:
        print(f"[RESUME] Skipping first {resume_global_step} blocks.")
        train_dataset_for_loader = Subset(
            train_dataset,
            range(resume_global_step, len(train_dataset))
        )
    else:
        train_dataset_for_loader = train_dataset

    train_loader = DataLoader(
        train_dataset_for_loader,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    global_step = resume_global_step
    optimizer_step = resume_optimizer_step
    running_loss = 0.0
    running_count = 0

    if RESUME_CHECKPOINT is not None:
        ckpt_dir = Path(RESUME_CHECKPOINT)

        optimizer_path = ckpt_dir / "optimizer.pt"
        scheduler_path = ckpt_dir / "scheduler.pt"

        if optimizer_path.exists():
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=DEVICE))
            print(f"[RESUME] Loaded optimizer state from {optimizer_path}")

        if scheduler_path.exists():
            scheduler.load_state_dict(torch.load(scheduler_path, map_location=DEVICE))
            print(f"[RESUME] Loaded scheduler state from {scheduler_path}")

        print(
            f"[RESUME] resume_global_step={resume_global_step}, "
            f"resume_optimizer_step={resume_optimizer_step}"
        )

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    # 6. 로그 파일 생성
    log_mode = "a" if RESUME_CHECKPOINT is not None and LOG_PATH.exists() else "w"

    with open(LOG_PATH, log_mode, newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)

        if log_mode == "w":
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

        try:
            for epoch in range(NUM_EPOCHS):
                print(f"[EPOCH] Start epoch {epoch}")

                for batch in train_loader:
                    input_ids = batch["input_ids"].to(DEVICE)
                    labels = batch["labels"].to(DEVICE)

                    with torch.amp.autocast("cuda", enabled=USE_AMP):
                        outputs = model(input_ids=input_ids, labels=labels)
                        loss = outputs["loss"]
                        loss_for_backward = loss / GRAD_ACCUM_STEPS

                    scaler.scale(loss_for_backward).backward()

                    running_loss += loss.item()
                    running_count += 1
                    global_step += 1

                    if global_step % GRAD_ACCUM_STEPS != 0:
                        continue

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

                    if MAX_TRAIN_STEPS is not None and optimizer_step >= MAX_TRAIN_STEPS:
                        print(f"[STOP] Reached MAX_TRAIN_STEPS={MAX_TRAIN_STEPS}")
                        save_checkpoint(model, optimizer, scheduler, optimizer_step)
                        return

        except KeyboardInterrupt:
            print("\n[INTERRUPT] KeyboardInterrupt received. Saving checkpoint before exit...")
            save_checkpoint(model, optimizer, scheduler, optimizer_step)

        finally:
            print("Training stopped/finished.")
            print(f"Last optimizer_step: {optimizer_step}")
            print(f"Log saved to: {LOG_PATH}")


if __name__ == "__main__":
    main(use_small_test_model=False)
