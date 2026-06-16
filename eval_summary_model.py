import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from base_model import KLLMConfig, KLLMForCausalLM


DATA_ROOT = Path('/home/aiselab/workspace/ko-llm/dataset')
TOKENIZER_DIR = DATA_ROOT / 'tokenizer_bpe_64k'
DEFAULT_SFT_DIR = DATA_ROOT / 'sft' / 'summary'
DEFAULT_VALID_JSONL = DEFAULT_SFT_DIR / 'valid.jsonl'
DEFAULT_TEST_JSONL = DEFAULT_SFT_DIR / 'test.jsonl'
DEFAULT_OUTPUT_DIR = Path('/home/aiselab/workspace/ko-llm/outputs/summary_model_eval')

MAX_LENGTH = 4096
BATCH_SIZE = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_tokenizer(tokenizer_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    if tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer must have eos_token_id.')
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


def load_checkpoint(model, checkpoint_dir: Path):
    ckpt_path = checkpoint_dir / 'pytorch_model.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    print(f'[CHECKPOINT] Loaded from {ckpt_path}')


def truncate_prompt_keep_suffix(prompt_ids: List[int], max_prompt_len: int, suffix_len: int = 32) -> List[int]:
    if len(prompt_ids) <= max_prompt_len:
        return prompt_ids
    if max_prompt_len <= suffix_len:
        return prompt_ids[-max_prompt_len:]
    head_len = max_prompt_len - suffix_len
    return prompt_ids[:head_len] + prompt_ids[-suffix_len:]


class SummarySFTDataset(Dataset):
    """For eval loss/perplexity. Loss is calculated only on response tokens."""
    def __init__(self, jsonl_path: Path, tokenizer, max_length: int):
        if not jsonl_path.exists():
            raise FileNotFoundError(f'SFT jsonl not found: {jsonl_path}')
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        bad = 0
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                prompt = ex.get('prompt')
                response = ex.get('response')
                if not prompt or not response:
                    bad += 1
                    continue
                self.samples.append({'prompt': prompt, 'response': response})
        if not self.samples:
            raise ValueError(f'No valid SFT samples found in {jsonl_path}')
        print(f'[SFT DATA] path: {jsonl_path}')
        print(f'[SFT DATA] valid samples: {len(self.samples)}')
        print(f'[SFT DATA] skipped bad samples: {bad}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ex = self.samples[idx]
        prompt_ids = self.tokenizer(ex['prompt'], add_special_tokens=False)['input_ids']
        response_ids = self.tokenizer(ex['response'], add_special_tokens=False)['input_ids']
        response_ids = response_ids + [self.tokenizer.eos_token_id]

        if len(response_ids) >= self.max_length:
            response_ids = response_ids[: self.max_length - 1] + [self.tokenizer.eos_token_id]
            prompt_ids = []

        max_prompt_len = self.max_length - len(response_ids)
        prompt_ids = truncate_prompt_keep_suffix(prompt_ids, max_prompt_len)

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids.copy()
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch, pad_token_id: int):
    max_len = max(x['input_ids'].size(0) for x in batch)
    input_ids_list, labels_list, attention_mask_list = [], [], []
    for x in batch:
        input_ids = x['input_ids']
        labels = x['labels']
        pad_len = max_len - input_ids.size(0)
        input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)], dim=0)
        labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)], dim=0)
        attention_mask = (input_ids != pad_token_id).long()
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)
    return {
        'input_ids': torch.stack(input_ids_list, dim=0),
        'labels': torch.stack(labels_list, dim=0),
        'attention_mask': torch.stack(attention_mask_list, dim=0),
    }


@torch.no_grad()
def evaluate_loss_ppl(model, dataloader) -> Tuple[float, float, int, int]:
    model.eval()
    total_loss = 0.0
    total_steps = 0
    skipped_steps = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        valid_label_count = (labels[:, 1:] != -100).sum().item()
        if valid_label_count == 0:
            skipped_steps += 1
            continue
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs['loss']
        if loss is None or torch.isnan(loss) or torch.isinf(loss):
            skipped_steps += 1
            continue
        total_loss += loss.item()
        total_steps += 1
    if total_steps == 0:
        eval_loss, perplexity = float('nan'), float('inf')
    else:
        eval_loss = total_loss / total_steps
        perplexity = math.exp(eval_loss) if eval_loss < 20 else float('inf')
    print(f'[VALID EVAL] eval_loss={eval_loss:.6f}, ppl={perplexity:.4f}, valid_steps={total_steps}, skipped_steps={skipped_steps}')
    return eval_loss, perplexity, total_steps, skipped_steps


def get_logits(outputs):
    if isinstance(outputs, dict):
        return outputs['logits']
    if hasattr(outputs, 'logits'):
        return outputs.logits
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    raise TypeError(f'Unsupported model output type: {type(outputs)}')


def sample_next_token(logits: torch.Tensor, temperature: float = 0.0, top_p: float = 0.9) -> torch.Tensor:
    if temperature is None or temperature <= 0:
        return torch.argmax(logits, dim=-1)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    if top_p is not None and 0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove_mask = cumulative > top_p
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum()
        next_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_indices.gather(-1, next_sorted_idx).squeeze(-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def generate_summary(model, tokenizer, prompt: str, max_length: int, max_new_tokens: int, temperature: float, top_p: float) -> str:
    model.eval()
    prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    max_prompt_len = max_length - max_new_tokens
    if max_prompt_len < 1:
        raise ValueError('max_length must be larger than max_new_tokens.')
    prompt_ids = truncate_prompt_keep_suffix(prompt_ids, max_prompt_len)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    generated_ids = []
    for _ in range(max_new_tokens):
        if input_ids.size(1) > max_length:
            input_ids = input_ids[:, -max_length:]
        outputs = model(input_ids=input_ids)
        logits = get_logits(outputs)
        next_token = sample_next_token(logits[0, -1, :], temperature=temperature, top_p=top_p)
        token_id = int(next_token.item())
        if token_id == tokenizer.eos_token_id:
            break
        generated_ids.append(token_id)
        input_ids = torch.cat([input_ids, torch.tensor([[token_id]], dtype=torch.long, device=DEVICE)], dim=1)
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def load_generation_samples(jsonl_path: Path, max_samples: Optional[int] = None):
    if not jsonl_path.exists():
        raise FileNotFoundError(f'test jsonl not found: {jsonl_path}')
    samples, bad = [], 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            prompt, response = ex.get('prompt'), ex.get('response')
            if not prompt or not response:
                bad += 1
                continue
            samples.append({'id': ex.get('id', line_idx), 'prompt': prompt, 'reference': response})
            if max_samples is not None and len(samples) >= max_samples:
                break
    print(f'[TEST DATA] path: {jsonl_path}')
    print(f'[TEST DATA] loaded samples: {len(samples)}')
    print(f'[TEST DATA] skipped bad samples: {bad}')
    return samples


def tokenize_for_rouge(text: str, mode: str = 'char') -> List[str]:
    text = (text or '').strip()
    if mode == 'whitespace':
        return text.split()
    if mode == 'char':
        return [ch for ch in text if not ch.isspace()]
    raise ValueError(f'Unsupported rouge tokenizer mode: {mode}')


def ngrams(tokens: List[str], n: int):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)] if len(tokens) >= n else []


def prf(overlap: int, pred_count: int, ref_count: int):
    if pred_count == 0 or ref_count == 0 or overlap == 0:
        return 0.0, 0.0, 0.0
    p = overlap / pred_count
    r = overlap / ref_count
    f1 = 2 * p * r / (p + r)
    return p, r, f1


def rouge_n(pred: str, ref: str, n: int, mode: str):
    pred_ng = Counter(ngrams(tokenize_for_rouge(pred, mode), n))
    ref_ng = Counter(ngrams(tokenize_for_rouge(ref, mode), n))
    if not pred_ng or not ref_ng:
        return {'p': 0.0, 'r': 0.0, 'f1': 0.0}
    overlap = sum((pred_ng & ref_ng).values())
    p, r, f1 = prf(overlap, sum(pred_ng.values()), sum(ref_ng.values()))
    return {'p': p, 'r': r, 'f1': f1}


def lcs_length(a: List[str], b: List[str]) -> int:
    if len(a) < len(b):
        short, long = a, b
    else:
        short, long = b, a
    prev = [0] * (len(short) + 1)
    for x in long:
        curr = [0]
        for j, y in enumerate(short, start=1):
            curr.append(prev[j - 1] + 1 if x == y else max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l(pred: str, ref: str, mode: str):
    pred_tokens = tokenize_for_rouge(pred, mode)
    ref_tokens = tokenize_for_rouge(ref, mode)
    if not pred_tokens or not ref_tokens:
        return {'p': 0.0, 'r': 0.0, 'f1': 0.0}
    lcs = lcs_length(pred_tokens, ref_tokens)
    p, r, f1 = prf(lcs, len(pred_tokens), len(ref_tokens))
    return {'p': p, 'r': r, 'f1': f1}


def compute_rouge(pred: str, ref: str, mode: str):
    return {'rouge1': rouge_n(pred, ref, 1, mode), 'rouge2': rouge_n(pred, ref, 2, mode), 'rougeL': rouge_l(pred, ref, mode)}


def average_metrics(metrics_list):
    avg = {k: {'p': 0.0, 'r': 0.0, 'f1': 0.0} for k in ['rouge1', 'rouge2', 'rougeL']}
    if not metrics_list:
        return avg
    for key in avg:
        for sub in avg[key]:
            avg[key][sub] = sum(m[key][sub] for m in metrics_list) / len(metrics_list)
    return avg


@torch.no_grad()
def evaluate_rouge(model, tokenizer, test_jsonl: Path, output_dir: Path, max_length: int, max_new_tokens: int, temperature: float, top_p: float, max_samples: Optional[int], rouge_tokenizer: str):
    samples = load_generation_samples(test_jsonl, max_samples=max_samples)
    prediction_path = output_dir / 'summary_predictions.jsonl'
    metrics_list = []
    with open(prediction_path, 'w', encoding='utf-8') as out_f:
        for idx, sample in enumerate(samples, start=1):
            pred = generate_summary(model, tokenizer, sample['prompt'], max_length, max_new_tokens, temperature, top_p)
            ref = sample['reference']
            rouge = compute_rouge(pred, ref, mode=rouge_tokenizer)
            metrics_list.append(rouge)
            out_f.write(json.dumps({'id': sample['id'], 'prediction': pred, 'reference': ref, 'rouge': rouge}, ensure_ascii=False) + '\n')
            if idx % 10 == 0:
                avg_so_far = average_metrics(metrics_list)
                print(f"[ROUGE] {idx}/{len(samples)} R1-F1={avg_so_far['rouge1']['f1']:.4f} R2-F1={avg_so_far['rouge2']['f1']:.4f} RL-F1={avg_so_far['rougeL']['f1']:.4f}")
    avg = average_metrics(metrics_list)
    print(f'[SAVE] predictions: {prediction_path}')
    return avg, prediction_path, len(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Summary model checkpoint dir, e.g. outputs/summary_model/checkpoints/step_XXXXX')
    parser.add_argument('--valid-jsonl', type=str, default=str(DEFAULT_VALID_JSONL))
    parser.add_argument('--test-jsonl', type=str, default=str(DEFAULT_TEST_JSONL))
    parser.add_argument('--tokenizer-dir', type=str, default=str(TOKENIZER_DIR))
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--max-length', type=int, default=MAX_LENGTH)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.0, help='0.0 means greedy decoding.')
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--max-rouge-samples', type=int, default=200, help='Use -1 for all test samples.')
    parser.add_argument('--rouge-tokenizer', type=str, default='char', choices=['char', 'whitespace'])
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint)
    valid_jsonl = Path(args.valid_jsonl)
    test_jsonl = Path(args.test_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_rouge_samples = None if args.max_rouge_samples == -1 else args.max_rouge_samples

    print(f'Device: {DEVICE}')
    print(f'Checkpoint: {checkpoint_dir}')
    print(f'Valid jsonl: {valid_jsonl}')
    print(f'Test jsonl: {test_jsonl}')
    print(f'Output dir: {output_dir}')
    print(f'ROUGE tokenizer: {args.rouge_tokenizer}')

    tokenizer = load_tokenizer(Path(args.tokenizer_dir))
    model = build_model(len(tokenizer))
    load_checkpoint(model, checkpoint_dir)
    model = model.to(DEVICE)
    model.eval()
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.3f}B')

    valid_dataset = SummarySFTDataset(valid_jsonl, tokenizer, args.max_length)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))
    eval_loss, eval_ppl, valid_steps, skipped_steps = evaluate_loss_ppl(model, valid_loader)

    rouge_avg, prediction_path, n_rouge_samples = evaluate_rouge(
        model=model,
        tokenizer=tokenizer,
        test_jsonl=test_jsonl,
        output_dir=output_dir,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_samples=max_rouge_samples,
        rouge_tokenizer=args.rouge_tokenizer,
    )

    results = {
        'checkpoint': str(checkpoint_dir),
        'valid_jsonl': str(valid_jsonl),
        'test_jsonl': str(test_jsonl),
        'eval_loss': eval_loss,
        'perplexity': eval_ppl,
        'valid_steps': valid_steps,
        'skipped_steps': skipped_steps,
        'rouge_tokenizer': args.rouge_tokenizer,
        'rouge_samples': n_rouge_samples,
        'rouge': rouge_avg,
        'prediction_path': str(prediction_path),
        'generation': {'max_new_tokens': args.max_new_tokens, 'temperature': args.temperature, 'top_p': args.top_p},
    }
    results_path = output_dir / 'summary_eval_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print('=' * 60)
    print('[FINAL SUMMARY MODEL EVALUATION]')
    print(f'Checkpoint: {checkpoint_dir.name}')
    print(f'Eval loss: {eval_loss:.6f}')
    print(f'Perplexity: {eval_ppl:.4f}')
    print(f'ROUGE samples: {n_rouge_samples}')
    print(f"ROUGE-1 F1: {rouge_avg['rouge1']['f1']:.4f}")
    print(f"ROUGE-2 F1: {rouge_avg['rouge2']['f1']:.4f}")
    print(f"ROUGE-L F1: {rouge_avg['rougeL']['f1']:.4f}")
    print(f'Results JSON: {results_path}')
    print(f'Predictions JSONL: {prediction_path}')
    print('=' * 60)


if __name__ == '__main__':
    main()
