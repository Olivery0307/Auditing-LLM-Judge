"""Train a LoRA judge on A/B answer tokens (prototype; needs a GPU; NOT RUN in this repo yet).

Loss is computed only on the single answer token after the chat prompt, so
the adapter learns the verdict, not the prompt text. The adapter with the
best development accuracy is kept. Run once per condition and seed, e.g.

  uv run --group model python -m judge_audit.train_lora \
    --train data/lora/train_augmented.jsonl --dev data/lora/dev.jsonl \
    --out results/lora/augmented_seed1 --seed 1

Then score it with `judge_audit.judge --adapter results/lora/augmented_seed1/adapter`
on `audit` or `mt_bench` only.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--revision")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--grad-accum", type=int, default=16, help="Examples per optimizer step (batch size 1)")
    p.add_argument("--max-tokens", type=int, default=4096, help="Longer examples are skipped and counted")
    p.add_argument("--eval-every", type=int, default=100, help="Optimizer steps between dev evaluations")
    p.add_argument("--dev-limit", type=int, default=400, help="Dev examples per evaluation")
    a = p.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    from .backends import _answer_token_ids

    out = Path(a.out)
    if out.exists():
        p.error(f"{out} already exists; choose a new directory")
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    id_a, id_b = _answer_token_ids(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, revision=a.revision, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.to(device)

    def encode(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    def prepare(rows):
        kept, skipped = [], 0
        for row in rows:
            ids = encode(row)
            if len(ids) + 1 > a.max_tokens:
                skipped += 1
                continue
            kept.append((ids, id_a if row["answer"] == "A" else id_b, row))
        return kept, skipped

    train, train_skipped = prepare(load_jsonl(a.train))
    dev, dev_skipped = prepare(load_jsonl(a.dev)[: a.dev_limit])

    @torch.no_grad()
    def evaluate():
        model.eval()
        correct, picks = 0, {}
        for ids, target, row in dev:
            logits = model(input_ids=torch.tensor([ids], device=device)).logits[0, -1]
            pick = id_a if logits[id_a] >= logits[id_b] else id_b
            correct += pick == target
            letter = "A" if pick == id_a else "B"
            chosen = (int(row["order"][0]) if letter == "A" else int(row["order"][1]))
            picks.setdefault(row["pair_id"], []).append(chosen)
        model.train()
        both = [v for v in picks.values() if len(v) == 2]
        return {"dev_accuracy": correct / max(len(dev), 1),
                "dev_position_consistency": sum(v[0] == v[1] for v in both) / max(len(both), 1)}

    optimizer = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=a.lr)
    total_steps = math.ceil(len(train) * a.epochs / a.grad_accum)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(0.03 * total_steps)), total_steps)
    history, best, step, t0 = [], -1.0, 0, time.time()
    out.mkdir(parents=True)
    model.train()
    for epoch in range(a.epochs):
        order = list(range(len(train)))
        random.shuffle(order)
        running = 0.0
        for n, idx in enumerate(order, start=1):
            ids, target, _ = train[idx]
            inputs = torch.tensor([ids + [target]], device=device)
            labels = torch.full_like(inputs, -100)
            labels[0, -1] = target
            loss = model(input_ids=inputs, labels=labels).loss / a.grad_accum
            loss.backward()
            running += loss.item()
            if n % a.grad_accum == 0 or n == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if step % a.eval_every == 0 or step == total_steps:
                    metrics = {"epoch": epoch, "step": step, "train_loss": running, **evaluate(),
                               "elapsed_seconds": round(time.time() - t0, 1)}
                    history.append(metrics)
                    print(json.dumps(metrics), flush=True)
                    if metrics["dev_accuracy"] > best:
                        best = metrics["dev_accuracy"]
                        model.save_pretrained(out / "adapter")
                running = 0.0
    summary = {"args": vars(a), "train_examples": len(train), "train_skipped_too_long": train_skipped,
               "dev_examples": len(dev), "dev_skipped_too_long": dev_skipped, "optimizer_steps": step,
               "best_dev_accuracy": best, "history": history,
               "train_seconds": round(time.time() - t0, 1), "device": device}
    (out / "training.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved best adapter (dev accuracy {best:.3f}) to {out / 'adapter'}")


if __name__ == "__main__":
    main()
