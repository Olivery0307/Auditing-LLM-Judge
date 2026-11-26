"""Run a judge over items in both answer orders and save one row per judgment.

Sources and the partitions they may touch:
  h1h3         ranks 1..k per margin group of the frozen ranking (+ optional edits)
  development  practice pairs: pilot, prompt and edit choices, LoRA settings (--limit n)
  audit        source validation pairs: H4 on unseen HelpSteer2 pairs
  mt_bench     all MT-Bench comparisons: H4 transfer

A LoRA adapter may only be used with development, audit or mt_bench, because
it trained on train-partition pairs. Each output row stores the verdict
mapped back to the original response (1 or 2), so swaps never need re-mapping.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pandas as pd

from .backends import make_backend
from .core import original_response_id
from .items import SOURCES, helpsteer_items, mt_bench_items, select_pairs
from .prompts import PROMPTS, prompt_sha, render

ORDERS = ((1, 2), (2, 1))


def p_first(logprob_a: float, logprob_b: float) -> float:
    """Probability of 'A' among the two answers."""
    if math.isnan(logprob_a) or math.isnan(logprob_b):
        return math.nan
    m = max(logprob_a, logprob_b)
    ea, eb = math.exp(logprob_a - m), math.exp(logprob_b - m)
    return ea / (ea + eb)


def judge_items(items, backend, *, prompt_id: str, batch_size: int = 16) -> pd.DataFrame:
    jobs = [(item, order) for item in items for order in ORDERS]
    rows = []
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start:start + batch_size]
        t0 = time.perf_counter()
        scores = backend.score([render(item, order, prompt_id) for item, order in chunk])
        seconds = (time.perf_counter() - t0) / len(chunk)
        for (item, order), s in zip(chunk, scores):
            pa = p_first(s.logprob_a, s.logprob_b)
            p2 = pa if order[0] == 2 else (1 - pa if not math.isnan(pa) else math.nan)
            verdict = original_response_id("A" if pa >= 0.5 else "B", order) if not math.isnan(pa) else 0
            rows.append({
                "item_id": item.item_id, "source": item.source, "edit_id": item.edit_id,
                "target_response": item.target_response, "order": f"{order[0]}{order[1]}",
                "logprob_a": s.logprob_a, "logprob_b": s.logprob_b, "p_a": pa, "p_response_2": p2,
                "verdict": verdict, "n_input_tokens": s.n_input_tokens, "skip_reason": s.skip_reason,
                "seconds": seconds,
            })
    return pd.DataFrame(rows)


def _resolve_revision(model: str, revision: str | None) -> str | None:
    if Path(model).exists():
        return None
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(model, revision=revision).sha
    except (OSError, ValueError):  # offline or not a hub model; record what was asked for
        return revision


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=SOURCES, required=True)
    p.add_argument("--k", type=int, default=400, help="h1h3: ranks per margin group")
    p.add_argument("--limit", type=int, help="development/audit: first n pairs")
    p.add_argument("--edits", help="Edits parquet from judge_audit.perturb (HelpSteer2 sources)")
    p.add_argument("--pairs", default="data/processed/helpsteer_pairs.parquet")
    p.add_argument("--manifest", default="data/splits/helpsteer_manifest.csv")
    p.add_argument("--ranking", default="data/splits/h1h3_ranking.csv")
    p.add_argument("--mt-bench", default="data/processed/mt_bench_judgments.parquet")
    p.add_argument("--backend", choices=("fake", "transformers", "vllm"), required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--revision")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--adapter", help="LoRA adapter directory (transformers backend)")
    p.add_argument("--prompt", choices=sorted(PROMPTS), default="baseline_v1")
    p.add_argument("--max-input-tokens", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", required=True, help="New parquet file; metadata goes next to it as .json")
    a = p.parse_args()

    out = Path(a.out)
    if out.exists() or out.with_suffix(".json").exists():
        p.error(f"{out} already exists; choose a new name")
    if a.adapter and a.source == "h1h3":
        p.error("A LoRA adapter trained on train-partition pairs cannot be scored on the H1-H3 sample")

    if a.source == "mt_bench":
        if a.edits:
            p.error("Edits are only prepared for HelpSteer2 pairs")
        items = mt_bench_items(pd.read_parquet(a.mt_bench))
    else:
        ranking = pd.read_csv(a.ranking, dtype={"margin_group": str}) if a.source == "h1h3" else None
        pairs = select_pairs(pd.read_parquet(a.pairs), pd.read_csv(a.manifest, keep_default_na=False), ranking,
                             source=a.source, k=a.k, limit=a.limit)
        items = helpsteer_items(pairs, a.source, pd.read_parquet(a.edits) if a.edits else None)

    revision = None if a.backend == "fake" else _resolve_revision(a.model, a.revision)
    options = {} if a.backend == "fake" else {
        "model": a.model, "revision": revision, "dtype": a.dtype, "adapter": a.adapter,
        "max_input_tokens": a.max_input_tokens, "batch_size": a.batch_size,
    }
    backend = make_backend(a.backend, **options)
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    result = judge_items(items, backend, prompt_id=a.prompt, batch_size=a.batch_size * 2)
    wall = time.perf_counter() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)
    meta = {
        "source": a.source, "k": a.k if a.source == "h1h3" else None, "limit": a.limit, "edits": a.edits,
        "backend": a.backend, "model": None if a.backend == "fake" else a.model, "model_revision": revision,
        "dtype": getattr(backend, "dtype", a.dtype), "device": getattr(backend, "device", None),
        "adapter": a.adapter, "prompt_id": a.prompt, "prompt_sha": prompt_sha(a.prompt),
        "max_input_tokens": a.max_input_tokens, "items": len(items), "judgments": len(result),
        "skipped": int((result.skip_reason != "").sum()), "wall_seconds": round(wall, 2),
        "started_utc": started.isoformat(timespec="seconds"), "host": platform.platform(),
        "versions": {k: _version(k) for k in ("judge-audit", "torch", "transformers", "peft", "vllm")},
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {len(result)} judgments for {len(items)} items to {out} in {wall:.1f}s "
          f"(skipped {meta['skipped']})")


if __name__ == "__main__":
    main()
