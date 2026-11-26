"""Flag near-duplicate prompts across partitions and against MT-Bench for manual review.

Similarity is Jaccard over word 5-gram shingles (lowercased word tokens).
Prefix filtering finds exactly the pairs at or above the threshold without
comparing every pair. Partitions are recomputed with the same seed and dev
fraction that `splits` will use, so this must run before the split is frozen.

Comparisons and the side proposed for removal if a reviewer confirms a copy:
  train_vs_development   drop the development pair
  train_or_dev_vs_val    drop the train/development pair (val stays intact)
  mt_bench_vs_train_dev  drop the HelpSteer2 pair (MT-Bench stays intact)
  mt_bench_vs_val        report only; both are evaluation data
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .splits import partition_pairs

SHINGLE_WORDS = 5


def shingles(text: str, k: int = SHINGLE_WORDS) -> frozenset[str]:
    words = re.findall(r"\w+", text.lower())
    if len(words) < k:
        return frozenset([" ".join(words)]) if words else frozenset()
    return frozenset(" ".join(words[i : i + k]) for i in range(len(words) - k + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def similar_pairs(left: list[frozenset[str]], right: list[frozenset[str]], threshold: float) -> list[tuple[int, int, float]]:
    """All (i, j, score) with jaccard(left[i], right[j]) >= threshold. Exact."""
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    freq = Counter(s for doc in (*left, *right) for s in doc)
    def prefix(doc: frozenset[str]) -> list[str]:
        ordered = sorted(doc, key=lambda s: (freq[s], s))
        keep = len(ordered) - math.ceil(threshold * len(ordered) - 1e-9) + 1
        return ordered[:keep]
    index: dict[str, list[int]] = defaultdict(list)
    for j, doc in enumerate(right):
        for s in prefix(doc):
            index[s].append(j)
    found = []
    for i, doc in enumerate(left):
        if not doc:
            continue
        candidates = {j for s in prefix(doc) for j in index.get(s, ())}
        for j in sorted(candidates):
            other = right[j]
            if min(len(doc), len(other)) < threshold * max(len(doc), len(other)) - 1e-9:
                continue
            score = jaccard(doc, other)
            if score >= threshold - 1e-12:
                found.append((i, j, score))
    return found


def _snippet(text: str, limit: int = 300) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit] + " ..."


def find_near_duplicates(pairs: pd.DataFrame, mt_bench: pd.DataFrame, *, seed: int, dev_fraction: float,
                         threshold: float) -> pd.DataFrame:
    part = partition_pairs(pairs, seed=seed, dev_fraction=dev_fraction).set_index("pair_id").partition
    hs = pairs.assign(partition=pairs.pair_id.map(part)).reset_index(drop=True)
    hs_shingles = [shingles(p) for p in hs.prompt]
    rows = []

    def add(kind, a, b, score, text_a, text_b, drop):
        rows.append({"kind": kind, "id_a": a["id"], "partition_a": a["part"], "id_b": b["id"],
                     "partition_b": b["part"], "jaccard": round(score, 4), "proposed_drop": drop,
                     "decision": "", "text_a": _snippet(text_a), "text_b": _snippet(text_b)})

    groups = {name: hs.index[hs.partition == name].tolist() for name in ("train", "development", "audit")}
    comparisons = [
        ("train_vs_development", groups["train"], groups["development"], "b"),
        ("train_or_dev_vs_val", groups["train"] + groups["development"], groups["audit"], "a"),
    ]
    for kind, left_ids, right_ids, drop_side in comparisons:
        for i, j, score in similar_pairs([hs_shingles[k] for k in left_ids], [hs_shingles[k] for k in right_ids], threshold):
            a, b = hs.loc[left_ids[i]], hs.loc[right_ids[j]]
            add(kind, {"id": a.pair_id, "part": a.partition}, {"id": b.pair_id, "part": b.partition}, score,
                a.prompt, b.prompt, a.pair_id if drop_side == "a" else b.pair_id)

    # MT-Bench user questions against every HelpSteer2 user turn.
    mt_questions = {}
    for row in mt_bench.itertuples(index=False):
        users = [m["content"] for m in json.loads(row.conversation_1) if m["role"] == "user"]
        for t, text in enumerate(users, start=1):
            mt_questions.setdefault(f"q{row.question_id}_turn{t}", text)
    mt_ids, mt_texts = list(mt_questions), list(mt_questions.values())
    hs_turn_owner, hs_turn_texts = [], []
    for k, row in hs.iterrows():
        for turn in json.loads(row.turns):
            if turn["role"] == "user":
                hs_turn_owner.append(k)
                hs_turn_texts.append(turn["content"])
    best: dict[tuple[str, int], tuple[float, str]] = {}
    for i, j, score in similar_pairs([shingles(t) for t in mt_texts], [shingles(t) for t in hs_turn_texts], threshold):
        key = (mt_ids[i], hs_turn_owner[j])
        if score > best.get(key, (-1.0, ""))[0]:
            best[key] = (score, hs_turn_texts[j])
    for (mt_id, k), (score, hs_text) in sorted(best.items()):
        b = hs.loc[k]
        kind = "mt_bench_vs_val" if b.partition == "audit" else "mt_bench_vs_train_dev"
        add(kind, {"id": mt_id, "part": "mt_bench"}, {"id": b.pair_id, "part": b.partition}, score,
            mt_questions[mt_id], hs_text, "" if kind == "mt_bench_vs_val" else b.pair_id)
    columns = ["kind", "id_a", "partition_a", "id_b", "partition_b", "jaccard", "proposed_drop", "decision", "text_a", "text_b"]
    return pd.DataFrame(rows, columns=columns).sort_values(["kind", "jaccard"], ascending=[True, False]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", required=True)
    p.add_argument("--mt-bench", required=True)
    p.add_argument("--out", required=True, help="Review CSV; contains dataset text, keep out of Git")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dev-fraction", type=float, default=0.1)
    p.add_argument("--threshold", type=float, default=0.8)
    a = p.parse_args()
    result = find_near_duplicates(pd.read_parquet(a.pairs), pd.read_parquet(a.mt_bench),
                                  seed=a.seed, dev_fraction=a.dev_fraction, threshold=a.threshold)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"Flagged {len(result)} pairs at Jaccard >= {a.threshold}: {result.kind.value_counts().to_dict()}; review {out}")


if __name__ == "__main__":
    main()
