"""Build LoRA training data for the H4 plain vs presentation-augmented comparison (prototype).

Both conditions use the same train-partition pairs, both answer orders and the
same labels, so they differ only in augmentation:
  plain       original responses
  augmented   for about half of the examples, one response (winner or loser,
              chosen by hash) gets a formatting or filler edit; the label stays
Labels: the response favored by both the majority and the mean vote. Pairs
where these disagree or tie are left out of training (see DECISIONS.md).
A development set (both orders) is written for choosing settings and checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .items import Item, helpsteer_items
from .perturb import EDITS
from .prompts import render

AUGMENT_EDITS = ("format_bullets", "strip_markdown", "lengthen_filler")
ORDERS = ((1, 2), (2, 1))


def training_label(mean_signed_margin: float, positive_votes: int, negative_votes: int) -> int | None:
    """1 or 2 when majority and mean agree on a winner; None otherwise."""
    majority = positive_votes - negative_votes
    if mean_signed_margin == 0 or majority == 0 or (mean_signed_margin > 0) != (majority > 0):
        return None
    return 2 if mean_signed_margin > 0 else 1


def _unit(*parts) -> float:
    digest = hashlib.sha256("\0".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _augment(item: Item, seed: int, order: tuple[int, int]) -> Item:
    """Apply one formatting/filler edit to one side for about half of the examples."""
    key = (seed, item.item_id, order)
    if _unit(*key, "augment") >= 0.5:
        return item
    target = 1 if _unit(*key, "side") < 0.5 else 2
    start = int(_unit(*key, "edit") * len(AUGMENT_EDITS))
    for k in range(len(AUGMENT_EDITS)):
        edit_id = AUGMENT_EDITS[(start + k) % len(AUGMENT_EDITS)]
        original = item.response_1 if target == 1 else item.response_2
        edited = EDITS[edit_id](original)
        if edited != original:
            r1, r2 = (edited, item.response_2) if target == 1 else (item.response_1, edited)
            return Item(item.item_id, item.source, item.context_1, item.context_2, r1, r2, edit_id, target)
    return item


def build_examples(pairs: pd.DataFrame, *, augment: bool, seed: int, prompt_id: str = "baseline_v1") -> list[dict]:
    labels = {r.pair_id: training_label(r.mean_signed_margin_raw, r.positive_votes, r.negative_votes)
              for r in pairs.itertuples(index=False)}
    examples = []
    for item in helpsteer_items(pairs, "lora"):
        label = labels[item.item_id]
        for order in ORDERS:
            shown = _augment(item, seed, order) if augment else item
            examples.append({
                "pair_id": item.item_id, "order": f"{order[0]}{order[1]}", "label": label,
                "answer": "A" if order[0] == label else "B",
                "edit_id": shown.edit_id, "target_response": shown.target_response,
                "messages": render(shown, order, prompt_id),
            })
    return examples


def select_training_pairs(pairs: pd.DataFrame, manifest: pd.DataFrame, *, partition: str, seed: int,
                          max_pairs: int | None) -> tuple[pd.DataFrame, dict]:
    frame = pairs.merge(manifest[["pair_id", "partition"]], on="pair_id", validate="one_to_one")
    frame = frame[(frame.partition == partition) & frame.audit_eligible]
    label = [training_label(r.mean_signed_margin_raw, r.positive_votes, r.negative_votes) for r in frame.itertuples()]
    stats = {"eligible": len(frame), "dropped_ambiguous_label": int(sum(x is None for x in label))}
    frame = frame[[x is not None for x in label]]
    frame = frame.assign(_o=[_unit(seed, p, "select") for p in frame.pair_id]).sort_values("_o").drop(columns="_o")
    if max_pairs:
        frame = frame.head(max_pairs)
    stats["selected"] = len(frame)
    return frame, stats


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", default="data/processed/helpsteer_pairs.parquet")
    p.add_argument("--manifest", default="data/splits/helpsteer_manifest.csv")
    p.add_argument("--max-pairs", type=int, default=3000, help="Training pairs (each gives two examples)")
    p.add_argument("--dev-pairs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/lora")
    a = p.parse_args()
    pairs, manifest = pd.read_parquet(a.pairs), pd.read_csv(a.manifest, keep_default_na=False)
    train, train_stats = select_training_pairs(pairs, manifest, partition="train", seed=a.seed, max_pairs=a.max_pairs)
    dev, dev_stats = select_training_pairs(pairs, manifest, partition="development", seed=a.seed, max_pairs=a.dev_pairs)
    out = Path(a.out)
    plain = build_examples(train, augment=False, seed=a.seed)
    augmented = build_examples(train, augment=True, seed=a.seed)
    _write_jsonl(out / "train_plain.jsonl", plain)
    _write_jsonl(out / "train_augmented.jsonl", augmented)
    _write_jsonl(out / "dev.jsonl", build_examples(dev, augment=False, seed=a.seed))
    meta = {
        "seed": a.seed, "train": train_stats, "dev": dev_stats, "examples_per_condition": len(plain),
        "augmented_examples_edited": sum(e["edit_id"] != "none" for e in augmented),
        "augment_edits": list(AUGMENT_EDITS), "label_rule": "majority and mean vote agree; others dropped",
        "answer_share_A": round(sum(e["answer"] == "A" for e in plain) / len(plain), 4),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
