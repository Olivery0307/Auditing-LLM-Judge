"""Prepare MT-Bench human pairwise judgments for the H4 transfer test.

One row per human judgment; nothing is filtered. Each comparison is put in a
fixed order (models sorted by name), so `winner` reads 'model_1', 'model_2'
or 'tie' regardless of the order the annotator saw. Conversations are cut to
the messages the annotator judged: two for turn 1, four for turn 2.
Empty model answers are real outputs (llama-13b has some) and are kept and
flagged; empty user messages are rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATASET = "lmsys/mt_bench_human_judgments"
SPLIT = "human"


def canonical_judgment(model_a: str, model_b: str, winner: str) -> tuple[str, str, str]:
    """Return (model_1, model_2, winner) with models sorted and winner remapped."""
    if model_a == model_b:
        raise ValueError("A comparison needs two different models")
    if winner not in {"model_a", "model_b", "tie"}:
        raise ValueError(f"Unexpected winner {winner!r}")
    model_1, model_2 = sorted((model_a, model_b))
    if winner == "tie":
        return model_1, model_2, "tie"
    chosen = model_a if winner == "model_a" else model_b
    return model_1, model_2, "model_1" if chosen == model_1 else "model_2"


def judged_messages(conversation: list[dict], turn: int) -> list[dict]:
    """Messages up to and including the judged turn, checked for alternating roles."""
    if turn not in (1, 2):
        raise ValueError(f"Unexpected turn {turn!r}")
    messages = [{"role": m["role"], "content": m["content"]} for m in conversation[: 2 * turn]]
    if [m["role"] for m in messages] != ["user", "assistant"] * turn:
        raise ValueError("Conversation must alternate user/assistant up to the judged turn")
    if any(not isinstance(m["content"], str) for m in messages):
        raise ValueError("Message content must be a string")
    if any(m["role"] == "user" and not m["content"].strip() for m in messages):
        raise ValueError("Empty user message in judged conversation")
    return messages


def comparison_id(question_id: int, turn: int, model_1: str, model_2: str) -> str:
    data = json.dumps([int(question_id), int(turn), model_1, model_2])
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def prepare(frame: pd.DataFrame, *, revision: str) -> pd.DataFrame:
    records = []
    for index, row in enumerate(frame.itertuples(index=False)):
        model_1, model_2, winner = canonical_judgment(row.model_a, row.model_b, row.winner)
        conv = {row.model_a: row.conversation_a, row.model_b: row.conversation_b}
        conv_1 = judged_messages(list(conv[model_1]), int(row.turn))
        conv_2 = judged_messages(list(conv[model_2]), int(row.turn))
        if [m["content"] for m in conv_1 if m["role"] == "user"] != [m["content"] for m in conv_2 if m["role"] == "user"]:
            raise ValueError(f"Row {index}: the two conversations have different user turns")
        records.append({
            "source_row": index,
            "dataset_revision": revision,
            "comparison_id": comparison_id(row.question_id, row.turn, model_1, model_2),
            "question_id": int(row.question_id),
            "turn": int(row.turn),
            "judge": row.judge,
            "model_1": model_1,
            "model_2": model_2,
            "shown_first": "model_1" if row.model_a == model_1 else "model_2",
            "winner": winner,
            "conversation_1": json.dumps(conv_1, ensure_ascii=False),
            "conversation_2": json.dumps(conv_2, ensure_ascii=False),
            "has_empty_response": any(
                m["role"] == "assistant" and not m["content"].strip() for m in conv_1 + conv_2
            ),
        })
    out = pd.DataFrame.from_records(records)
    out["repeat_judgment"] = out.duplicated(["comparison_id", "judge"], keep=False)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--revision", required=True, help="Pinned 40-character dataset commit SHA")
    p.add_argument("--out", required=True)
    p.add_argument("--report", required=True)
    a = p.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", a.revision):
        p.error("--revision must be a 40-character commit SHA")
    source = load_dataset(DATASET, split=SPLIT, revision=a.revision).to_pandas()
    frame = prepare(source, revision=a.revision)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    per_comparison = frame.groupby("comparison_id").size()
    report = {
        "dataset": DATASET, "split": SPLIT, "revision": a.revision,
        "judgments": len(frame),
        "comparisons": int(per_comparison.size),
        "questions": int(frame.question_id.nunique()),
        "judgments_by_turn": {str(k): int(v) for k, v in frame.turn.value_counts().sort_index().items()},
        "winner_counts": dict(Counter(frame.winner)),
        "judgments_per_comparison": {str(k): int(v) for k, v in per_comparison.value_counts().sort_index().items()},
        "repeat_judgment_rows": int(frame.repeat_judgment.sum()),
        "empty_response_rows": int(frame.has_empty_response.sum()),
        "models": sorted(set(frame.model_1) | set(frame.model_2)),
        "note": "No rows filtered. Tie handling, minimum judgments and turn-2 use are analysis decisions.",
    }
    report_path = Path(a.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {len(frame)} judgments ({report['comparisons']} comparisons) to {out}; report: {report_path}")


if __name__ == "__main__":
    main()
