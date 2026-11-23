"""CPU-only per-pair descriptive preprocessing after raw-vote schema verification."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from .core import canonical_pair_id, vote_summary


def main() -> None:
    p = argparse.ArgumentParser()
    for arg in ("dataset", "votes-field", "prompt-field", "response1-field", "response2-field", "out", "report"):
        p.add_argument("--" + arg, required=True)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--revision", required=True, help="Resolved 40-character dataset commit SHA from inspection report")
    p.add_argument("--source-split-field", default="split")
    p.add_argument("--vote-format", choices=("helpsteer2", "numeric"), default="helpsteer2")
    p.add_argument("--counts", help="Optional CSV of observed strata; no margin bins are assumed")
    p.add_argument("--min-annotators", type=int, default=1)
    p.add_argument("--strict", action="store_true", help="Stop on first unparseable vote record")
    a = p.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", a.revision):
        p.error("--revision must be a 40-character commit SHA")
    ds = load_dataset(a.dataset, data_dir=a.data_dir, split=a.split, revision=a.revision)
    requested = [a.votes_field, a.prompt_field, a.response1_field, a.response2_field, a.source_split_field]
    missing = [key for key in requested if key not in ds.column_names]
    if missing:
        p.error(f"Missing columns {missing}; run inspect_dataset first")
    records, errors = [], []
    for index, row in enumerate(ds):
        try:
            prompt = row[a.prompt_field]
            r1 = row[a.response1_field]
            r2 = row[a.response2_field]
            if not all(isinstance(x, str) and x.strip() for x in (prompt, r1, r2)):
                raise ValueError("Prompt and responses must be nonempty strings")
            metrics = vote_summary(row[a.votes_field], vote_format=a.vote_format)
            if metrics["n_annotators"] < a.min_annotators:
                continue
            records.append({
                "source_row": index,
                "pair_id": canonical_pair_id(prompt, r1, r2),
                "prompt": prompt,
                "response_1": r1,
                "response_2": r2,
                "source_split": row.get(a.source_split_field),
                "raw_annotations": json.dumps(row[a.votes_field], ensure_ascii=False),
                "published_preference_strength": row.get("preference_strength"),
                **metrics,
            })
        except (ValueError, TypeError) as exc:
            if a.strict:
                raise ValueError(f"Row {index}: {exc}") from exc
            errors.append({"source_row": index, "reason": str(exc)})
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise RuntimeError("No valid rows; inspect schema and parser assumptions before proceeding")
    if frame.pair_id.duplicated().any():
        raise ValueError("Duplicate canonical pair IDs; resolve duplicate source rows before writing output")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    if a.counts:
        count_path = Path(a.counts)
        count_path.parent.mkdir(parents=True, exist_ok=True)
        (frame.groupby(["source_split", "n_annotators", "unanimous_raw", "contested_raw", "all_ties_raw"], dropna=False)
         .size().rename("count").reset_index().to_csv(count_path, index=False))
    counts = frame.groupby(["unanimous_raw", "contested_raw", "n_annotators"], dropna=False).size()
    report = {
        "dataset": a.dataset, "data_dir": a.data_dir, "split": a.split,
        "requested_revision": a.revision,
        "vote_format": a.vote_format,
        "rows_in_source": len(ds), "rows_kept": len(frame),
        "rows_skipped_due_to_min_annotators": len(ds)-len(frame)-len(errors),
        "rows_invalid": len(errors), "invalid_examples": errors[:30],
        "invalid_reason_counts": dict(Counter(e["reason"] for e in errors)),
        "raw_zero_vote_rows": int((frame.tie_votes > 0).sum()),
        "all_tie_rows": int(frame.all_ties_raw.sum()),
        "published_zero_strength_rows": int((frame.published_preference_strength == 0).sum()),
        "duplicate_canonical_pair_ids": int(frame.pair_id.duplicated().sum()),
        "source_split_counts": {str(k): int(v) for k, v in Counter(frame.source_split.astype(str)).items()},
        "stratum_count_by_n_annotators": [
            {"unanimous": bool(u), "contested": bool(c), "n_annotators": int(n), "count": int(v)}
            for (u,c,n),v in counts.items()
        ],
        "caution": "Raw mean margin is NOT published preference_strength. Zero votes are retained; unanimous requires all votes to have one nonzero sign. No train/eval splits or H2 matching created.",
    }
    report_path = Path(a.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {len(frame)} rows to {out}; audit: {report_path}; invalid={len(errors)}")


if __name__ == "__main__":
    main()
