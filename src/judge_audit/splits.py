"""Freeze prompt-grouped HelpSteer2 partitions before creating variants.

The source's internal validation rows are kept as a separate held-out audit
partition. Exact whitespace-normalized prompts define groups. Near-duplicate
prompts confirmed by manual review (see `near_duplicates`) are listed in an
exclusions file and marked `excluded`; every other assignment is unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from .core import assert_disjoint


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def partition_pairs(frame: pd.DataFrame, *, seed: int, dev_fraction: float) -> pd.DataFrame:
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    required = {"pair_id", "prompt", "response_1", "response_2", "source_split"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty or frame.pair_id.isna().any() or frame.pair_id.duplicated().any():
        raise ValueError("Pair IDs must be present and unique")
    source = frame.source_split.map(str)
    if not set(source).issubset({"train", "val", "validation"}) or "train" not in set(source) or not ({"val", "validation"} & set(source)):
        raise ValueError("Expected internal source_split train and val/validation values")
    prompt_key = frame.prompt.map(normalized_text)
    if (prompt_key == "").any():
        raise ValueError("Empty normalized prompt")
    crossed = pd.DataFrame({"key": prompt_key, "source": source}).groupby("key").source.nunique()
    if (crossed > 1).any():
        raise ValueError("Normalized prompt crosses the publisher's train/validation split")
    # Detect repeated pairs even if source text differs only in whitespace or order.
    norm_pair = [
        json.dumps([p, *sorted((normalized_text(a), normalized_text(b)))], ensure_ascii=False)
        for p, a, b in zip(prompt_key, frame.response_1, frame.response_2)
    ]
    if len(set(norm_pair)) != len(norm_pair):
        raise ValueError("Duplicate normalized response pair; resolve before splitting")
    def choose(key: str, src: str) -> str:
        if src in {"val", "validation"}:
            return "audit"
        digest = hashlib.sha256(f"{seed}\0{key}".encode()).digest()
        proportion = int.from_bytes(digest[:8], "big") / 2**64
        return "development" if proportion < dev_fraction else "train"
    result = frame[["pair_id", "source_split"]].copy()
    result["prompt_hash"] = prompt_key.map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    result["partition"] = [choose(key, src) for key, src in zip(prompt_key, source)]
    assert_disjoint(*(set(result.loc[result.partition == part, "prompt_hash"]) for part in ("train", "development", "audit")))
    return result.sort_values("pair_id").reset_index(drop=True)


def check_revision(frame: pd.DataFrame, revision: str) -> None:
    """The pairs file must come from exactly the pinned dataset revision."""
    if "dataset_revision" not in frame.columns:
        raise ValueError("Pairs file has no dataset_revision column; rerun preprocess")
    found = set(frame.dataset_revision.astype(str))
    if found != {revision}:
        raise ValueError(f"Pairs file revision {sorted(found)} does not match --source-revision {revision}")


def apply_exclusions(result: pd.DataFrame, exclusions: pd.DataFrame) -> pd.DataFrame:
    """Mark reviewed near-duplicates as excluded. Audit (source validation) pairs are never excluded."""
    if not {"pair_id", "reason"}.issubset(exclusions.columns):
        raise ValueError("Exclusions need pair_id and reason columns")
    if exclusions.pair_id.duplicated().any():
        raise ValueError("Duplicate pair_id in exclusions")
    known = result.set_index("pair_id").partition
    if missing := set(exclusions.pair_id) - set(known.index):
        raise ValueError(f"{len(missing)} excluded pair IDs are not in the pairs file")
    if (known.loc[list(exclusions.pair_id)] == "audit").any():
        raise ValueError("Exclusions may not remove audit (source validation) pairs")
    result = result.copy()
    reason = result.pair_id.map(exclusions.set_index("pair_id").reason)
    result["exclusion_reason"] = reason.fillna("")
    result.loc[reason.notna(), "partition"] = "excluded"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True, help="New CSV manifest path; refuses overwrite")
    parser.add_argument("--source-revision", required=True, help="Pinned Hugging Face commit SHA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--exclusions", help="Reviewed near-duplicate exclusions CSV (pair_id, reason)")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        parser.error("--source-revision must be a 40-character commit SHA")
    pairs = pd.read_parquet(args.pairs)
    check_revision(pairs, args.source_revision)
    result = partition_pairs(pairs, seed=args.seed, dev_fraction=args.dev_fraction)
    if args.exclusions:
        result = apply_exclusions(result, pd.read_csv(args.exclusions))
    out = Path(args.out)
    meta = out.with_suffix(".json")
    if out.exists() or meta.exists():
        raise FileExistsError("Split manifest already exists; frozen partitions cannot be overwritten")
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    payload = {
        "source_revision": args.source_revision,
        "pairs_file": str(args.pairs),
        "pairs_sha256": hashlib.sha256(Path(args.pairs).read_bytes()).hexdigest(),
        "seed": args.seed,
        "dev_fraction": args.dev_fraction,
        "grouping": "exact whitespace-normalized prompt; reviewed near-duplicates excluded",
        "exclusions_file": str(args.exclusions) if args.exclusions else None,
        "exclusions_sha256": hashlib.sha256(Path(args.exclusions).read_bytes()).hexdigest() if args.exclusions else None,
        "source_validation_policy": "held-out audit; never train or development",
        "counts": {str(k): int(v) for k, v in result.partition.value_counts().items()},
        "manifest_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    meta.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {len(result)} frozen pair assignments to {out} and metadata to {meta}")


if __name__ == "__main__":
    main()
