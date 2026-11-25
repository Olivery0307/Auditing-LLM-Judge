"""Rank every H1-H3-eligible pair within its margin group, in a fixed seeded order.

Eligible: `audit_eligible` in the pairs file and partition `train` or `audit`
in the frozen manifest (development and excluded pairs never enter). Within
each margin group, pairs are ordered by SHA-256 of seed and pair_id, so the
order does not depend on row order. Taking ranks 1..k from each group gives
a sample of any size, and a larger k always contains a smaller one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

SAMPLE_PARTITIONS = ("train", "audit")


def rank_key(seed: int, pair_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{pair_id}".encode()).hexdigest()


def rank_pairs(frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """frame needs pair_id, margin_group, partition. Returns ranks 1..n per margin group."""
    if frame.pair_id.duplicated().any():
        raise ValueError("Pair IDs must be unique")
    if frame.margin_group.isna().any():
        raise ValueError("Every ranked pair needs a margin group")
    out = frame[["pair_id", "margin_group", "partition"]].copy()
    out["_key"] = [rank_key(seed, p) for p in out.pair_id]
    out = out.sort_values(["margin_group", "_key"]).drop(columns="_key")
    out["rank"] = out.groupby("margin_group").cumcount() + 1
    return out[["margin_group", "rank", "pair_id", "partition"]].reset_index(drop=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", required=True)
    p.add_argument("--manifest", required=True, help="Frozen manifest CSV; its .json metadata must sit beside it")
    p.add_argument("--out", required=True, help="New ranking CSV; refuses overwrite")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    pairs_path, manifest_path, out = Path(a.pairs), Path(a.manifest), Path(a.out)
    meta = json.loads(manifest_path.with_suffix(".json").read_text())
    if _sha256(manifest_path) != meta["manifest_sha256"]:
        raise ValueError("Manifest does not match its frozen metadata hash")
    if _sha256(pairs_path) != meta["pairs_sha256"]:
        raise ValueError("Pairs file is not the one the manifest was frozen from")
    if out.exists() or out.with_suffix(".json").exists():
        raise FileExistsError("Ranking already exists; refusing to overwrite")
    pairs = pd.read_parquet(pairs_path)
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    frame = pairs.merge(manifest[["pair_id", "partition"]], on="pair_id", how="left", validate="one_to_one")
    if frame.partition.isna().any():
        raise ValueError("Some pairs are missing from the manifest")
    eligible = frame[frame.audit_eligible & frame.partition.isin(SAMPLE_PARTITIONS)]
    ranking = rank_pairs(eligible, seed=a.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(out, index=False)
    counts = ranking.groupby("margin_group").size()
    payload = {
        "seed": a.seed,
        "eligible": "audit_eligible and partition in " + ", ".join(SAMPLE_PARTITIONS),
        "order": "sha256(seed NUL pair_id) ascending within margin_group",
        "pairs_sha256": meta["pairs_sha256"],
        "manifest_sha256": meta["manifest_sha256"],
        "counts_by_margin_group": {str(k): int(v) for k, v in counts.items()},
        "counts_by_partition": {str(k): int(v) for k, v in ranking.partition.value_counts().items()},
        "ranking_sha256": _sha256(out),
    }
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Ranked {len(ranking)} pairs to {out}: {payload['counts_by_margin_group']}")


if __name__ == "__main__":
    main()
