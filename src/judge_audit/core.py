"""Pure transformations. All raw preference assumptions are explicit.

The numbers are individual signed preferences, not the publisher's
aggregated preference_strength. Zero is an individual tie vote.
"""
from __future__ import annotations

import hashlib
import json
from typing import Iterable


def canonical_pair_id(prompt: str, response_1: str, response_2: str) -> str:
    """Order-invariant hash for exact original prompt/response text."""
    if not all(isinstance(s, str) for s in (prompt, response_1, response_2)):
        raise TypeError("prompt and both responses must be strings")
    data = json.dumps([prompt, *sorted((response_1, response_2))], ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def parse_raw_votes(raw: object, *, vote_format: str = "helpsteer2") -> list[int]:
    """Parse an explicitly selected vote schema, rejecting unexpected records."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Expected JSON array of individual numeric votes") from exc
    if not isinstance(raw, (tuple, list)) or not raw:
        raise ValueError("Votes must be a nonempty flat list; inspect source schema")
    if vote_format not in {"helpsteer2", "numeric"}:
        raise ValueError("vote_format must be helpsteer2 or numeric")
    values: list[int] = []
    for v in raw:
        if vote_format == "helpsteer2":
            if not isinstance(v, dict) or set(v) != {"strength", "justification"}:
                raise ValueError("Expected HelpSteer2 record with strength and justification")
            if not isinstance(v["justification"], str):
                raise ValueError("Expected string justification")
            v = v["strength"]
        if isinstance(v, bool) or not isinstance(v, (float, int)):
            raise ValueError("Vote strength must be numeric, not boolean")
        f = float(v)
        if not (-3 <= f <= 3) or not f.is_integer():
            raise ValueError("Expected integer vote in -3..3")
        values.append(int(f))
    return values


def vote_summary(raw: object, *, vote_format: str = "helpsteer2") -> dict:
    votes = parse_raw_votes(raw, vote_format=vote_format)
    n = len(votes)
    positives = sum(v > 0 for v in votes)
    negatives = sum(v < 0 for v in votes)
    ties = n - positives - negatives
    mean = sum(votes) / n
    return {
        "raw_votes": json.dumps(votes),
        "n_annotators": n,
        "positive_votes": positives,
        "negative_votes": negatives,
        "tie_votes": ties,
        "mean_signed_margin_raw": mean,
        "absolute_mean_margin_raw": abs(mean),
        "mean_absolute_strength_raw": sum(abs(v) for v in votes) / n,
        "contested_raw": positives > 0 and negatives > 0,
        "unanimous_raw": positives == n or negatives == n,
        "all_ties_raw": ties == n,
        "vote_share_response_2": positives / n,
        "vote_share_response_1": negatives / n,
        "vote_share_tie": ties / n,
    }


def original_response_id(label: str, order: tuple[int, int]) -> int:
    """Convert generated A/B label into original response identity (1 or 2)."""
    if sorted(order) != [1, 2]:
        raise ValueError("order must be a permutation of (1,2)")
    label = label.strip().upper()
    if label not in {"A", "B"}:
        raise ValueError("Only binary A/B labels supported; tie policy TODO")
    return order[0] if label == "A" else order[1]


def assert_disjoint(*groups: Iterable[str]) -> None:
    """Check exact identifiers don't cross partitions; not fuzzy leakage detection."""
    seen: set[str] = set()
    for group in groups:
        current = set(group)
        intersection = seen.intersection(current)
        if intersection:
            raise ValueError(f"Cross-split overlap found ({len(intersection)} identifiers)")
        seen.update(current)
