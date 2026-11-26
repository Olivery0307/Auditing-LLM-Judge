"""Turn prepared data into judge items: two responses with the conversation before them.

HelpSteer2 items share one context. MT-Bench turn-2 items have a different
context per side, because each model's own turn-1 answer is part of it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import pandas as pd

SOURCES = ("h1h3", "audit", "development", "mt_bench")


@dataclass(frozen=True)
class Item:
    item_id: str                 # pair_id or MT-Bench comparison_id
    source: str
    context_1: tuple = field(repr=False)   # messages before response 1
    context_2: tuple = field(repr=False)   # messages before response 2
    response_1: str = field(repr=False)
    response_2: str = field(repr=False)
    edit_id: str = "none"
    target_response: int = 0     # 0 when unedited


def _freeze(messages: list[dict]) -> tuple:
    return tuple((m["role"], m["content"]) for m in messages)


def helpsteer_items(pairs: pd.DataFrame, source: str, edits: pd.DataFrame | None = None) -> list[Item]:
    """Unedited items for every pair, plus one item per applicable edit of these pairs."""
    items, by_id = [], {}
    for row in pairs.itertuples(index=False):
        context = _freeze(json.loads(row.turns))
        item = Item(row.pair_id, source, context, context, row.response_1, row.response_2)
        items.append(item)
        by_id[row.pair_id] = item
    if edits is not None:
        chosen = edits[edits.applicable & edits.pair_id.isin(by_id.keys())]
        for e in chosen.itertuples(index=False):
            base = by_id[e.pair_id]
            r1 = e.edited_text if e.target_response == 1 else base.response_1
            r2 = e.edited_text if e.target_response == 2 else base.response_2
            items.append(Item(base.item_id, source, base.context_1, base.context_2, r1, r2,
                              e.edit_id, int(e.target_response)))
    return items


def mt_bench_items(judgments: pd.DataFrame) -> list[Item]:
    """One item per comparison; the judged reply is the last assistant message."""
    items = []
    for row in judgments.drop_duplicates("comparison_id").sort_values("comparison_id").itertuples(index=False):
        conv_1, conv_2 = json.loads(row.conversation_1), json.loads(row.conversation_2)
        items.append(Item(row.comparison_id, "mt_bench", _freeze(conv_1[:-1]), _freeze(conv_2[:-1]),
                          conv_1[-1]["content"], conv_2[-1]["content"]))
    return items


def _hash_order(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}\0{key}".encode()).hexdigest()


def select_pairs(pairs: pd.DataFrame, manifest: pd.DataFrame, ranking: pd.DataFrame | None, *,
                 source: str, k: int | None = None, limit: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Pairs for a HelpSteer2 source, enforcing which partition each source may use."""
    frame = pairs.merge(manifest[["pair_id", "partition"]], on="pair_id", validate="one_to_one")
    if source == "h1h3":
        if ranking is None or k is None:
            raise ValueError("h1h3 needs the ranking and k")
        chosen = ranking.loc[ranking["rank"] <= k, "pair_id"]
        out = frame[frame.pair_id.isin(chosen)]
        if out.partition.isin(["development", "excluded"]).any():
            raise ValueError("Ranking contains development or excluded pairs")
        return out
    if source == "audit":
        out = frame[(frame.partition == "audit") & frame.audit_eligible]
    elif source == "development":
        out = frame[(frame.partition == "development") & frame.audit_eligible]
        out = out.assign(_o=[_hash_order(seed, p) for p in out.pair_id]).sort_values("_o").drop(columns="_o")
    else:
        raise ValueError(f"Unknown HelpSteer2 source {source!r}")
    return out.head(limit) if limit else out
