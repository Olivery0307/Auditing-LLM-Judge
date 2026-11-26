import pandas as pd
import pytest

from judge_audit.ranking import rank_pairs


def frame(n=40):
    groups = ["1", "2", "3", "contested"]
    return pd.DataFrame({"pair_id": [f"p{i}" for i in range(n)],
                         "margin_group": [groups[i % 4] for i in range(n)],
                         "partition": ["train"] * n})


def test_ranks_are_complete_per_group_and_order_independent():
    ranked = rank_pairs(frame(), seed=42)
    for _, group in ranked.groupby("margin_group"):
        assert sorted(group["rank"]) == list(range(1, len(group) + 1))
    shuffled = rank_pairs(frame().sample(frac=1, random_state=1), seed=42)
    assert ranked.equals(shuffled)
    assert not ranked.equals(rank_pairs(frame(), seed=7))


def test_removing_a_pair_keeps_everyone_else_in_order():
    full = rank_pairs(frame(), seed=42)
    fewer = rank_pairs(frame().iloc[1:], seed=42)
    order = lambda r: r.sort_values(["margin_group", "rank"]).pair_id.tolist()
    assert [p for p in order(full) if p != "p0"] == order(fewer)


def test_rejects_duplicates_and_missing_groups():
    bad = frame()
    bad.loc[1, "pair_id"] = "p0"
    with pytest.raises(ValueError):
        rank_pairs(bad, seed=42)
    bad = frame()
    bad.loc[0, "margin_group"] = None
    with pytest.raises(ValueError):
        rank_pairs(bad, seed=42)
