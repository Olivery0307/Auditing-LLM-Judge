import pandas as pd
import pytest

from judge_audit.splits import partition_pairs


def sample():
    return pd.DataFrame([
        {"pair_id": "a", "prompt": "same  prompt", "response_1": "a", "response_2": "b", "source_split": "train"},
        {"pair_id": "b", "prompt": "same\nprompt", "response_1": "c", "response_2": "d", "source_split": "train"},
        {"pair_id": "c", "prompt": "held out", "response_1": "e", "response_2": "f", "source_split": "val"},
    ])


def test_prompt_grouping_and_source_holdout():
    result = partition_pairs(sample(), seed=42, dev_fraction=0.5)
    assert result.loc[0, "partition"] == result.loc[1, "partition"]
    assert result.loc[2, "partition"] == "audit"
    assert result.equals(partition_pairs(sample().iloc[::-1], seed=42, dev_fraction=0.5))


def test_rejects_source_crossing_and_normalized_duplicate():
    frame = sample()
    frame.loc[2, "prompt"] = "same prompt"
    with pytest.raises(ValueError, match="crosses"):
        partition_pairs(frame, seed=42, dev_fraction=0.5)
    frame = sample()
    frame.loc[1, ["response_1", "response_2"]] = ["b", "a"]
    with pytest.raises(ValueError, match="Duplicate normalized"):
        partition_pairs(frame, seed=42, dev_fraction=0.5)


def test_exclusions_mark_pairs_without_moving_others():
    from judge_audit.splits import apply_exclusions
    base = partition_pairs(sample(), seed=42, dev_fraction=0.5)
    result = apply_exclusions(base, pd.DataFrame({"pair_id": ["a"], "reason": ["greeting"]}))
    assert result.set_index("pair_id").partition["a"] == "excluded"
    assert result.set_index("pair_id").exclusion_reason["a"] == "greeting"
    assert (result.partition[result.pair_id != "a"] == base.partition[base.pair_id != "a"]).all()
    with pytest.raises(ValueError, match="audit"):
        apply_exclusions(base, pd.DataFrame({"pair_id": ["c"], "reason": ["x"]}))
    with pytest.raises(ValueError, match="not in the pairs file"):
        apply_exclusions(base, pd.DataFrame({"pair_id": ["zzz"], "reason": ["x"]}))


def test_revision_must_match():
    from judge_audit.splits import check_revision
    frame = sample().assign(dataset_revision="abc")
    check_revision(frame, "abc")
    with pytest.raises(ValueError, match="does not match"):
        check_revision(frame, "def")
    with pytest.raises(ValueError, match="no dataset_revision"):
        check_revision(sample(), "abc")
