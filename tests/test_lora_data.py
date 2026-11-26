import json

import pandas as pd

from judge_audit.lora_data import build_examples, select_training_pairs, training_label


def test_training_label_requires_majority_and_mean_to_agree():
    assert training_label(1.33, 3, 0) == 2
    assert training_label(-0.33, 1, 2) == 1
    assert training_label(0.0, 2, 2) is None           # no winner
    assert training_label(0.2, 2, 3) is None           # mean says 2, majority says 1
    assert training_label(0.5, 2, 2) is None           # tied vote count


def frame():
    turns = json.dumps([{"role": "user", "content": "Q?"}])
    long = "First paragraph here.\n\nSecond paragraph here."
    rows = [
        {"pair_id": f"p{i}", "turns": turns, "response_1": long, "response_2": f"Answer {i}.\n\nMore.",
         "audit_eligible": True, "mean_signed_margin_raw": 1.0 if i % 2 else -2.0,
         "positive_votes": 3 if i % 2 else 0, "negative_votes": 0 if i % 2 else 3}
        for i in range(20)
    ]
    rows.append({**rows[0], "pair_id": "amb", "mean_signed_margin_raw": 0.0, "positive_votes": 2, "negative_votes": 2})
    return pd.DataFrame(rows)


def test_selection_drops_ambiguous_and_respects_partition():
    manifest = pd.DataFrame({"pair_id": frame().pair_id, "partition": ["train"] * 20 + ["train"]})
    chosen, stats = select_training_pairs(frame(), manifest, partition="train", seed=1, max_pairs=None)
    assert "amb" not in set(chosen.pair_id) and stats == {"eligible": 21, "dropped_ambiguous_label": 1, "selected": 20}
    manifest.loc[0, "partition"] = "audit"
    chosen, _ = select_training_pairs(frame(), manifest, partition="train", seed=1, max_pairs=None)
    assert "p0" not in set(chosen.pair_id)


def test_conditions_share_pairs_labels_and_size():
    pairs = frame().head(20)
    plain = build_examples(pairs, augment=False, seed=3)
    aug = build_examples(pairs, augment=True, seed=3)
    assert len(plain) == len(aug) == 40
    assert [(e["pair_id"], e["order"], e["answer"]) for e in plain] == [(e["pair_id"], e["order"], e["answer"]) for e in aug]
    edited = [e for e in aug if e["edit_id"] != "none"]
    assert 0 < len(edited) < len(aug)
    assert all(e["edit_id"] == "none" for e in plain)
    # the answer letter follows the order: label 2 shown second is "B", shown first is "A"
    p1 = [e for e in plain if e["pair_id"] == "p1"]
    assert {e["order"]: e["answer"] for e in p1} == {"12": "B", "21": "A"}
