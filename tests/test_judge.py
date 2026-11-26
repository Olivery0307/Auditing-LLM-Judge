import json

import pandas as pd
import pytest

from judge_audit.backends import FakeBackend
from judge_audit.items import helpsteer_items, mt_bench_items, select_pairs
from judge_audit.judge import judge_items, p_first
from judge_audit.prompts import render


def pairs():
    turns = json.dumps([{"role": "user", "content": "Question?"}])
    return pd.DataFrame([
        {"pair_id": "p1", "turns": turns, "response_1": "short", "response_2": "a much longer answer " * 20,
         "audit_eligible": True},
        {"pair_id": "p2", "turns": turns, "response_1": "x", "response_2": "y", "audit_eligible": True},
        {"pair_id": "p3", "turns": turns, "response_1": "x", "response_2": "z", "audit_eligible": True},
    ])


def manifest():
    return pd.DataFrame({"pair_id": ["p1", "p2", "p3"], "partition": ["train", "development", "audit"]})


def test_render_swaps_responses_and_keeps_context():
    item = helpsteer_items(pairs().head(1), "h1h3")[0]
    first = render(item, (1, 2), "baseline_v1")[-1]["content"]
    swapped = render(item, (2, 1), "baseline_v1")[-1]["content"]
    assert first.index("short") < first.index("longer")
    assert swapped.index("longer") < swapped.index("short")
    assert "User: Question?" in first and first.endswith("A or B.")
    assert "order of importance" in render(item, (1, 2), "rubric_v1")[-1]["content"]


def test_verdicts_map_back_to_original_response():
    items = helpsteer_items(pairs().head(1), "h1h3")
    result = judge_items(items, FakeBackend(position_bias=0.0, length_weight=5.0), prompt_id="baseline_v1")
    assert result.order.tolist() == ["12", "21"]
    assert (result.verdict == 2).all()          # longer response wins in both orders
    assert result.p_response_2.gt(0.5).all()
    assert p_first(0.0, float("-inf")) == 1.0


def test_position_bias_shows_up_as_a_flip():
    items = helpsteer_items(pairs().iloc[[1]], "h1h3")   # equal-length responses
    result = judge_items(items, FakeBackend(position_bias=1.0, length_weight=0.0), prompt_id="baseline_v1")
    assert result.verdict.tolist() == [1, 2]       # always picks whatever is shown first


def test_edits_create_extra_items():
    edits = pd.DataFrame([{"pair_id": "p1", "edit_id": "lengthen_filler", "target_response": 1,
                           "applicable": True, "edited_text": "short, but longer now"},
                          {"pair_id": "p1", "edit_id": "format_bullets", "target_response": 1,
                           "applicable": False, "edited_text": "short"}])
    items = helpsteer_items(pairs().head(1), "h1h3", edits)
    assert [(i.edit_id, i.target_response) for i in items] == [("none", 0), ("lengthen_filler", 1)]
    assert items[1].response_1 == "short, but longer now" and items[1].response_2 == items[0].response_2


def test_select_pairs_enforces_partitions():
    ranking = pd.DataFrame({"pair_id": ["p1"], "margin_group": ["1"], "rank": [1]})
    assert select_pairs(pairs(), manifest(), ranking, source="h1h3", k=1).pair_id.tolist() == ["p1"]
    assert select_pairs(pairs(), manifest(), None, source="development").pair_id.tolist() == ["p2"]
    assert select_pairs(pairs(), manifest(), None, source="audit").pair_id.tolist() == ["p3"]
    bad = pd.DataFrame({"pair_id": ["p2"], "margin_group": ["1"], "rank": [1]})
    with pytest.raises(ValueError, match="development"):
        select_pairs(pairs(), manifest(), bad, source="h1h3", k=1)


def test_mt_bench_turn2_keeps_each_sides_context():
    def conv(answer1, answer2):
        return json.dumps([{"role": "user", "content": "q1"}, {"role": "assistant", "content": answer1},
                           {"role": "user", "content": "q2"}, {"role": "assistant", "content": answer2}])
    frame = pd.DataFrame([{"comparison_id": "c", "conversation_1": conv("m1 t1", "m1 t2"),
                           "conversation_2": conv("m2 t1", "m2 t2")}])
    item = mt_bench_items(frame)[0]
    assert (item.response_1, item.response_2) == ("m1 t2", "m2 t2")
    text = render(item, (1, 2), "baseline_v1")[-1]["content"]
    assert "[Conversation A]" in text and "m1 t1" in text and "m2 t1" in text
