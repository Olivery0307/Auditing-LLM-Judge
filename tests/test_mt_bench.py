import json

import pandas as pd
import pytest

from judge_audit.mt_bench import canonical_judgment, judged_messages, prepare


def conv(q1, a1, q2="q2", a2="a2"):
    return [
        {"role": "user", "content": q1}, {"role": "assistant", "content": a1},
        {"role": "user", "content": q2}, {"role": "assistant", "content": a2},
    ]


def test_canonical_judgment_is_order_independent():
    assert canonical_judgment("gpt-4", "alpaca-13b", "model_a") == ("alpaca-13b", "gpt-4", "model_2")
    assert canonical_judgment("alpaca-13b", "gpt-4", "model_b") == ("alpaca-13b", "gpt-4", "model_2")
    assert canonical_judgment("gpt-4", "alpaca-13b", "tie") == ("alpaca-13b", "gpt-4", "tie")
    with pytest.raises(ValueError):
        canonical_judgment("gpt-4", "gpt-4", "tie")
    with pytest.raises(ValueError):
        canonical_judgment("gpt-4", "alpaca-13b", "gpt-4")


def test_judged_messages_cut_to_turn():
    assert len(judged_messages(conv("q", "a"), 1)) == 2
    assert len(judged_messages(conv("q", "a"), 2)) == 4
    with pytest.raises(ValueError):
        judged_messages(conv("q", "a"), 3)
    assert judged_messages(conv("q", " "), 1)[1]["content"] == " "  # empty model answer is kept
    with pytest.raises(ValueError):
        judged_messages(conv(" ", "a"), 1)


def test_prepare_maps_conversations_to_sorted_models_and_flags_repeats():
    frame = pd.DataFrame([
        {"question_id": 81, "turn": 1, "judge": "expert_1", "model_a": "gpt-4", "model_b": "alpaca-13b",
         "winner": "model_a", "conversation_a": conv("q", "from gpt"), "conversation_b": conv("q", "from alpaca")},
        {"question_id": 81, "turn": 1, "judge": "expert_1", "model_a": "alpaca-13b", "model_b": "gpt-4",
         "winner": "model_b", "conversation_a": conv("q", "from alpaca"), "conversation_b": conv("q", "from gpt")},
    ])
    out = prepare(frame, revision="r")
    assert out.winner.tolist() == ["model_2", "model_2"]
    assert out.shown_first.tolist() == ["model_2", "model_1"]
    assert json.loads(out.conversation_2[0])[1]["content"] == "from gpt"
    assert out.comparison_id.nunique() == 1
    assert out.repeat_judgment.all()
    assert not out.has_empty_response.any()


def test_prepare_rejects_different_user_turns():
    frame = pd.DataFrame([{"question_id": 1, "turn": 1, "judge": "j", "model_a": "x", "model_b": "y",
                           "winner": "tie", "conversation_a": conv("q", "a"), "conversation_b": conv("other", "b")}])
    with pytest.raises(ValueError, match="different user turns"):
        prepare(frame, revision="r")
