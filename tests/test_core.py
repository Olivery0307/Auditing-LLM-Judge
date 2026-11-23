import pytest
from judge_audit.core import canonical_pair_id, vote_summary, parse_raw_votes, original_response_id, assert_disjoint


def test_pair_hash_is_order_independent():
    assert canonical_pair_id("q", "a", "b") == canonical_pair_id("q", "b", "a")
    assert canonical_pair_id("q", "a", "b") != canonical_pair_id("q2", "a", "b")


def test_vote_summary_contested_and_unanimous():
    records = lambda strengths: [{"strength": s, "justification": "review"} for s in strengths]
    x = vote_summary(records([-2, -1, 1]))
    assert x["contested_raw"] is True
    assert x["unanimous_raw"] is False
    assert x["mean_signed_margin_raw"] == pytest.approx(-2/3)
    y = vote_summary(records([3,3,1]))
    assert y["unanimous_raw"] is True
    assert y["vote_share_response_2"] == 1.0
    z = vote_summary(records([0, 1, 0]))
    assert z["tie_votes"] == 2
    assert z["unanimous_raw"] is False
    assert z["contested_raw"] is False
    assert z["vote_share_tie"] == pytest.approx(2/3)


def test_votes_fail_closed_for_unverified_shapes():
    with pytest.raises(ValueError):
        parse_raw_votes([{"strength": 1}])
    with pytest.raises(ValueError):
        parse_raw_votes([0,1])
    with pytest.raises(ValueError):
        parse_raw_votes([])
    with pytest.raises(ValueError):
        parse_raw_votes([{"strength": True, "justification": "review"}])
    assert parse_raw_votes([-1, 0, 2], vote_format="numeric") == [-1, 0, 2]


def test_swapped_label_mapping():
    assert original_response_id("A", (1,2)) == 1
    assert original_response_id("A", (2,1)) == 2
    assert original_response_id("B", (2,1)) == 1
    with pytest.raises(ValueError):
        original_response_id("tie", (1,2))


def test_disjoint():
    assert_disjoint({"a", "b"}, {"c"})
    with pytest.raises(ValueError):
        assert_disjoint({"a", "b"}, {"b", "c"})
