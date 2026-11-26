import numpy as np
import pandas as pd
import pytest

from judge_audit.analysis import (
    Bootstrap,
    _standardized_difference,
    add_agreement,
    edit_level,
    mt_bench_reference,
    pair_level,
)


def judgment(item, order, p2, edit="none", target=0):
    verdict = 2 if p2 >= 0.5 else 1
    return {"item_id": item, "order": order, "p_response_2": p2, "verdict": verdict, "edit_id": edit,
            "target_response": target}


def test_pair_level_flips_and_first_position():
    j = pd.DataFrame([
        judgment("a", "12", 0.9), judgment("a", "21", 0.8),   # stable: response 2 both times
        judgment("b", "12", 0.2), judgment("b", "21", 0.7),   # flips: always picks the first shown
    ])
    d = pair_level(j).set_index("item_id")
    assert d.position_flip.to_dict() == {"a": 0.0, "b": 1.0}
    assert d.prefers_first["b"] == 1.0 and d.prefers_first["a"] == 0.5
    assert d.verdict_balanced["b"] == 1          # (0.2 + 0.7) / 2 = 0.45


def test_edit_level_flip_and_shift_toward_edited_response():
    j = pd.DataFrame([
        judgment("a", "12", 0.4), judgment("a", "21", 0.4),
        judgment("a", "12", 0.6, "lengthen_filler", 2), judgment("a", "21", 0.3, "lengthen_filler", 2),
    ])
    e = edit_level(j).iloc[0]
    assert e.edit_flip == 0.5
    assert e.shift_to_edited == pytest.approx(((0.6 - 0.4) + (0.3 - 0.4)) / 2)


def test_agreement_uses_vote_share_of_chosen_response():
    d = pd.DataFrame({"item_id": ["a"], "verdict_12": [2], "verdict_21": [1], "verdict_balanced": [2]})
    out = add_agreement(d, pd.Series({"a": 0.25}), pd.Series({"a": 0.75}))
    assert out.agreement_single[0] == pytest.approx(0.5)
    assert out.agreement_balanced[0] == pytest.approx(0.75)


def test_bootstrap_point_and_interval():
    x = np.r_[np.zeros(50), np.ones(50)]
    point, boots = Bootstrap(np.arange(100), n_boot=500).mean(x)
    assert point == pytest.approx(0.5)
    assert np.percentile(boots, 2.5) < 0.5 < np.percentile(boots, 97.5)
    weighted, _ = Bootstrap(np.arange(100), unit_weights=np.r_[np.full(50, 3.0), np.ones(50)]).mean(x)
    assert weighted == pytest.approx(0.25)


def test_standardized_difference_removes_stratum_mix():
    # Within each stratum contested pairs flip 0.2 more, but contested pairs sit mostly in the
    # high-flip stratum, so the raw difference is larger than 0.2.
    rows = []
    for stratum, base, n_c, n_u in (("low", 0.1, 10, 90), ("high", 0.5, 90, 10)):
        rows += [(stratum, True, base + 0.2)] * n_c + [(stratum, False, base)] * n_u
    f = pd.DataFrame(rows, columns=["stratum", "contested", "x"])
    boot = Bootstrap(np.arange(len(f)), n_boot=200)
    point, boots, n = _standardized_difference(boot, f.x.to_numpy(), f.contested.to_numpy(), f.stratum.to_numpy())
    raw = f[f.contested].x.mean() - f[~f.contested].x.mean()
    assert raw > 0.4 and point == pytest.approx(0.2) and n == 200
    assert np.nanmean(boots) == pytest.approx(0.2, abs=1e-5)


def test_mt_bench_reference_ignores_ties():
    mt = pd.DataFrame({"comparison_id": ["c", "c", "c", "t"], "winner": ["model_2", "model_1", "tie", "tie"],
                       "question_id": [1, 1, 1, 2], "turn": [1, 1, 1, 1]})
    ref = mt_bench_reference(mt)
    assert ref.loc["c", "share_2"] == 0.5 and ref.loc["c", "n"] == 2
    assert "t" not in ref.index
