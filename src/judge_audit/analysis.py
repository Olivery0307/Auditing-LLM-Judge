"""Turn judge runs into H1-H4 tables with cluster-bootstrap confidence intervals (prototype).

Each run is a parquet from `judge_audit.judge` plus its .json metadata.
Runs with source h1h3 feed H1-H3; runs with source audit or mt_bench feed H4.

Definitions (see DECISIONS.md):
  position flip     the verdict changes when the two responses swap places
  edit flip         the verdict changes when one response is edited (same order), averaged over both orders
  shift to edited   change in the probability of the edited response winning
  soft agreement    share of human annotators who chose the judge's response
  single-order      a standard judge: each order counts as its own verdict
  order-balanced    mitigation: average p(response 2) over both orders, then decide
CIs resample prompt clusters (HelpSteer2) or questions (MT-Bench). H1-H3
overall numbers reweight each margin group to its size in the full pool.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

UNANIMOUS = ("1", "2", "3")


# ---------- per-item tables ----------

def pair_level(judgments: pd.DataFrame) -> pd.DataFrame:
    """One row per unedited item with both orders scored."""
    base = judgments[judgments.edit_id == "none"]
    wide = base.pivot_table(index="item_id", columns="order", values=["p_response_2", "verdict"], aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide[(wide.verdict_12 > 0) & (wide.verdict_21 > 0)].copy()
    wide["position_flip"] = (wide.verdict_12 != wide.verdict_21).astype(float)
    wide["prefers_first"] = ((wide.verdict_12 == 1).astype(float) + (wide.verdict_21 == 2).astype(float)) / 2
    wide["p2_balanced"] = (wide.p_response_2_12 + wide.p_response_2_21) / 2
    wide["verdict_balanced"] = np.where(wide.p2_balanced >= 0.5, 2, 1)
    return wide.reset_index()


def edit_level(judgments: pd.DataFrame) -> pd.DataFrame:
    """One row per (item, edit, edited side): flip rate and shift over the two orders."""
    base = judgments[judgments.edit_id == "none"].set_index(["item_id", "order"])[["verdict", "p_response_2"]]
    e = judgments[judgments.edit_id != "none"].merge(base.reset_index(), on=["item_id", "order"], suffixes=("", "_base"))
    e = e[(e.verdict > 0) & (e.verdict_base > 0)].copy()
    e["edit_flip"] = (e.verdict != e.verdict_base).astype(float)
    to_target = lambda p, t: np.where(t == 2, p, 1 - p)
    e["shift_to_edited"] = to_target(e.p_response_2, e.target_response) - to_target(e.p_response_2_base, e.target_response)
    return e.groupby(["item_id", "edit_id", "target_response"], as_index=False)[["edit_flip", "shift_to_edited"]].mean()


def add_agreement(frame: pd.DataFrame, share_1: pd.Series, share_2: pd.Series) -> pd.DataFrame:
    """Soft agreement with humans; shares are indexed by item_id (non-tie human votes)."""
    s1, s2 = frame.item_id.map(share_1).to_numpy(), frame.item_id.map(share_2).to_numpy()
    pick = lambda v: np.where(v == 2, s2, s1)
    out = frame.copy()
    out["agreement_single"] = (pick(out.verdict_12) + pick(out.verdict_21)) / 2
    out["agreement_balanced"] = pick(out.verdict_balanced)
    return out


# ---------- cluster bootstrap ----------

class Bootstrap:
    """Resample clusters with replacement; every statistic is a weighted mean of unit values."""

    def __init__(self, clusters, *, n_boot: int = 1000, seed: int = 0, unit_weights=None):
        codes, uniques = pd.factorize(pd.Series(clusters).to_numpy())
        rng = np.random.default_rng(seed)
        c = len(uniques)
        draws = rng.multinomial(c, np.full(c, 1 / c), size=n_boot).astype(np.float32)
        self.unit_w = np.ones(len(codes), np.float32) if unit_weights is None else np.asarray(unit_weights, np.float32)
        self.W = draws[:, codes] * self.unit_w

    def mean(self, values, mask=None):
        x = np.asarray(values, np.float32)
        m = np.ones_like(x, bool) if mask is None else np.asarray(mask, bool)
        m = m & ~np.isnan(x)
        x0 = np.where(m, x, 0.0)
        point = float((self.unit_w * x0).sum() / max((self.unit_w * m).sum(), 1e-12))
        num, den = self.W @ x0, self.W @ m.astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            boots = num / den
        return point, boots


def summary(point: float, boots: np.ndarray) -> dict:
    lo, hi = np.nanpercentile(boots, [2.5, 97.5]) if np.isfinite(boots).any() else (np.nan, np.nan)
    return {"estimate": round(point, 4), "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4)}


# ---------- hypothesis tables ----------

def h1_table(d: pd.DataFrame, run: str, n_boot: int) -> list[dict]:
    u = d[d.margin_group.isin(UNANIMOUS)]
    boot = Bootstrap(u.prompt_hash, n_boot=n_boot)
    rows, by_group = [], {}
    for g in UNANIMOUS:
        by_group[g] = boot.mean(u.position_flip, u.margin_group == g)
        rows.append({"run": run, "measure": f"position_flip_margin_{g}", "n": int((u.margin_group == g).sum()),
                     **summary(*by_group[g])})
    diff = (by_group["1"][0] - by_group["3"][0], by_group["1"][1] - by_group["3"][1])
    rows.append({"run": run, "measure": "difference_margin_1_minus_3", "n": len(u), **summary(*diff)})
    return rows


def _standardized_difference(boot, x, contested, strata):
    """Sum over strata of (contested - unanimous) mean flip, weighted by each stratum's share of contested pairs.

    Only strata with both kinds of pair count. Returns (point, bootstrap draws, pairs used).
    """
    keep = [s for s in pd.unique(strata) if (contested & (strata == s)).any() and (~contested & (strata == s)).any()]
    if not keep:
        return float("nan"), np.full(boot.W.shape[0], np.nan), 0
    count = {s: float((contested & (strata == s)).sum()) for s in keep}
    count_boot = {s: boot.W @ (contested & (strata == s)).astype(np.float32) for s in keep}
    total, total_boot = sum(count.values()), sum(count_boot.values())
    point, boots = 0.0, np.zeros(boot.W.shape[0], np.float32)
    for s in keep:
        pc, bc = boot.mean(x, contested & (strata == s))
        pu, bu = boot.mean(x, ~contested & (strata == s))
        point += count[s] / total * (pc - pu)
        boots = boots + count_boot[s] / total_boot * (bc - bu)
    return point, boots, int(np.isin(strata, keep).sum())


def h2_table(d: pd.DataFrame, run: str, n_boot: int) -> list[dict]:
    boot = Bootstrap(d.prompt_hash, n_boot=n_boot)
    contested = (d.margin_group == "contested").to_numpy()
    x = d.position_flip.to_numpy()
    pc, pu = boot.mean(x, contested), boot.mean(x, ~contested)
    rows = [
        {"run": run, "measure": "position_flip_contested", "n": int(contested.sum()), **summary(*pc)},
        {"run": run, "measure": "position_flip_unanimous", "n": int((~contested).sum()), **summary(*pu)},
        {"run": run, "measure": "difference_contested_minus_unanimous", "n": len(d),
         **summary(pc[0] - pu[0], pc[1] - pu[1])},
    ]
    margin = d.absolute_mean_margin_raw.to_numpy()
    in_overlap = margin >= 1
    margin_bin = np.where(margin < 2, "1-2", np.where(margin < 3, "2-3", "3"))
    for label, strata in (("margin_bin", margin_bin),
                          ("margin_bin_and_annotators", np.char.add(margin_bin, "_n" + d.n_annotators.astype(str).to_numpy()))):
        sub = Bootstrap(d.prompt_hash[in_overlap], n_boot=n_boot)
        point, boots, n = _standardized_difference(sub, x[in_overlap], contested[in_overlap], strata[in_overlap])
        rows.append({"run": run, "measure": f"difference_adjusted_by_{label}", "n": n, **summary(point, boots)})
    return rows


def h3_table(d: pd.DataFrame, edits: pd.DataFrame, run: str, weights: dict, n_boot: int) -> list[dict]:
    w = d.margin_group.map(weights).to_numpy()
    boot = Bootstrap(d.prompt_hash, n_boot=n_boot, unit_weights=w)
    rows = []
    for measure, values in (("agreement_single", d.agreement_single), ("agreement_balanced", d.agreement_balanced),
                            ("position_consistency", 1 - d.position_flip), ("prefers_first", d.prefers_first)):
        rows.append({"run": run, "measure": measure, "n": len(d), **summary(*boot.mean(values))})
    if len(edits):
        e = edits.merge(d[["item_id", "margin_group", "prompt_hash"]], on="item_id")
        eboot = Bootstrap(e.prompt_hash, n_boot=n_boot, unit_weights=e.margin_group.map(weights).to_numpy())
        for edit_id in sorted(e.edit_id.unique()):
            mask = (e.edit_id == edit_id).to_numpy()
            for col in ("edit_flip", "shift_to_edited"):
                rows.append({"run": run, "measure": f"{col}_{edit_id}", "n": int(mask.sum()),
                             **summary(*eboot.mean(e[col], mask))})
    return rows


def h4_table(d: pd.DataFrame, edits: pd.DataFrame, run: str, source: str, n_boot: int) -> list[dict]:
    boot = Bootstrap(d.cluster, n_boot=n_boot)
    rows = []
    for measure, values in (("agreement_single", d.agreement_single), ("agreement_balanced", d.agreement_balanced),
                            ("position_consistency", 1 - d.position_flip)):
        rows.append({"run": run, "source": source, "measure": measure, "n": len(d), **summary(*boot.mean(values))})
    if source == "mt_bench":
        for turn in (1, 2):
            mask = (d.turn == turn).to_numpy()
            rows.append({"run": run, "source": source, "measure": f"agreement_single_turn{turn}", "n": int(mask.sum()),
                         **summary(*boot.mean(d.agreement_single, mask))})
    if len(edits):
        e = edits.merge(d[["item_id", "cluster"]], on="item_id")
        eboot = Bootstrap(e.cluster, n_boot=n_boot)
        for edit_id in sorted(e.edit_id.unique()):
            mask = (e.edit_id == edit_id).to_numpy()
            rows.append({"run": run, "source": source, "measure": f"edit_flip_{edit_id}", "n": int(mask.sum()),
                         **summary(*eboot.mean(e.edit_flip, mask))})
    return rows


# ---------- loading ----------

def run_label(meta: dict) -> str:
    model = (meta.get("model") or meta["backend"]).split("/")[-1]
    adapter = Path(meta["adapter"]).name if meta.get("adapter") else "base"
    return f"{model} / {meta['prompt_id']} / {adapter}"


def mt_bench_reference(mt: pd.DataFrame) -> pd.DataFrame:
    """Per comparison: share of non-tie human votes for each model; all-tie comparisons drop out."""
    votes = mt[mt.winner != "tie"].groupby("comparison_id").winner.agg(
        share_2=lambda w: (w == "model_2").mean(), n="size")
    votes["share_1"] = 1 - votes.share_2
    info = mt.drop_duplicates("comparison_id").set_index("comparison_id")[["question_id", "turn"]]
    return votes.join(info)


def analyze(run_paths: list[Path], *, pairs: pd.DataFrame, manifest: pd.DataFrame, ranking: pd.DataFrame,
            mt: pd.DataFrame | None, n_boot: int, dollars_per_hour: float) -> dict[str, pd.DataFrame]:
    info = pairs.merge(manifest[["pair_id", "prompt_hash", "partition"]], on="pair_id").rename(columns={"pair_id": "item_id"})
    info = info[["item_id", "prompt_hash", "margin_group", "n_annotators", "absolute_mean_margin_raw",
                 "vote_share_response_1", "vote_share_response_2"]]
    pool = ranking.groupby("margin_group").size()
    tables = {"h1": [], "h2": [], "h3": [], "h4": [], "cost": []}
    for path in run_paths:
        meta = json.loads(path.with_suffix(".json").read_text())
        judgments = pd.read_parquet(path)
        run, source = run_label(meta), meta["source"]
        d, edits = pair_level(judgments), edit_level(judgments)
        if source == "mt_bench":
            if mt is None:
                raise ValueError("MT-Bench run given but the MT-Bench judgments file is missing")
            ref = mt_bench_reference(mt)
            d = d[d.item_id.isin(ref.index)]
            d = add_agreement(d, ref.share_1, ref.share_2).join(ref[["question_id", "turn"]], on="item_id")
            d["cluster"] = d.question_id
        else:
            d = d.merge(info, on="item_id")
            d = add_agreement(d, info.set_index("item_id").vote_share_response_1,
                              info.set_index("item_id").vote_share_response_2)
            d["cluster"] = d.prompt_hash
        if source == "h1h3":
            sample = d.margin_group.value_counts()
            weights = {g: pool[g] / sample[g] for g in sample.index}
            tables["h1"] += h1_table(d, run, n_boot)
            tables["h2"] += h2_table(d, run, n_boot)
            tables["h3"] += h3_table(d, edits, run, weights, n_boot)
        elif source in ("audit", "mt_bench"):
            tables["h4"] += h4_table(d, edits, run, source, n_boot)
        n = meta["judgments"]
        tables["cost"].append({
            "run": run, "source": source, "judgments": n, "skipped": meta["skipped"],
            "seconds_per_judgment": round(meta["wall_seconds"] / max(n, 1), 4),
            "dollars_per_1000_judgments": round(meta["wall_seconds"] / max(n, 1) * 1000 / 3600 * dollars_per_hour, 4),
            "dollars_per_hour_assumed": dollars_per_hour, "device": meta.get("device"),
        })
    return {k: pd.DataFrame(v) for k, v in tables.items()}


def _markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No runs._\n"
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="Judgment parquet files from judge_audit.judge")
    p.add_argument("--pairs", default="data/processed/helpsteer_pairs.parquet")
    p.add_argument("--manifest", default="data/splits/helpsteer_manifest.csv")
    p.add_argument("--ranking", default="data/splits/h1h3_ranking.csv")
    p.add_argument("--mt-bench", default="data/processed/mt_bench_judgments.parquet")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--dollars-per-hour", type=float, default=0.75, help="Machine price for cost per 1,000 judgments")
    p.add_argument("--out", required=True, help="Output directory for CSV tables and summary.md")
    a = p.parse_args()
    mt_path = Path(a.mt_bench)
    tables = analyze([Path(r) for r in a.runs], pairs=pd.read_parquet(a.pairs),
                     manifest=pd.read_csv(a.manifest, keep_default_na=False),
                     ranking=pd.read_csv(a.ranking, dtype={"margin_group": str}),
                     mt=pd.read_parquet(mt_path) if mt_path.exists() else None,
                     n_boot=a.n_boot, dollars_per_hour=a.dollars_per_hour)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    titles = {"h1": "H1: position flips by margin (unanimous pairs)",
              "h2": "H2: contested vs unanimous position flips",
              "h3": "H3: agreement, consistency and edit sensitivity per judge (pool-weighted)",
              "h4": "H4: mitigations on audit pairs and MT-Bench",
              "cost": "Cost"}
    text = ["# Analysis summary (prototype)\n", "95% intervals from a cluster bootstrap. See DECISIONS.md for definitions.\n"]
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
        text += [f"\n## {titles[name]}\n\n", _markdown(frame)]
    (out / "summary.md").write_text("".join(text))
    print(f"Wrote {', '.join(f'{k}.csv' for k in tables)} and summary.md to {out}")


if __name__ == "__main__":
    main()
