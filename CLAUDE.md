# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CPU-first scaffold for a COMS 4995 research project auditing LLM-as-a-Judge behavior on HelpSteer2-Preference (position instability, human agreement, formatting bias, model size, LoRA mitigation). `IMPLEMENTATION_PLAN.md` is the source of truth for hypotheses (H1–H4), methodology, and the decision log. Read it before designing any experiment.

Only the data-audit stage exists. The judge harness, perturbations, H2 matching, hypothesis tests, LoRA training, and MT-Bench transfer are **intentionally absent** until the open decisions in the plan are settled. Don't add placeholder implementations, and don't resolve a `TODO` decision (tie policy, margin bins, minimum annotators, model roster, CI clusters, etc.) on your own by picking a default in code; ask.

## Commands

```bash
uv sync                                   # Python 3.12 via .python-version; dev group included
uv run pytest -q
uv run pytest -q tests/test_core.py::test_swapped_label_mapping   # single test
uv run ruff check .
```

Ruff has no project config, and the initial scaffold already has 5 findings under ruff 0.17 defaults. Don't apply its `TRY004` suggestion in `parse_raw_votes`: the tests require `ValueError` there.

`uv.lock` is not committed yet; the README asks for it to be committed after the first successful sync. No test touches the network; `inspect_dataset` and `preprocess` download from Hugging Face.

Paths in the README and plan (`/Users/oliveryang/...`) belong to a teammate's machine; run from the repo root instead.

## Data pipeline

Three CLIs run in order, each with a manual review gate before the next:

1. `python -m judge_audit.inspect_dataset` writes a schema JSON, including `resolved_revision`. Review it before preprocessing.
2. `python -m judge_audit.preprocess` writes `pairs.parquet` (one row per original pair), an audit report, and optional stratum counts. Requires that 40-character SHA as `--revision`.
3. `python -m judge_audit.splits` reads the parquet and freezes a manifest CSV plus a sidecar `.json` with hashes. `--source-revision` must be the same SHA; it is recorded but not checked against the parquet.

Full invocations are in `README.md`. `core.py` holds the pure functions shared by the CLIs. `configs/experiment.example.yaml` documents intended settings only: no code reads it. All outputs (`data/`, `reports/`, `results/`, `experiments/`) are git-ignored.

## Semantics that are easy to get wrong

- **Vote sign:** individual strengths are integers in -3..3; positive favors response 2, zero is an individual tie. Verified against the dataset card, not yet against a pinned snapshot.
- **Raw vs. published:** `*_raw` fields are computed from `all_preferences_unprocessed`. The published `preference_strength` is an aggregate; never substitute it for raw votes or margins.
- **Unanimous** means every vote has the same nonzero sign; **contested** means both signs are present. They are not complements (e.g. `[0, 1, 0]` is neither). A single-annotator nonzero vote counts as unanimous, and `--min-annotators` defaults to 1.
- **`pair_id` is order-independent** (responses are sorted before hashing), so it does not identify which text is response 1. Keep the row's `response_1`/`response_2` numbering alongside it, and map judge A/B output back through `original_response_id(label, order)`.
- **Splits:** the Hugging Face `train` split contains the publisher's internal `train` and `val` rows, distinguished by the `split` column (`source_split` in the parquet). `val` rows always become `audit`; the rest are assigned to `train` or `development` by a seeded hash of the whitespace-normalized prompt.

## Invariants to preserve

- **Fail closed:** unexpected vote shapes, unpinned revisions, duplicate pairs, prompts crossing the publisher's split, and existing manifests all raise. Don't loosen these checks to get data through. (`preprocess` records unparseable rows in its report unless `--strict` is set.)
- Split manifests are immutable. Any derived variant (perturbation, augmentation) inherits its original pair's partition.
- MT-Bench stays entirely out of training, prompt tuning, edit selection, and adapter selection.
- `assert_disjoint` checks exact identifiers only. Semantic near-duplicate and MT-Bench overlap checks remain open, so don't describe the splits as fully leakage-free.
- Do not commit raw dataset text or credentials.
