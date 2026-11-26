# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CPU-first scaffold for a COMS 4995 research project auditing LLM-as-a-Judge behavior on HelpSteer2-Preference (position instability, human agreement, formatting bias, model size, LoRA mitigation). `IMPLEMENTATION_PLAN.md` is the source of truth for hypotheses (H1–H4), methodology and the decision log. `DATA.md` documents the prepared data, and `DECISIONS.md` lists every default the experiment prototypes use.

Data preparation is done and the split is frozen. Each experiment has a prototype that a teammate will refine: edits (`perturb`), judging (`items`, `prompts`, `backends`, `judge`), analysis (`analysis`) and LoRA (`lora_data`, `train_lora`). When a prototype needs a new default, choose a sensible one and record it in `DECISIONS.md` with its status (Default / Untested); don't leave decisions only in code.

Model work runs on a GCP GPU, never on the user's laptop: keep local commands light (no model loading or large installs). Use `--backend fake` for local dry runs. The vLLM backend and `train_lora` have not been run yet.

## Commands

```bash
uv sync                                   # Python 3.12 via .python-version; uv.lock is committed
uv run pytest -q
uv run pytest -q tests/test_core.py::test_swapped_label_mapping   # single test
uv run ruff check .
```

Ruff has no project config, and the scaffold has 5 pre-existing findings under ruff 0.17 defaults. Don't apply its `TRY004` suggestion in `parse_raw_votes`: the tests require `ValueError` there. No test touches the network; the preprocessing CLIs download from Hugging Face.

## Data pipeline

Six CLIs prepare the data, in order: `inspect_dataset`, `preprocess`, `mt_bench`, `near_duplicates`, `splits`, `ranking`. The full commands, pinned revisions and expected hashes are in `DATA.md`. `core.py` holds the pure functions they share. Experiment commands are in the README's "Running the experiments" section. Model libraries (torch, transformers, peft) are in the optional `model` dependency group: `uv sync --group model`.

- `preprocess` keeps every valid pair and adds `margin_group`, `audit_eligible`, parsed `turns` and `dataset_revision`. Rows are never deleted for eligibility.
- `near_duplicates` writes a review CSV with dataset text to `reports/`. Reviewed copies are listed by pair ID in `configs/near_duplicate_exclusions.csv`, which `splits` applies.
- `splits` and `ranking` refuse to overwrite, and `ranking` checks the frozen manifest and pairs hashes. Frozen metadata is committed in `configs/*.frozen.json`.

All outputs (`data/`, `reports/`, `results/`, `experiments/`) are git-ignored. `configs/experiment.example.yaml` documents intended settings only: no code reads it.

## Semantics that are easy to get wrong

- **Vote sign:** individual strengths are integers in -3..3; positive favors response 2. Annotators could not choose a tie, so HelpSteer2 has **no zero votes**. Zero handling in the code exists only for the generic numeric format.
- **Raw vs. published:** `*_raw` fields come from `all_preferences_unprocessed`. The published `preference_strength` is a rounded aggregate; never substitute it for raw votes or margins.
- **Margin groups:** unanimous margins are rounded half up to `1`, `2` or `3` (cutoffs 1.5 and 2.5); disagreeing pairs are `contested`. Unanimous margins are always at least 1, so contested pairs below 1 have no unanimous counterpart.
- **`pair_id` is order-independent** (responses are sorted before hashing), so it does not identify which text is response 1. Keep the row's `response_1`/`response_2` numbering, and map judge A/B output back through `original_response_id(label, order)`.
- **Partitions:** `train` means "LoRA may train on it", not "training only". H1–H3 sample from `train` + `audit` (untuned judges); H4 tests only on `audit` and MT-Bench; `development` is for tuning only.

## Invariants to preserve

- **Fail closed:** unexpected vote shapes, malformed turn markers, unpinned or mismatched revisions, duplicate pairs, prompts crossing the publisher's split, excluding an `audit` pair, and existing manifests or rankings all raise. Don't loosen these checks to get data through. (`preprocess` records unparseable rows in its report unless `--strict` is set.)
- The split manifest and ranking are frozen. Changing them means deleting the files deliberately with the team's agreement. Any derived variant (perturbation, augmentation) inherits its original pair's partition.
- MT-Bench stays entirely out of training, prompt tuning, edit selection and adapter selection. `judge` refuses a LoRA adapter on the H1–H3 sample, and `lora_data` reads only the `train` and `development` partitions.
- Do not commit raw dataset text or credentials. Review CSVs and samples in `reports/` contain dataset text.
