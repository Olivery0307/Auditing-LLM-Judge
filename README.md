# Auditing LLM-as-a-Judge

A CPU-first scaffold for the COMS 4995 project. Read [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) before running an experiment.

## Quick start

```bash
cd '/Users/oliveryang/Desktop/github repo/auditing_llm_judge'
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python -m judge_audit.inspect_dataset --dataset nvidia/HelpSteer2 --data-dir preference --split train --limit 2 --out reports/helpsteer_schema.json
```

**Review schema JSON before preprocessing.** The [publisher's dataset card](https://huggingface.co/datasets/nvidia/HelpSteer2/blob/main/README.md) shows `all_preferences_unprocessed` as a list of `{strength, justification}` records. Strengths range from -3 to 3; zero is an individual tie. Positive means response 2 is preferred. Confirm these details against the downloaded snapshot and pin its commit SHA before processing. The CLI rejects records that do not match this exact structure.

```bash
python -m judge_audit.preprocess \
  --dataset nvidia/HelpSteer2 --data-dir preference --split train \
  --revision YOUR_40_CHARACTER_COMMIT_SHA \
  --votes-field all_preferences_unprocessed \
  --prompt-field prompt --response1-field response_1 --response2-field response_2 \
  --out data/processed/helpsteer_pairs.parquet \
  --report reports/helpsteer_audit.json \
  --counts reports/stratum_counts.csv
```

The preference download's Hugging Face `train` split contains both the publisher's internal `train` and `val` rows, identified by the `split` column. After reviewing counts and duplicates, freeze partitions with a pinned commit SHA:

```bash
python -m judge_audit.splits \
  --pairs data/processed/helpsteer_pairs.parquet \
  --source-revision YOUR_40_CHARACTER_COMMIT_SHA \
  --out data/splits/helpsteer_manifest.csv
```

The split command holds publisher validation rows in `audit` and hashes normalized prompts to assign the remaining rows to `train` or `development`. It rejects duplicate normalized pairs and prompts crossing the publisher's split. It refuses to overwrite a manifest. Semantic near-duplicate checks and MT-Bench overlap checks still require a separate review before experiments. Output files are ignored by Git.

## Included

- Schema inspection of Hugging Face datasets and bounded sample records.
- Deterministic pair hashes and per-pair raw-vote statistics, including ties.
- Descriptive category counts, vote count distribution, and invalid row reporting.
- Pure label mapping between A/B order and original response identity.
- Exact-set leakage check helpers and unit tests.
- Prompt-grouped, immutable split manifests with source validation held out.

## Not included yet (intentional)

Full vLLM harness, formatted/no-op/content-removal transforms, semantic near-duplicate detection, MT-Bench overlap checks, matched H2 analysis, hypothesis tests, LoRA training and MT-Bench transfer. See the TODO/decision log in the plan. These require agreed evaluation definitions and actual dataset/schema validation.
