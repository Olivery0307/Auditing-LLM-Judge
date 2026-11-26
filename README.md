# Auditing LLM-as-a-Judge

A CPU-first scaffold for the COMS 4995 project. Read [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) before running an experiment.

**The data is prepared.** To use it, read [DATA.md](DATA.md): what each file and partition means, which data each hypothesis uses, how to take the H1–H3 sample, and how to rebuild and verify the files. Every experiment has a working prototype; the defaults they use, and what still needs confirming, are in [DECISIONS.md](DECISIONS.md).

## Quick start

From the repo root:

```bash
uv sync
uv run pytest -q
uv run python -m judge_audit.inspect_dataset --dataset nvidia/HelpSteer2 --data-dir preference --split train \
  --revision 990b2711a36180dd19d9c94b8627844866f8982a --limit 2 --out reports/helpsteer_schema.json
```

`uv sync` uses Python 3.12 from `.python-version`, installs the default development group, and creates or updates `.venv` from the committed `uv.lock`.

## Data pipeline

The full commands with pinned revisions and expected hashes are in [DATA.md](DATA.md#rebuilding-and-checking-the-files). In order:

| Step | Command | Output |
|---|---|---|
| 1 | `judge_audit.inspect_dataset` | Schema and resolved revision; review before preprocessing |
| 2 | `judge_audit.preprocess` | One row per HelpSteer2 pair with raw-vote statistics, margin group, audit eligibility and parsed turns |
| 3 | `judge_audit.mt_bench` | One row per MT-Bench human judgment, winner mapped to a fixed model order |
| 4 | `judge_audit.near_duplicates` | Near-duplicate prompts for manual review; confirmed copies go in `configs/near_duplicate_exclusions.csv` |
| 5 | `judge_audit.splits` | Frozen train / development / audit manifest; refuses to overwrite |
| 6 | `judge_audit.ranking` | Seeded rank of every H1–H3-eligible pair within its margin group; refuses to overwrite |

**Vote schema.** `all_preferences_unprocessed` is a list of `{strength, justification}` records with strengths -3 to 3. Positive means response 2 is preferred; this was checked against the annotators' written statements in the pinned snapshot. Annotators could not choose a tie, so HelpSteer2 has no zero votes. The CLI rejects records that do not match this structure.

**Splits.** The preference download's Hugging Face `train` split contains both the publisher's internal `train` and `val` rows, identified by the `split` column. `val` rows always go to `audit`. Other rows go to `train` or `development` by a seeded hash of the whitespace-normalized prompt, so pairs sharing a prompt stay together. The command rejects duplicate normalized pairs, prompts crossing the publisher's split, and a pairs file from a different revision. Output files are ignored by Git; frozen hashes are in `configs/`.

## Running the experiments

Model steps need a GPU and the optional model libraries (`uv sync --group model`). Use `--backend fake` for a dry run without a GPU. Run from the repo root:

```bash
# 1. Edits (CPU): H1-H3 sample ranks 1-100 per group, and all eligible audit pairs for H4
uv run python -m judge_audit.perturb --source h1h3 --edit-k 100 --out data/perturbations/h1h3_edits.parquet
uv run python -m judge_audit.perturb --source audit --out data/perturbations/audit_edits.parquet

# 2. Pilot on development pairs
uv run --group model python -m judge_audit.judge --source development --limit 50 \
  --backend transformers --model Qwen/Qwen2.5-1.5B-Instruct --out results/pilot/qwen1p5b_dev50.parquet

# 3. H1-H3: one run per model size, same pairs
uv run --group model python -m judge_audit.judge --source h1h3 --k 400 --edits data/perturbations/h1h3_edits.parquet \
  --backend transformers --model Qwen/Qwen2.5-7B-Instruct --out results/h1h3/qwen7b_baseline.parquet

# 4. H4 inference-time mitigations: baseline and rubric prompts on audit pairs and MT-Bench
uv run --group model python -m judge_audit.judge --source audit --edits data/perturbations/audit_edits.parquet \
  --prompt rubric_v1 --backend transformers --model Qwen/Qwen2.5-1.5B-Instruct --out results/h4/qwen1p5b_rubric_audit.parquet
uv run --group model python -m judge_audit.judge --source mt_bench \
  --backend transformers --model Qwen/Qwen2.5-1.5B-Instruct --out results/h4/qwen1p5b_baseline_mt.parquet

# 5. H4 LoRA: build data (CPU), train each condition and seed, then score on audit and MT-Bench only
uv run python -m judge_audit.lora_data
uv run --group model python -m judge_audit.train_lora --train data/lora/train_augmented.jsonl \
  --dev data/lora/dev.jsonl --out results/lora/augmented_seed1 --seed 1
uv run --group model python -m judge_audit.judge --source audit --edits data/perturbations/audit_edits.parquet \
  --adapter results/lora/augmented_seed1/adapter --backend transformers --model Qwen/Qwen2.5-1.5B-Instruct \
  --out results/h4/lora_augmented_s1_audit.parquet

# 6. Tables with confidence intervals (CPU)
uv run python -m judge_audit.analysis --runs results/h1h3/*.parquet results/h4/*.parquet --out reports/analysis
```

Every run writes a parquet of judgments and a `.json` with the model revision, prompt, device, timing and library versions. The order-balanced mitigation needs no extra run; the analysis computes it from both orders.

**Google Cloud.** The course organization blocks public IP addresses on machines. Create the machine with `--no-address`, connect with `gcloud compute ssh --tunnel-through-iap`, and add a Cloud Router with Cloud NAT in the machine's region so it can download packages and models. A `g2-standard-4` with one L4 and Google's `common-cu129-ubuntu-2204-nvidia-580` image worked for the pilot. Delete the machine (and NAT, if not needed) when done.

## Included

- Schema inspection of Hugging Face datasets and bounded sample records.
- Deterministic pair hashes and per-pair raw-vote statistics.
- Margin groups, audit eligibility and parsed multi-turn prompts.
- MT-Bench human judgments in a fixed model order.
- Near-duplicate prompt detection across partitions and against MT-Bench.
- Prompt-grouped, immutable split manifests with source validation held out and reviewed near-duplicates excluded.
- Seeded H1–H3 sample ranking.
- Pure label mapping between A/B order and original response identity.
- Prototypes: presentation edits, judge harness (transformers backend verified on an L4), H1–H4 analysis with cluster-bootstrap intervals, LoRA data builder and training script.

## Not verified yet

The vLLM backend and LoRA training have not been run. Edits have not been reviewed by two people. See [DECISIONS.md](DECISIONS.md) for every default that still needs confirming.
