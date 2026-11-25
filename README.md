# Auditing LLM-as-a-Judge

A CPU-first scaffold for the COMS 4995 project. Read [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) before running an experiment.

**The data is prepared.** To use it, read [DATA.md](DATA.md): what each file and partition means, which data each hypothesis uses, how to take the H1–H3 sample, and how to rebuild and verify the files. Choices still open for the experiments are in [OPEN_DECISIONS.md](OPEN_DECISIONS.md).

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

## Included

- Schema inspection of Hugging Face datasets and bounded sample records.
- Deterministic pair hashes and per-pair raw-vote statistics.
- Margin groups, audit eligibility and parsed multi-turn prompts.
- MT-Bench human judgments in a fixed model order.
- Near-duplicate prompt detection across partitions and against MT-Bench.
- Prompt-grouped, immutable split manifests with source validation held out and reviewed near-duplicates excluded.
- Seeded H1–H3 sample ranking.
- Pure label mapping between A/B order and original response identity.

## Not included yet (intentional)

Full vLLM harness, formatted/no-op/content-removal transforms, matched H2 analysis, hypothesis tests, LoRA training and MT-Bench transfer. See the decision log in the plan and [OPEN_DECISIONS.md](OPEN_DECISIONS.md).
