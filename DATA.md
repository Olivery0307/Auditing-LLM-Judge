# Data guide

How to use the prepared HelpSteer2 and MT-Bench data. Data files are not in Git; rebuild them with the commands at the end and check the hashes. Defaults used by the experiment prototypes are in [DECISIONS.md](DECISIONS.md).

| Source | Revision | License |
|---|---|---|
| `nvidia/HelpSteer2`, `preference` folder | `990b2711a36180dd19d9c94b8627844866f8982a` | CC-BY-4.0 |
| `lmsys/mt_bench_human_judgments`, `human` split | `f7d2896d2cc5d80f8b55c2bbc722613555233c25` | CC-BY-4.0 |

## Files

| File | One row per | Rows | Use it for |
|---|---|---|---|
| `data/processed/helpsteer_pairs.parquet` | HelpSteer2 response pair | 9,125 | Text, votes, margin group, eligibility |
| `data/splits/helpsteer_manifest.csv` | HelpSteer2 response pair | 9,125 | Which partition each pair is in (frozen) |
| `data/splits/h1h3_ranking.csv` | Eligible pair | 7,559 | Taking the H1–H3 sample |
| `data/processed/mt_bench_judgments.parquet` | Human judgment | 3,355 | H4 transfer test |

Join them on `pair_id`. Reports with counts are in `reports/`; frozen hashes are in `configs/`.

## What train, development, audit and excluded mean

Every HelpSteer2 pair is in exactly one partition. Pairs sharing a prompt are always in the same partition.

| Partition | Pairs | What it is | Use it for | Never use it for |
|---|---|---|---|---|
| `train` | 7,760 | Most of the publisher's `train` rows | LoRA training; H1–H3 sample | Testing a LoRA judge (H4) |
| `development` | 886 | About 10% of the publisher's `train` rows, set aside as a practice set | The 50-pair pilot, choosing the judge prompt, testing edits, choosing LoRA settings, debugging | Reporting any H1–H4 result |
| `audit` | 448 | The publisher's `val` rows | H4 test on unseen pairs; also part of the H1–H3 sample | LoRA training; tuning anything |
| `excluded` | 31 | Near-duplicates of a pair in another partition, removed after manual review | Nothing | Anything |

**Why `train` pairs can be in the H1–H3 sample.** "Train" means "LoRA may train on it". H1–H3 test the original Qwen models, which never trained on HelpSteer2, so these pairs are new to them. Only H4 tests LoRA models, so only H4 must stay on `audit` and MT-Bench.

**Why development is separate.** Choosing prompts or edits on the same pairs you report would make results look better than they are. Do all trial and error on `development`.

## Which data each experiment uses

| Experiment | Data |
|---|---|
| Pilot, prompt and edit choices, LoRA settings | `development` only |
| H1: flips by margin | Sample from `h1h3_ranking.csv`, groups `1`, `2`, `3` |
| H2: contested vs unanimous | Same sample, all four groups |
| H3: model size | Same sample, all four groups; every model judges the same pairs |
| H4: LoRA training | `train` partition |
| H4: unseen HelpSteer2 pairs | `audit` partition (412 eligible pairs; report overall, not by group) |
| H4: transfer | `mt_bench_judgments.parquet` |

Never present H1–H3 results on `train` pairs as evidence for a LoRA judge. Compare LoRA judges with the original judges only on `audit` or MT-Bench.

## Margin groups and eligibility

Each annotator voted -3 to 3, with no zero: annotators could not choose a tie. **Positive means response 2 is better.** The **margin** is the absolute value of the mean vote, such as 1.33 for `[1, 1, 2]`.

| `margin_group` | Rule | Pairs in the H1–H3 pool |
|---|---|---|
| `1` | Unanimous, margin from 1 to under 1.5 | 1,727 |
| `2` | Unanimous, margin from 1.5 to under 2.5 | 1,532 |
| `3` | Unanimous, margin from 2.5 to 3 | 982 |
| `contested` | Annotators chose different responses | 3,318 |

A pair is `audit_eligible` if it has at least 3 annotators and its two responses differ. The 719 ineligible pairs (718 with 1–2 annotators, and 1 with identical responses) stay in the parquet with `audit_exclusion_reason` filled in. The H1–H3 pool is every eligible pair in `train` or `audit`.

## Taking the H1–H3 sample

Each group is shuffled with seed 42. Take ranks 1 to *k* from every group. A larger *k* always contains a smaller one, so you can start small and extend without rerunning anything. *k* = 400 gives 1,600 pairs; the team picks *k* after the pilot.

```python
import json
import pandas as pd

pairs = pd.read_parquet("data/processed/helpsteer_pairs.parquet")
ranking = pd.read_csv("data/splits/h1h3_ranking.csv", dtype={"margin_group": str})

k = 400
sample = ranking[ranking["rank"] <= k].merge(
    pairs.drop(columns="margin_group"), on="pair_id", validate="one_to_one"
)
turns = json.loads(sample.turns.iloc[0])  # [{"role": "user", "content": ...}, ...]
```

The sample has the same number of pairs per group, which is not the dataset's natural mix. Report results by group, or weight each group by its pool size divided by *k* when you need one overall number.

For other partitions:

```python
manifest = pd.read_csv("data/splits/helpsteer_manifest.csv", keep_default_na=False)
pairs = pairs.merge(manifest[["pair_id", "partition"]], on="pair_id", validate="one_to_one")
lora_train = pairs[pairs.partition == "train"]
dev = pairs[pairs.partition == "development"]
h4_unseen = pairs[(pairs.partition == "audit") & pairs.audit_eligible]
```

## Main HelpSteer2 columns

| Column | Meaning |
|---|---|
| `pair_id` | SHA-256 of prompt and both responses. The same whichever response is first, so it does **not** say which is response 1 |
| `prompt`, `response_1`, `response_2` | Original text. Vote signs refer to this order |
| `turns`, `n_prompt_turns`, `is_multi_turn` | The prompt as JSON chat turns; 2,634 prompts (29%) include earlier turns. Always ends with the user |
| `raw_votes`, `n_annotators` | Individual votes as JSON, and how many |
| `mean_signed_margin_raw`, `absolute_mean_margin_raw` | Mean vote, and its absolute value (the margin) |
| `vote_share_response_1`, `vote_share_response_2` | Share of annotators preferring each response |
| `unanimous_raw`, `contested_raw`, `margin_group` | Agreement and group |
| `audit_eligible`, `audit_exclusion_reason` | Eligibility for the H1–H3 pool |
| `published_preference_strength` | The publisher's rounded aggregate. Not used for margins |
| `source_split`, `source_row`, `dataset_revision` | Provenance |

When you swap response order for the judge, map its answer back with `judge_audit.core.original_response_id(label, order)`: `order=(1, 2)` shows response 1 as A, and `order=(2, 1)` shows response 2 as A.

## MT-Bench columns

| Column | Meaning |
|---|---|
| `comparison_id` | Question, turn and the two models; 1,814 comparisons |
| `question_id`, `turn` | 80 questions; turn 1 or 2 |
| `model_1`, `model_2` | The two models, sorted by name |
| `winner` | `model_1`, `model_2` or `tie` (780 ties) |
| `shown_first` | Which model the annotator saw first |
| `conversation_1`, `conversation_2` | JSON messages the annotator judged: 2 for turn 1, 4 for turn 2 |
| `judge` | Anonymous annotator ID |
| `has_empty_response` | 18 judgments where `llama-13b` gave an empty answer; annotators always chose the other model |
| `repeat_judgment` | 2 rows where one annotator judged the same comparison twice |

Nothing is filtered. 853 comparisons have only one judgment. Votes have no strength, so HelpSteer2 margin groups do not apply.

## Limits to keep in mind

- **The publisher removed high-disagreement pairs.** Tasks whose three closest votes spread by more than 2 were dropped (HelpSteer2-Preference paper, Section 2), so many of the strongest disagreements are missing.
- **Contested pairs with 3 annotators are always `[-1, 1, 1]` or its mirror.** All contested pairs at margin 1 or more have 4 or more annotators. See item 23 in `DECISIONS.md`.
- **Near-duplicates:** prompts with 5-word-shingle Jaccard similarity of at least 0.8 were reviewed by hand. 31 copies were excluded (`configs/near_duplicate_exclusions.csv`); 5 pairs sharing a template but asking different questions were kept. No MT-Bench prompt matched HelpSteer2.
- **Edited versions of a pair** (perturbations, augmentation) belong to the original pair's partition.

## Rebuilding and checking the files

Run from the repo root. The split and ranking steps refuse to overwrite existing files; delete them on purpose only if the team agrees to change the split.

```bash
uv sync
HS=990b2711a36180dd19d9c94b8627844866f8982a
MT=f7d2896d2cc5d80f8b55c2bbc722613555233c25
uv run python -m judge_audit.preprocess --dataset nvidia/HelpSteer2 --data-dir preference --split train \
  --revision $HS --votes-field all_preferences_unprocessed \
  --prompt-field prompt --response1-field response_1 --response2-field response_2 \
  --out data/processed/helpsteer_pairs.parquet --report reports/helpsteer_audit.json --counts reports/stratum_counts.csv
uv run python -m judge_audit.mt_bench --revision $MT \
  --out data/processed/mt_bench_judgments.parquet --report reports/mt_bench_audit.json
uv run python -m judge_audit.splits --pairs data/processed/helpsteer_pairs.parquet --source-revision $HS \
  --exclusions configs/near_duplicate_exclusions.csv --out data/splits/helpsteer_manifest.csv
uv run python -m judge_audit.ranking --pairs data/processed/helpsteer_pairs.parquet \
  --manifest data/splits/helpsteer_manifest.csv --out data/splits/h1h3_ranking.csv
shasum -a 256 data/splits/helpsteer_manifest.csv data/splits/h1h3_ranking.csv
```

The two CSVs must match these hashes, also recorded in `configs/*.frozen.json`:

| File | SHA-256 |
|---|---|
| `helpsteer_manifest.csv` | `12faa8657b99ec4709251d82deba4ea0cf2c16107187b793315be1d0e06be79a` |
| `h1h3_ranking.csv` | `bed1e8fc8a5a429abb661b97a5ad4703f40757ef315fa924ba59cfd4ee1466f6` |

Parquet files were also identical across rebuilds here (`helpsteer_pairs.parquet` `2b08a831…`, `mt_bench_judgments.parquet` `cf7d45d3…`), but their bytes can depend on library versions, so compare the CSVs first. To repeat the near-duplicate search, run `python -m judge_audit.near_duplicates`; it writes a review file with dataset text to `reports/`.
