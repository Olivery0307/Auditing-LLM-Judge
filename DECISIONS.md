# Decisions

Every experiment has a working prototype. Where a choice was needed, the prototype picked a default so the pipeline runs end to end. This file lists those defaults so they can be confirmed or changed. Copy confirmed ones to the decision log in `IMPLEMENTATION_PLAN.md`.

**Status:** *Settled* = agreed by the team. *Default* = chosen by the prototype; confirm or change. *Untested* = code exists but has not been run.

Data preparation, partitions and file formats are in [DATA.md](DATA.md). How to run each step is in the [README](README.md#running-the-experiments).

## Data (settled)

| Decision | Choice | Status |
|---|---|---|
| Partition use | H1–H3 sample from `train` + `audit`; LoRA trains on `train`; H4 tests on `audit` and MT-Bench; `development` for tuning only | Settled (team, 2026-11-25) |
| Eligibility | At least 3 annotators, raw votes, distinct responses | Settled |
| Margin groups | Unanimous margin rounded to 1, 2 or 3, plus contested | Settled |
| Near-duplicates | 31 removed after manual review | Settled |
| Sample order | Ranked within each group with seed 42 | Settled |

## Sample size and compute

| # | Decision | Prototype default | Why / what to check | Where |
|---|---|---|---|---|
| 1 | H1–H3 pairs per group (*k*) | **400** per group = 1,600 pairs | Proposal's "up to 400 per stratum". A larger *k* always contains a smaller one | `judge --k` |
| 2 | Pairs that receive edits | Ranks **1–100** per group (400 pairs) for H1–H3; all 412 eligible `audit` pairs for H4 | Edits multiply cost by about 4.8 | `perturb --edit-k`, `--source audit` |
| 3 | Judgments per model | H1–H3: 9,210; `audit` with edits: 7,034; MT-Bench: 3,628 | From the dry runs with these defaults | — |
| 4 | Model roster and precision | Qwen2.5-1.5B, 7B, 14B-Instruct, all bf16 | Plan asks for 4 sizes or a second family before claiming a size trend. 14B in bf16 needs the A100 (quota request pending) | `judge --model` |
| 5 | Inference backend | `transformers` (verified on an L4: 0.11 s per judgment for 1.5B, 9.5 GB peak memory before scoring was limited to the last position) | `vllm` is faster but **untested**; check it on a GPU before relying on it. For the 14B on a 40 GB A100, start with `--batch-size 2`. The last-position change is **untested** on a GPU | `backends.py` |
| 6 | Cost assumption | $0.75 per hour (L4, `g2-standard-4`) | Pass the real price with `analysis --dollars-per-hour` | `analysis.py` |

## Judging

| # | Decision | Prototype default | Why / what to check | Where |
|---|---|---|---|---|
| 7 | Output method | Probability of "A" vs "B" as the first answer token; verdict is the more likely letter | No parsing failures and gives a confidence. The model may answer differently when generating freely | `backends.py` |
| 8 | Ties | Judge cannot tie | HelpSteer2 annotators could not tie either | `judge.py` |
| 9 | Prompt wording | `baseline_v1` and `rubric_v1` (rubric: HelpSteer attributes in order; tells the judge to ignore order, length and formatting) | Any wording change gets a new prompt ID. Pick or refine on `development` only | `prompts.py` |
| 10 | Multi-turn prompts | Show the full conversation as a transcript; the judged responses answer the last user turn | 29% of prompts are multi-turn. Report them separately as a check | `prompts.py` |
| 11 | Long inputs | No truncation; skip and log inputs over 8,192 tokens | Pilot maximum was 2,370 tokens; longest pair is about 14,400 characters | `judge --max-input-tokens` |
| 12 | Repeat no-op | Not run | Scoring is deterministic, so an exact repeat measures nothing; `neutral_whitespace` is the no-op instead | — |
| 13 | Seeds | Not used for judging | `evaluation_seeds` in the example config only matter for sampling-based judging | — |

## Edits

All edits change one response at a time, regardless of which one humans preferred, and every edit starts as `unreviewed`.

| # | Edit | Prototype rule | What to check |
|---|---|---|---|
| 14 | `neutral_whitespace` (no-op) | Strip trailing spaces per line, end with one newline | Invisible to readers but changes tokens |
| 15 | `format_bullets` | Put "- " before each paragraph, keeping line breaks; only with 2+ paragraphs and no list, heading or code | Applies to 38% of responses |
| 16 | `strip_markdown` | Remove bold, italic, heading and list markers; skip code blocks | Applies to 43% of responses |
| 17 | `lengthen_filler` | Add "Sure! Here is my response to your request." and "I hope this helps! Let me know if you have any other questions." | Only 2 sentences, so a mild lengthening. Longer filler risks sounding like content |
| 18 | `content_removal` (positive control) | Keep the first half of the paragraphs (or sentences) | Removes from under 1% to 99% of the text, median 48%. Check that it really harms useful answers |
| 19 | Reviewer validation | Not done | Plan: two reviewers check a prespecified subset; drop unreliable edit types |

## Analysis

| # | Decision | Prototype default | Why / what to check | Where |
|---|---|---|---|---|
| 20 | Agreement with humans | **Soft agreement**: share of annotators who chose the judge's response | Works for contested pairs too. Exact-match agreement is not implemented | `analysis.py` |
| 21 | Standard vs balanced judge | Single-order (each order counts) is the baseline; order-balanced (average both orders) is the mitigation | — | `analysis.py` |
| 22 | H1 test | Position-flip rate per margin group; difference margin 1 minus margin 3 | No trend test across 1, 2, 3 yet | `h1_table` |
| 23 | H2 test | Raw contested minus unanimous flip rate, plus differences standardized within margin bins, and within margin bins × annotator counts | Only contested pairs with margin ≥ 1 overlap unanimous pairs; with *k* = 400 about 80 of them are sampled, so the adjusted CIs are wide. Take more from the ranking if needed. The proposal mentions regression; not implemented | `h2_table` |
| 24 | H2 outcome | Position flips only | Add edit flips if "less stable" should include them | `h2_table` |
| 25 | H3 comparison | Separate CIs per model; overall numbers reweight groups to the full pool | Paired model-vs-model differences are not computed yet | `h3_table` |
| 26 | Confidence intervals | 1,000 cluster-bootstrap draws; clusters = prompts (HelpSteer2) or questions (MT-Bench) | Pairs and prompts nearly coincide on HelpSteer2 | `Bootstrap` |
| 27 | Multiple testing | None | Decide before final results | — |

## LoRA (H4)

| # | Decision | Prototype default | Why / what to check | Where |
|---|---|---|---|---|
| 28 | Training labels | Response favored by both majority and mean vote; 230 ambiguous pairs dropped | Soft labels would need a different loss | `lora_data.py` |
| 29 | Pairs with 1–2 annotators | Not used | 612 more `train` pairs are available if wanted | `lora_data.py` |
| 30 | Training size | 3,000 pairs × 2 orders = 6,000 examples per condition | Chosen to fit the budget; raise if training is fast | `lora_data --max-pairs` |
| 31 | Augmentation | About half the examples get one edit (`format_bullets`, `strip_markdown` or `lengthen_filler`) on a randomly chosen side; label unchanged | Same pairs, orders and labels in both conditions. No content removal, which could change the right answer | `lora_data.py` |
| 32 | Loss and settings | Loss on the answer token only; rank 16, alpha 32, dropout 0.05, learning rate 1e-4, 1 epoch, 16 examples per step, skip inputs over 4,096 tokens | **Untested**; needs a GPU | `train_lora.py` |
| 33 | Seeds and selection | 3 seeds per condition; keep the adapter with the best development accuracy | Report seed-to-seed spread | `train_lora --seed` |
| 34 | LoRA scoring | `transformers` backend only, on `audit` and MT-Bench | vLLM LoRA support not wired up | `judge --adapter` |

## MT-Bench (H4)

| # | Decision | Prototype default | Why / what to check | Where |
|---|---|---|---|---|
| 35 | Human reference | Share of non-tie votes per comparison; all-tie comparisons drop out | 780 of 3,355 judgments are ties | `mt_bench_reference` |
| 36 | Turns | Both turns analyzed, also reported separately | Proposal treats turn 2 as exploratory | `h4_table` |
| 37 | Empty answers | Kept (18 judgments with an empty `llama-13b` answer) | Easy cases for a judge; drop them as a check | — |
| 38 | Minimum judgments | None (853 comparisons have one judgment) | Consider requiring 2+ as a check | — |
