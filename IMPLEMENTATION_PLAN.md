# LLM-as-a-Judge Audit — implementation plan

> Working plan for COMS 4995 Idea #20, based on the team's latest abstract and the instructor's comments. This is an **implementation scaffold**, not an assertion that annotation semantics, sample availability, or training choices have been verified. No GPU jobs are launched by this repository.

## Scope and principles

**Primary:** HelpSteer2-Preference; unanimous human preference margin vs. position instability (H1); contested-vs-unanimous comparison only where margins overlap (H2); presentation perturbations; model-size comparison with precision controlled (H3); inference-time and possible LoRA mitigation, with **MT-Bench held out** for the primary H4 generalization test.

**Non-negotiable:** preserve raw annotations, validate schema from a real dataset snapshot, freeze data partitions before augmentation, keep human votes separate from judge predictions, maintain a stable original response identity through A/B swaps, and cluster uncertainty intervals by original pair (or prompt where multiple pairs share a prompt).

## 0. Local setup

1. The starter contents have been placed in `/Users/oliveryang/Desktop/github repo/auditing_llm_judge`.
2. `cd '/Users/oliveryang/Desktop/github repo/auditing_llm_judge'`
3. `python3 -m venv .venv && source .venv/bin/activate`
4. `pip install -e '.[dev]'`
5. `pytest -q`
6. `python -m judge_audit.inspect_dataset --dataset nvidia/HelpSteer2 --data-dir preference --split train --limit 2 --out reports/helpsteer_schema.json`
7. Examine `reports/helpsteer_schema.json` and the actual HF dataset card. **Confirm preference field structure, value meaning, and original split metadata** before running preprocessing.

See the README for commands. Do not commit raw licensed datasets or credentials.

## 1. Raw data inventory and annotation audit — first task

Tools: Hugging Face `datasets`, `pandas`, `pyarrow`; local CPU.

- Download `nvidia/HelpSteer2` with `data_dir='preference'`; inspect column names, data types and a few redacted/sample records. Save dataset revision, license, provenance and original train/validation split distinctions.
- Independently verify how `preference_strength`, `all_preferences_unprocessed`, and any filtered annotation array are formed. Some fields are published **aggregates** and should not be mistaken for raw individual votes. A published aggregate of zero need not mean an individual annotator chose a tie.
- Run `python -m judge_audit.preprocess` **only after selecting the confirmed raw vote column**. The HelpSteer2 parser requires individual `{strength, justification}` records and refuses other nested structures. The publisher's card defines positive strength as favoring response 2 and zero as a tie; confirm against the downloaded snapshot.
- Output `pairs.parquet` (one row per original pair), `annotation_report.json`, and an audit of invalid/missing vote entries. Check duplicate pair IDs and source `split` values.
- Verify row counts and stratum availability rather than assume 400 samples per group. Record totals by number of annotators and unanimous/contested status, and by **documented** margin bins.

**Provisional, to be approved:** for raw individual signed values `s_i`, define signed mean `m = mean(s_i)`, absolute margin `abs(m)`, average vote magnitude `mean(abs(s_i))`, and `contested = any(s_i < 0) and any(s_i > 0)`; unanimous is all valid signs identical. Do not call these the publisher's aggregate score. Require a minimum annotation count (e.g. >=3) only after checking availability and reporting exclusions. The code currently computes descriptive raw-vote statistics, but deliberately **does not** implement final binning or group matching.

## 2. Leakage-free data splits

Tools: hashing, pandas, `scikit-learn` if needed; no GPU.

- Assign canonical, **order-independent** pair hashes (prompt + two original response texts, sorted for hashing), plus the original response numbering.
- Create frozen train, development, and untouched HelpSteer2 audit partitions. The initial split command keeps the source's internal `val`/`validation` rows in audit and refuses overwrite.
- Split at **prompt-level** if multiple response pairs share a prompt; canonical near-duplicates should be detected across splits. Variants of a pair inherit the original pair's partition.
- Keep **MT-Bench entirely outside** training, prompt tuning, edit selection and adapter selection. Check normalized prompt/response overlaps across datasets, not just IDs.
- Save split manifests with seeds, source revision, counts, and hashes; assert intersections are empty. Starter helpers implement exact disjointness checks, not fuzzy leakage detection.

**Initial split rule:** whitespace-normalized exact prompt groups; source-defined `val`/`validation` is audit only. Inspect semantic near-duplicates and cross-dataset overlaps before experiments.

## 3. Judging harness — 50-pair pilot first

Tools: `transformers` tokenizer/chat templates, `vLLM` for inference, JSONL/Parquet logging; L4 for smaller models.

- Define one baseline prompt requiring a pairwise A/B verdict and a rubric prompt. Keep the prompt revision fixed and store it with every run.
- Run original order and reversed order. Convert the model's output **back to response_1/response_2** using the position map. Unit-test this mapping before spending GPU time.
- Log model repo + revision, precision/quantization, seed, decoding settings, prompt/response tokens, truncation, GPU hardware, wall time, failure cases.
- Pilot 50 **audit-safe development** pairs and measure throughput, max input length, precision feasibility, failure rate, and cost. Count inference calls rather than just pairs.
- For size claims, use at least four sizes from a family **or** add a second family; the current three-size proposal supports only a limited within-family observation. Hold precision consistent (e.g., BF16) across the size comparison; a quantized 14B against nonquantized smaller models is confounded. If unavoidable, run a matched precision ablation.

**TODO decisions:** model checkpoints and licenses, BF16/AWQ vs. GPU availability, vLLM constrained decoding vs. normalized A/B log probabilities, whether ties are allowed at inference, and choice of generation seeds. The starter does not fake a production judge implementation.

## 4. Controlled perturbations and validation

- Generate all transformations **offline** with immutable manifests: `pair_id`, `edit_id`, `target_response_id` (1 or 2), original/edited text hashes, validation status.
- Each of response_1 and response_2 is edited **separately**, regardless of human-majority winner status. For pairs with a human majority, analyze winner-edited vs. loser-edited as a secondary view.
- Conditions: original, A/B swap, **exact-repeat no-op** (repeatability), **neutral-edit no-op** (noise floor), rule-based formatting, and limited **content removal** (positive quality-control). Optional lengthening only after validation.
- Evaluate edited and unedited versions under both positions. Never modify both sides in the primary causal comparison.
- Two reviewers validate a prespecified number of edits for preservation of substantive quality; content-removal should be verified to actually harm an otherwise useful answer. Log reject reasons and reject uncertain edits.

**TODO:** exact neutral-edit operation and allowed formatting transformations: semantically inert changes are not guaranteed to be usefulness-inert. Avoid automatic edits that silently introduce new factual content.

## 5. Human agreement and uncertainty analysis

- Preserve the **full raw human vote distribution** for each pair. For a judge choosing response_1, a possible descriptive measure is `fraction of human votes favoring response_1`; analogous for response_2. For a contested pair, this is **soft agreement with a randomly drawn annotator**, not exact-match accuracy.
- **TODO:** choose and preregister tie handling: human ties, judge ties/abstentions, zero aggregate margins, and whether raw zero votes are allowed in this dataset. Report tie/abstention frequency and coverage; do not conflate 0.5 credit for abstention with correctness.
- H1: compare position-flip rates by raw margin **within unanimous groups**.
- H2: draw margin distributions for contested vs unanimous; **match/caliper within common support** and adjust for number of annotators. If overlap is weak, report H2 descriptively rather than asserting an independent effect.
- Report bootstrap 95% CIs with **original response pair** as cluster. Where multiple pairs share a prompt, use a prompt-cluster sensitivity analysis. MT-Bench should cluster by original question ID, retaining turns and repeated annotation together.
- Report agreement, position flips, formatting flips, no-op flips, positive-control sensitivity, and cost/1,000 pairwise judgments. A mechanically order-invariant averaging rule must be evaluated on human agreement and other perturbations, not its guaranteed order consistency.

**TODO:** define final confidence-interval unit, margin-matching algorithm, tie policy, regression covariates, and multiple-testing correction before primary audit results.

## 6. Mitigation and LoRA (after baseline is frozen)

- Baselines: **untuned Qwen2.5-1.5B**, ordinary (unaugmented) preference LoRA, and formatting-augmented preference LoRA. Also evaluate rubric prompt, position balancing and their combination at inference time.
- Same original training IDs and fixed update/token budget for both LoRA conditions; augmented variants are created **only from training pairs**.
- Use >=3 seeds per LoRA variant when feasible; report seed-to-seed variance. Keep audit pair IDs and **all derived variants** excluded from training. Exclude MT-Bench, including any overlapping prompts/answers.
- MT-Bench is the decisive **out-of-domain H4** test: report augmented-vs-standard LoRA differences in human agreement and formatting sensitivity. Do not tune on MT-Bench.

**TODO:** SFT vs. preference loss for the A/B classifier, handling contested training votes, output tokenization, augmentation ratio, exact hyperparameters, and selection criteria. Implement when these are resolved; no placeholder training pretending to be validated code.

## 7. Reporting

Main outputs: `stratum_counts.csv`, `margin_overlap.png`, `baseline_by_size.csv`, `flips_by_margin.png` with pair/prompt clustered CIs, `matched_h2.csv`, `mitigation_comparison.csv`, `mtbench_transfer.csv`, `cost_per_1000.csv`, plus a failure-case gallery with data-use restrictions respected.

Measure GPU-hours, peak memory, end-to-end throughput/latency, price and assumptions; track both model inference and LoRA training separately. Attach reproducible commands and random seeds to each figure.

## Explicit decision log (fill before full runs)

| Decision | Status | Owner / notes |
|---|---|---|
| HelpSteer2 raw vote schema / sign mapping | CARD VERIFIED; SNAPSHOT TODO | Card shows `{strength, justification}`; positive=response 2; zero=tie |
| Aggregated vs. raw margin and valid annotation counts | TODO APPROVE | Avoid circular definition |
| Tie/abstention and soft human agreement metric | TODO APPROVE | Include contested pairs |
| Minimum annotations and stratum sample sizes | TODO COUNT | Report available and selected |
| H2 common-support matching | TODO APPROVE | No independent effect without overlap |
| Exact no-op and formatting transformations | TODO VALIDATE | Both target sides |
| Model roster / precision across sizes | TODO DECIDE | Scale confounding |
| Judge output method and token handling | TODO DECIDE | Validate tokenizer |
| LoRA loss, labels, augmentation and multiple seeds | TODO DECIDE | MT-Bench untouched |
| CI clusters / regression specifications | TODO APPROVE | Pair vs. prompt |

## Safe definition of done for initial scaffold

`pytest -q` passes; schema report exists; the team has inspected actual raw vote representation; per-pair summaries and strata count output are reproducible; data manifests are immutable and leakage checks are tested. **Nothing about model fairness should be claimed until these are done.**
