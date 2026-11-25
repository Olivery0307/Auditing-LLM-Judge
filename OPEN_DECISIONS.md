# Open data decisions

These are choices about using the data that preprocessing does **not** make. They belong to whoever writes the experiments. Suggestions are not binding. Fill in each **Decision** line before full runs, and copy the result to the decision log in `IMPLEMENTATION_PLAN.md`.

How the data is organized, what the partitions mean and how to load it are in [DATA.md](DATA.md). Counts below come from the frozen data (HelpSteer2 revision `990b2711…`, manifest `12faa865…`). "Pool" means the 7,559 eligible pairs in `train` and `audit` that the H1–H3 sample is drawn from.

## Already settled

| Item | Settled as |
|---|---|
| Partition use | H1–H3 from `train` + `audit`; LoRA trains on `train`; H4 tests on `audit` and MT-Bench; `development` for tuning (team, 2026-11-25) |
| Eligibility | At least 3 annotators, raw votes, distinct responses |
| Groups | Unanimous margin rounded to 1, 2 or 3, plus contested |
| Near-duplicates | 31 removed after review |
| Sample order | Ranked within each group with seed 42 |

## Sample size

### 1. [H1–H3] How many pairs per group (*k*)

Every group has at least 982 pairs in the pool, so *k* up to 982 keeps the groups equal. *k* = 400 gives 1,600 pairs, the proposal's "up to 400 per stratum". The proposal's cost estimate does not say how many pairs it assumed.

- **Suggestion:** run the 50-pair pilot on `development`, measure cost per pair, then pick *k*. Ranks 1..*k* can be extended later without redoing earlier work.
- **Decision:**

## Analysis and edits

### 2. [H2] How to adjust for margin and annotator count

Unanimous margins are always at least 1, so the 2,641 contested pairs below 1 have no unanimous counterpart. They support only a descriptive contested-vs-unanimous comparison. For "disagreement predicts flips beyond margin", use the 677 contested pairs in the pool at margin 1 or more. All of them have at least 4 annotators, while 89% of unanimous pairs have 3.

| One-to-one matching rule (whole pool) | Contested pairs matched |
|---|---|
| Margin bins (1 to <2, 2 to <3, exactly 3), any annotator count | 677 |
| Exact margin, any annotator count | 419 |
| Margin bins, same annotator count | 264 |
| Exact margin, same annotator count | 236 |

A sample of *k* = 400 random contested pairs includes only about 80 of the 677, since the contested group is not split by margin. Take more from the ranking if H2 needs them.

The publisher removed the 10% of tasks whose three most similar votes spread by more than 2 (HelpSteer2-Preference paper, Section 2). Many of the strongest disagreements are therefore missing, and the report should say so.

- **Suggestion:** report H2 both with and without adjusting for annotator count. Extra annotators may have been added because of disagreement, so adjusting can remove part of the effect.
- **Decision:**

### 3. [H3, H4, maybe H2] Which pairs receive edits

H2 needs edits only if "less stable" includes flips caused by edits; the proposal does not say whether it means position flips only.

- **Options:** the whole H1–H3 sample, or its top ranks only (for example ranks 1–200 per group) to save compute.
- **Decision:**

## Labels

### 4. [H3, H4] Reference answer for contested pairs (evaluation)

125 contested pairs with at least 3 annotators have margin 0, so they have no winner by mean vote (109 in `train`, 13 in `development`, 3 in `audit`).

- **Options:** exact match with a majority or mean-vote winner; or soft agreement, the share of annotators who chose the judge's response.
- **Suggestion:** soft agreement for contested pairs, and report how many have no winner.
- **Decision:**

### 5. [H4] Training labels for contested pairs

Of the 7,147 eligible pairs in the `train` partition, 3,134 (44%) are contested:

| Contested pairs | Count | Example votes |
|---|---|---|
| Majority and mean pick the same winner | 2,904 | `[-1, 1, 1]` |
| Mean is 0 or vote count is tied | 156 | `[-1, -1, 1, 1]` |
| Majority and mean pick different winners | 74 | `[-1, -1, -1, 1, 3]` |

- **Options:** unanimous pairs only; the agreed winner, dropping the 230 ambiguous pairs; or soft labels from the vote shares.
- **Suggestion:** the agreed winner, dropping the 230 from training only. Both LoRA conditions must use the same labels.
- **Decision:**

### 6. [H4] Training pairs with 1 or 2 annotators

612 such pairs are in the `train` partition. They are not eligible for H1–H3 but remain in `helpsteer_pairs.parquet`.

- **Options:** use them for LoRA, or leave them out.
- **Decision:**

## Text format

### 7. [H1–H4] Showing multi-turn prompts to the judge

2,634 prompts (29%) contain earlier user and assistant turns. All alternate cleanly and end with a user turn, and the `turns` column has them parsed.

- **Suggestion:** show the full conversation, and report results for multi-turn pairs separately as a check.
- **Decision:**

### 8. [H1–H4] Context limit and truncation

Prompt plus both responses: median 3,421 characters, maximum 14,429, and 10 pairs above 12,000. Truncation could remove content unevenly between responses.

- **Suggestion:** set the context limit high enough to avoid truncation, and log any that happens.
- **Decision:**

## MT-Bench

### 9. [H4, exploratory turn-2 test] Which MT-Bench judgments to analyze

3,355 judgments cover 1,814 comparisons (question, turn, unordered model pair). 853 have only one judgment, 780 judgments are ties, and votes have no strength, so HelpSteer2 margin groups do not carry over. 18 judgments involve an empty `llama-13b` answer. Nothing is filtered.

- **To choose:** tie handling, a minimum number of judgments, whether to keep the empty-answer judgments, and whether turn 2 is only exploratory, as in the proposal. No tuning on MT-Bench.
- **Decision:**
