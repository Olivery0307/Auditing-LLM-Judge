# Open data decisions

These are choices about using the data that preprocessing will **not** make. They belong to whoever writes the experiments. Suggestions are not binding. Fill in each **Decision** line before full runs, and copy the result to the decision log in `IMPLEMENTATION_PLAN.md`.

Counts come from `nvidia/HelpSteer2` (`preference`, revision `990b2711a36180dd19d9c94b8627844866f8982a`), split with seed 42 and `dev_fraction` 0.1. They are from a trial run: **the official manifest is not frozen yet.**

## Already agreed

| Use | Data |
|---|---|
| H1–H3 (untuned judges) | Stratified sample from publisher `train` + `val`, excluding the development partition |
| LoRA training | `train` partition (H1–H3 may use some of the same pairs) |
| H4, unseen pairs | `val` only: 448 pairs, 412 with at least 3 annotators |
| H4, transfer | MT-Bench |
| Prompt and edit choices | Development partition only |

Two rules follow from this:

- Never use H1–H3 results as H4 evidence for pairs LoRA trained on. Compare LoRA judges with the untuned 7B/14B judges only on `val` or MT-Bench.
- H4 is too small to report by stratum (for example, 30 unanimous margin-3 pairs in `val`). Report it overall.

This relaxes the separate partitions still described in `IMPLEMENTATION_PLAN.md`.

## Handled by preprocessing

These are not for you to decide; they are listed so you know what to expect. The counts below assume at least 3 annotators for audit pairs, raw votes as the margin source, the five strata shown, and row 3960 excluded because its two responses are identical. Preprocessing also settles the sample size and whether extra H2 pairs are added, parses multi-turn prompts, flags near-duplicates, and prepares the MT-Bench table. Rows outside the audit sample stay in `pairs.parquet`.

## Terms and data you will need

- **Vote:** an integer of size 1, 2 or 3. Positive favors response 2. Annotators could not choose a tie, and there are no zero votes.
- **Margin:** absolute value of the mean vote (`absolute_mean_margin_raw`). It can be fractional, such as 1.33.
- **Unanimous / contested:** all votes have the same sign / both signs appear.
- **Columns already in `pairs.parquet`:** `raw_votes`, `n_annotators`, `positive_votes`, `negative_votes`, `mean_signed_margin_raw`, `absolute_mean_margin_raw`, `mean_absolute_strength_raw`, `vote_share_response_1`, `vote_share_response_2`, `unanimous_raw`, `contested_raw`, `published_preference_strength`.
- **Planned, not implemented yet:** parsed conversation turns, `is_multi_turn`, stratum label, H1–H3 sample rank, MT-Bench table.

Eligible pairs (at least 3 annotators, row 3960 excluded):

| Stratum | All | Outside development | `val` | `train` partition |
|---|---|---|---|---|
| Unanimous, margin 1 | 1,059 | 956 | 54 | 902 |
| Unanimous, between 1 and 3 | 3,040 | 2,747 | 144 | 2,603 |
| Unanimous, margin 3 | 606 | 545 | 30 | 515 |
| Contested, margin below 1 | 2,956 | 2,644 | 139 | 2,505 |
| Contested, margin 1 or more | 745 | 678 | 45 | 633 |
| **Total** | **8,406** | **7,570** | **412** | **7,158** |

## Analysis and edits

### 1. [H2] How to adjust for margin and annotator count

Unanimous margins are always at least 1, so the 2,644 contested pairs below 1 have no unanimous counterpart. They support only a descriptive contested-vs-unanimous comparison. For "disagreement predicts flips beyond margin", use the 678 contested pairs at margin 1 or more. All of them have at least 4 annotators, while 89% of unanimous pairs have 3.

| One-to-one matching rule (outside development) | Contested pairs matched |
|---|---|
| Margin bins (1 to <2, 2 to <3, exactly 3), any annotator count | 678 |
| Exact margin, any annotator count | 420 |
| Margin bins, same annotator count | 264 |
| Exact margin, same annotator count | 236 |

The publisher removed the 10% of tasks whose three most similar votes spread by more than 2 (HelpSteer2-Preference paper, Section 2). Many of the strongest disagreements are therefore missing, and the report should say so.

- **Suggestion:** regression on all 678 contested pairs, adjusting for margin and annotator count, with the 236 exact matches as a robustness check. With 236 pairs per group, a difference of about 10 percentage points is detectable (20% base flip rate, 80% power, unpaired).
- **Decision:**

### 2. [H3, H4, maybe H2] Which pairs receive edits

H2 needs edits only if "less stable" includes flips caused by edits; the proposal does not say whether it means position flips only.

- **Options:** all of the H1–H3 sample, or a subset to save compute.
- **Decision:**

## Labels

### 3. [H3, H4] Reference answer for contested pairs (evaluation)

125 contested pairs with at least 3 annotators have margin 0, so they have no winner by mean vote (122 in publisher `train`, 3 in `val`).

- **Options:** exact match with a majority or mean-vote winner; or soft agreement, the share of annotators who chose the judge's response.
- **Suggestion:** soft agreement for contested pairs, and report how many have no winner.
- **Decision:**

### 4. [H4] Training labels for contested pairs

Of the 7,158 eligible pairs in the `train` partition, 3,138 (44%) are contested:

| Contested pairs | Count | Example votes |
|---|---|---|
| Majority and mean pick the same winner | 2,908 | `[-1, 1, 1]` |
| Mean is 0 or vote count is tied | 156 | `[-1, -1, 1, 1]` |
| Majority and mean pick different winners | 74 | `[-1, -1, -1, 1, 3]` |

- **Options:** unanimous pairs only; the agreed winner, dropping the 230 ambiguous pairs; or soft labels from the vote shares.
- **Suggestion:** the agreed winner, dropping the 230 from training only. Both LoRA conditions must use the same labels.
- **Decision:**

### 5. [H4] Training pairs with 1 or 2 annotators

These 718 pairs (all partitions) are excluded from audit sampling but remain in `pairs.parquet`.

- **Options:** use those in the `train` partition for LoRA, or leave them out.
- **Decision:**

## Text format

### 6. [H1–H4] Showing multi-turn prompts to the judge

2,634 prompts (29%) contain earlier user and assistant turns. All alternate cleanly and end with a user turn. Preprocessing parses them into turns; the judge prompt decides how to show them.

- **Suggestion:** show the full conversation, and report results for multi-turn pairs separately as a check.
- **Decision:**

### 7. [H1–H4] Context limit and truncation

Prompt plus both responses: median 3,421 characters, maximum 14,429, and 10 pairs above 12,000. Truncation could remove content unevenly between responses.

- **Suggestion:** set the context limit high enough to avoid truncation, and log any that happens.
- **Decision:**

## MT-Bench

### 8. [H4, exploratory turn-2 test] Which MT-Bench judgments to analyze

3,355 judgments cover 1,814 comparisons (question, turn, unordered model pair). 853 have only one judgment, and votes have no strength, so HelpSteer2 margin strata do not carry over. Preprocessing will keep every vote, tie and turn.

- **To choose:** tie handling, a minimum number of judgments, and whether turn 2 is only exploratory, as in the proposal. No tuning on MT-Bench.
- **Decision:**
