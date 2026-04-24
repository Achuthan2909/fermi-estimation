# Data Split Documentation

This document records the exact composition of the eval set and training pool,
and describes how the filtered training pool will be split for experiments.
All counts reflect the data as of the benchmark-construction commit.

---

## Eval Set — LOCKED

**Total: 100 problems**

The eval set is held out permanently. It is never used for training,
fine-tuning, or any filtering decision. See `eval/README.md` for full details.

### Source breakdown

| Source | Count |
|---|---|
| `realFP` (allenai/fermi test+val) | 34 |
| `science_olympiad` | 33 |
| `eric_fermi` | 33 |

### Distractor variant breakdown

| Variant | Count |
|---|---|
| Standard (no distractor) | 86 |
| Has distractor | 14 |

Only the `realFP` source provides distractor variants. Distractor rate in the
RealFP slice is 41% (14/34); overall benchmark distractor rate is 14%.

### Length variant breakdown

| Length tier | Count |
|---|---|
| Short (≤1 sentence, ≤20 words) | 43 |
| Medium (2–3 sentences, 21–60 words) | 34 |
| Long (4+ sentences or >60 words) | 23 |

---

## Filtered Training Pool

The training pool consists of all problems from all three sources **excluding**
the 100 eval IDs, then filtered by `scripts/filter_science_domain.py`.

### Pre-filter totals (training splits only)

| Source | Raw count (excl. eval) |
|---|---|
| `realFP` train split | ~696 |
| `realFP` train distractor | ~696 |
| `synthFP` train distractor | ~10,000 |
| `science_olympiad` (non-eval) | ~3,594 |
| `eric_fermi` (non-eval) | ~223 |

### Post-filter counts (after `filter_science_domain.py`)

> Run `python scripts/filter_science_domain.py` to populate
> `data/filtered/science_domain_results.jsonl`, then update this section.

| Source | Total in | Passed filter | Survival rate |
|---|---|---|---|
| `realFP` | — | — | — |
| `science_olympiad` | — | — | — |
| `eric_fermi` | — | — | — |

### Survival rate concerns

Sources where survival rate < 40% are flagged here after filtering runs.
The science-domain filter is intentionally broad (keeps `core` and `applied`)
to maximize training data retention.

### Per-relevance-tier breakdown

| Tier | Count |
|---|---|
| `core` | — |
| `applied` | — |
| `tangential` (excluded) | — |
| `non_science` (excluded) | — |

### Per-complexity-tier breakdown (after `score_complexity.py`)

> Run `python scripts/score_complexity.py` to populate
> `data/filtered/science_domain_scored.jsonl`.

| Complexity tier | Count |
|---|---|
| `low` | — |
| `medium` | — |
| `high` | — |

---

## Proposed Train / Validation Split

The filtered training pool will be split **80% train / 20% validation**,
stratified jointly by `source` and `complexity_tier` to ensure representation
of all source × complexity combinations in both halves.

This stratified split is implemented in the fine-tuning pipeline (not in this
repo yet) using `sklearn.model_selection.StratifiedShuffleSplit` with
a combined `source+complexity_tier` label.

---

## Distractor Experiment

Problems tagged `has_distractor: true` are tracked with that flag throughout
the pipeline. Model performance on distractor vs. standard variants will be
analyzed separately as an ablation:

- **Distractor source files:**
  - `data/realFP/distractor_setting/` (realFP distractor setting)
  - `data/synthFP/distractor_setting/` (synthFP distractor setting)
- **Eval set distractor problems:** 14 of 100 (in the `realFP` slice)
- **Training pool distractor problems:** drawn from the train distractor splits

The `has_distractor` flag is propagated through all output files
(`science_domain_results.jsonl`, `science_domain_scored.jsonl`) so that
post-training analysis can compare fp_score on distractor vs. standard subsets.

---

## Low Survival Rate Warnings

Any source with < 40% survival after the science-domain filter should be
treated with caution — there may not be enough training samples to learn
source-specific patterns.

> This section will be populated after `filter_science_domain.py` completes.
> Expected concern: `eric_fermi` and parts of `science_olympiad` may have
> low science-domain survival due to cultural/sports/history problems common
> in competition Fermi Question sets.
