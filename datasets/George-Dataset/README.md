# Eval Benchmark — Fermi Estimation

## ⚠ HELD OUT — DO NOT USE FOR TRAINING OR FILTERING DECISIONS

This benchmark is frozen at construction time and must never be used for:
- training or fine-tuning any model
- deciding which problems survive the science-domain filter
- any hyperparameter or threshold selection

It exists solely for final evaluation.

---

## Contents

`eval_benchmark.jsonl` — 100 held-out Fermi estimation problems, one per line.

### Record schema

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique problem ID (e.g. `realfp_test_001`, `scioly_0042`) |
| `source` | string | `realFP`, `science_olympiad`, or `eric_fermi` |
| `question` | string | Full question text |
| `answer_log10` | float | Ground-truth answer as log₁₀ (base-10 exponent) |
| `answer_raw` | string | Original answer string from source (e.g. `"2.3e+8 kg"` or `"10^9"`) |
| `has_distractor` | bool | Whether irrelevant/misleading context facts are included |
| `length_tier` | string | `short` (≤1 sentence, ≤20 words), `medium` (2–3 sentences, 21–60 words), `long` (4+ sentences or >60 words) |
| `notes` | string | Source split or competition attribution |

---

## Source Breakdown

| Source | Total | Standard | Distractor | Short | Medium | Long |
|---|---|---|---|---|---|---|
| `realFP` | 34 | 20 | 14 | 12 | 12 | 10 |
| `science_olympiad` | 33 | 33 | 0 | 11 | 11 | 11 |
| `eric_fermi` | 33 | 33 | 0 | 20 | 11 | 2 |
| **Total** | **100** | **86** | **14** | **43** | **34** | **23** |

---

## Data Sources

### 1. RealFP (`realFP`)
**Origin:** [allenai/fermi](https://github.com/allenai/fermi) — Kalyan et al. (2021)

928 real-world Fermi problems with executable program solutions (intermediate
reasoning steps, not just final answers). Three task settings: facts-provided,
distractor, and blind. Standard problems are drawn from the **test** and **val**
splits; distractor problems are drawn from `test_distractor_realfp.json` and
`test_distractor_synthfp.json` (synthFP distractor setting, grouped under
the same allenai/fermi source slice).

**Answer verification:** Answers are computed from human-curated executable
programs using unit-aware arithmetic. Log₁₀ is derived by parsing the
scientific-notation answer string.

### 2. Science Olympiad (`science_olympiad`)
**Origin:** [landy8697/open-scioly-fermi](https://github.com/landy8697/open-scioly-fermi)

Competition Fermi questions from Science Olympiad invitationals (2020–2024).
Answer field stores the base-10 exponent directly (already `answer_log10`).

**Answer verification:** Answers are competition-graded ground truth. Problems
where the answer field falls outside the range −10 to 39 (suggesting
a math computation rather than a Fermi estimate) were excluded, as were
explicit math puzzles detected by regex (factorial, power-of-integer notation,
questions starting with "Calculate" or "Compute" followed by arithmetic).

**No distractor variants available in this source.**

### 3. EricAndrechek FermiQuestions (`eric_fermi`)
**Origin:** [EricAndrechek/FermiQuestions](https://github.com/EricAndrechek/FermiQuestions)

300+ community-sourced problems in `question-bank.json`, organized by
competition/category (og, NC State, Yale YPO, Cornell, UMich, etc.).
Answer field stores the base-10 exponent directly.

**Answer verification:** Same as Science Olympiad: answers are community-curated
base-10 exponents. Same math-puzzle filters applied (HTML tags stripped first).

**No distractor variants available in this source.**

---

## Stratification Logic

### Source stratification
Target: ~33–34 problems per source (roughly equal thirds).
Actual: 34 / 33 / 33. All targets met.

### Variant (distractor) stratification
Target: ~60% standard, ~40% distractor within each source slice.

**Documented deviation:** Only the allenai/fermi source has distractor variants.
Science Olympiad and EricAndrechek do not publish distractor settings, so their
slices are 100% standard. Overall benchmark distractor rate is 14% (14/100),
not the 40% target. This is a structural limitation of the available data, not
a curation error. The 14 distractor problems present in the RealFP slice (40%
of that slice) still allow analysis of model sensitivity to irrelevant context.

### Length stratification
Target: spread across short / medium / long within each source.
Actual: 43 short / 34 medium / 23 long overall.

Science Olympiad problems are evenly split (11/11/11) because they are
competition questions with a controlled, uniform style.
EricAndrechek skews short (20/11/2) because most problems in that dataset
are single-sentence questions. This reflects the character of the source
rather than a curation failure.

---

## Scoring

Model answers are scored with the fp_score metric from the proposal:

```
fp_score = max(0, 1 − (1/3) × |log₁₀(predicted / actual)|)
```

- Score 1.0 = exact order-of-magnitude match
- Score 0.0 = off by ≥ 3 orders of magnitude

Predicted values are extracted by parsing the final numeric answer from model
output. `answer_log10` is the reference.
