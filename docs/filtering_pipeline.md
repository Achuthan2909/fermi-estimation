# Filtering Pipeline

Documents every step that reduces the raw training pool to the final
science-domain filtered set used for fine-tuning.

---

## Run order

```
python scripts/deduplicate.py
python scripts/filter_science_domain.py
python scripts/score_complexity.py
```

---

## Step 1 — Exact Deduplication (`deduplicate.py`)

**What it does:**
Compares every training problem against all earlier problems. If two questions
are identical after lowercasing and stripping punctuation, the later one is
dropped. Differently-worded questions that ask the same thing are both kept —
the goal is only to remove true copy-pastes across sources.

**Output:** `data/filtered/dedup_manifest.json`
Each problem ID maps to either `"keep"` or `"duplicate_of:<original_id>"`.

**Result from our data:** 412 exact duplicates removed from 12,187 problems,
leaving 11,775 for the next step.

---

## Step 2 — Science-Domain Filter (`filter_science_domain.py`)

### What gets classified

Every problem in the deduplicated training set is sent to
`claude-sonnet-4-6` with this classifier prompt:

> List ALL science domain tags that apply:
> `physics`, `chemistry`, `biology`, `astronomy`, `earth_science`,
> `engineering`, `environmental_science`, `medicine`, `materials_science`,
> `energy`, `other_science` — or `non_science` if none apply.
>
> Rate the science relevance:
> - `core` — cannot be solved without domain-specific scientific knowledge
>   (e.g. ATP hydrolysis, Stefan-Boltzmann law)
> - `applied` — uses science in a practical/everyday context
>   (e.g. energy in food, car fuel efficiency)
> - `tangential` — science appears peripherally; estimation is mostly
>   common-sense (e.g. "how many steps to walk across the US")

### Keep / skip rule

| Condition | Decision |
|---|---|
| relevance = `core` AND at least one tag ≠ `non_science` | **PASS** ✓ |
| relevance = `applied` AND at least one tag ≠ `non_science` | **PASS** ✓ |
| relevance = `tangential` | **skip** ✗ |
| relevance = `non_science` | **skip** ✗ |
| API error / malformed response | **skip** ✗ (logged as `error`) |

So yes — **all `tangential` problems are excluded**, along with `non_science`.
The threshold is deliberate: `tangential` problems don't require scientific
reasoning to estimate, so they don't help the model learn the target skill.

### Examples from our data

| Question (truncated) | Tags | Relevance | Decision |
|---|---|---|---|
| "If I took an English penny and stretched it into a hollow sphere 1 atom thick…" | physics, materials_science, chemistry | core | **PASS** |
| "How long would it take to pluck each hair out of your body one at a time?" | biology | tangential | skip |
| "Estimate the total mass of atmospheric CO₂ above every tennis court…" | physics, chemistry | core | **PASS** |
| "How many steps to walk across the US?" | *(none)* | tangential | skip |

### Output

`data/filtered/science_domain_results.jsonl` — one record per classified
problem with fields: `id`, `source`, `question`, `tags`, `relevance`,
`passed_filter` (bool).

Problems that were skipped by the dedup manifest are not classified at all
and do not appear in this file.

---

## Step 3 — Complexity Scoring (`score_complexity.py`)

Runs only on problems where `passed_filter = true`. Computes:

| Metric | How |
|---|---|
| `word_count` | total words in question |
| `unit_count` | distinct physical unit mentions (regex over 60+ units) |
| `sentence_count` | number of sentences |
| `tag_count` | number of science tags assigned in Step 2 |
| `complexity_tier` | `low` / `medium` / `high` via tertile splits on a weighted composite score |

Output: `data/filtered/science_domain_scored.jsonl` and
`data/filtered/complexity_distribution.png`.

---

## What gets removed at each step

| Step | Removed | Reason |
|---|---|---|
| Exact dedup | 412 | Identical question text across sources |
| `tangential` | TBD after full run | Science is peripheral to the estimate |
| `non_science` | TBD after full run | No scientific reasoning required |
| API errors | TBD after full run | Malformed classifier response |

---

## Design rationale

**Why remove `tangential`?**
The project goal is to improve LLM performance on science-domain Fermi
problems specifically. Including problems that don't require scientific
knowledge would dilute the training signal and make it harder to measure
whether the model has actually learned domain-specific estimation skills.

**Why keep `applied` but not `tangential`?**
`applied` problems (e.g. fuel efficiency, food energy) still require the
model to use scientific quantities and units correctly, just in an everyday
context. `tangential` problems require no scientific knowledge at all —
the estimation relies entirely on common sense or general knowledge.

**Why exact dedup only (not fuzzy)?**
Differently-worded problems that ask the same thing provide useful variety —
different phrasings, contexts, and units help the model generalise. Only
true copy-pastes are removed.
