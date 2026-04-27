# Fermi Estimation Fine-Tuning Pipeline — Overview

## Goal

Fine-tune a small language model (Qwen3-4B-Instruct-2507) to answer Fermi estimation questions by producing structured, step-by-step reasoning traces. The model should output a valid JSON object with factors, a formula, a calculation, and a final answer with an order-of-magnitude (OOM) estimate.

---

## Dataset

The master dataset contains ~5,008 Fermi estimation questions drawn from multiple sources (Science Olympiad, real FermiP problems, and synthetic questions). It is split into:

| Split | Size | Purpose |
|-------|------|---------|
| train | 2,818 | Verifier-filtered SFT training data |
| val   | 1,693 | Held-out validation |
| eval  | 500 (→ 317 after filtering) | Final model evaluation (held out until the end) |

### Eval set filtering

The original eval set of 500 questions includes 183 synthetic FermiP questions (`fermi_problems_synth_1.json` and `fermi_problems_synth_2.json`). These synthetic questions are known to have unreliable ground truth labels — the answers were generated programmatically and in some cases are wrong. Including them would give a misleading accuracy number.

We therefore filtered the eval set to **317 questions** containing only real/human-curated questions (Science Olympiad, real FermiP, and filtered George dataset entries). This ensures the final evaluation reflects performance on questions with correct ground truth.

---

## Output Format (Prompt Contract)

The model is asked to produce a single JSON object — no prose before or after. The schema is:

```json
{
  "restatement": "one sentence in your own words",
  "assumptions": ["short bullet", "..."],
  "factors": [
    {
      "name": "label",
      "symbol": "short_var_name",
      "low": <number>,
      "high": <number>,
      "point": <number>,
      "unit": "unit or ''",
      "why": "one sentence justification"
    }
  ],
  "formula": "e.g. N_people * liters_per_day * 365",
  "calculation": "one-line arithmetic using point values",
  "final": {
    "value": <number>,
    "unit": "unit",
    "oom": <integer>
  },
  "confidence": "low | medium | high"
}
```

This format was designed to be machine-parseable for scoring (we score on `final.oom` — the order-of-magnitude exponent) while also being human-readable and educationally meaningful. The `factors` array forces the model to decompose the problem explicitly, and the `formula` + `calculation` fields make the reasoning auditable.

---

## What We Tried First — And Why It Didn't Work

### Original approach: Live Tavily web search

The original verifier (`run_verifier.py`) made live calls to the Tavily search API at inference time. For each question, the model would:
1. Generate search queries
2. Call Tavily to retrieve web results
3. Use those results to compute an answer

**Why this didn't work:**
- **Expensive**: Tavily charges per query. Running this across thousands of questions was cost-prohibitive.
- **Rate limits**: Running with concurrency caused Tavily rate limit errors.
- **Slow**: Each question required multiple API round-trips, making large runs impractical.
- **Too strict on assumption lookup**: The assumption-matching logic tried to look up specific named facts from search results but often failed to find a relevant match, leaving the model with no grounding evidence.
- **Cache files became a maintenance problem**: We tried caching Tavily responses to a JSON file (`search_cache.json`) to avoid repeat calls, but the cache grew large, was hard to share between teammates, and still required Tavily for new questions.

### Why we moved to a static lookup

We pre-fetched 500 Fermi-relevant facts using Tavily once, saved them to `datasets/fermi_facts.json`, and replaced all live API calls with a keyword-overlap lookup against this static file. This meant:

- **Zero API calls at inference time** — no cost, no rate limits
- **Fully reproducible** — the same file is used by everyone
- **Shareable** — committed to the repo, no setup required
- **Fast** — a simple in-memory keyword match takes microseconds

The tradeoff is that the static file only has 500 entries and may not always contain a perfectly matched fact. But in practice, the verifier still generates reasonable answers because the model can fall back on its own knowledge when no match is found.

---

## Current Pipeline

### Step 1 — Verification (`run_verifier_cached.py`)

The verifier uses `Qwen/Qwen3-30B-A3B-Instruct-2507` (a larger model) to judge whether each training question has a correct answer. It runs in two stages per question:

**Stage 1 — Search plan**: The model is asked to identify what facts/constants are needed to solve the problem and generate search queries for them.

**Stage 2 — Computation**: The model is given the search evidence (retrieved from `fermi_facts.json` via keyword overlap) and asked to produce a full structured answer in the prompt contract JSON format.

The verifier then compares the model's predicted OOM against the ground-truth OOM label. Each record gets a `consensus_verdict`:
- `correct` — predicted OOM matches ground truth
- `incorrect` — predicted OOM is off
- `uncertain` — model failed to produce a parseable response

**Keyword overlap matching** (`lookup_fermi_facts`): The model's search queries are matched against the 500 pre-fetched facts by counting word overlap (filtering stopwords). The best-matching fact is returned as search evidence. A minimum overlap of 2 content words is required for a match.

**Batching strategy**: The verification was split into batches of 250 questions to protect against session timeouts on the university CRC network. Ria covered train[0–1408] + val[0–846]; Achuthan covered train[1409–2817] + val[847–1692].

**Session management**: A module-level `ServiceClient` singleton was used to avoid Tinker's concurrent session limit. With `concurrency=4` and two inference calls per question, naively creating a new `ServiceClient` per call would hit the session cap immediately.

**Checkpointing**: The verifier saves a `.partial` checkpoint every 25 questions so that a crash mid-batch doesn't require restarting from scratch.

#### Verdict summary across all train + val batches

| Split | Correct | Incorrect | Uncertain | Total |
|-------|---------|-----------|-----------|-------|
| train | 975 | 1,783 | 60 | 2,818 |
| val   | 627 | 1,120 | 58 | 1,693* |
| **Total** | **1,602** | **2,903** | **118** | **4,511** |

*Val counts are approximate.

### Step 2 — Merging (`merge_verification_batches.py`)

After both teammates finished their batches, all per-batch JSON files were merged:

```bash
python scripts/merge_verification_batches.py \
    --inputs run/verification_train_ria_b*.json run/verification_train_achuthan_b*.json \
    --output run/verification_train_all.json
```

### Step 3 — Filtering and Relabeling (`filter_relabel_dataset.py`)

The merged verification output is filtered to produce a clean training set:

- `correct` verdicts → kept as-is
- `uncertain` verdicts → dropped
- `incorrect` verdicts with OOM error ≤ 2 and high/medium model confidence → **relabeled** (ground truth is updated to the verifier's predicted OOM and kept)
- `incorrect` verdicts with OOM error > 2 → dropped

Relabeling recovers some borderline-incorrect examples where the ground truth label in the original dataset was off, rather than the model being wrong.

**Output sizes:**

| Split | Kept (filtered) |
|-------|----------------|
| train | 1,951 records |
| val   | 1,169 records |

### Step 4 — Trace preparation (`prepare_traces.py`)

The filtered verification records are converted to the SFT messages format expected by the Tinker training framework. Each record becomes a three-message conversation:

```
system: [Fermi estimation system prompt]
user:   [question + unit]
assistant: [structured JSON answer from the verifier's computation stage]
```

Output: `datasets/sft/filtered_v1/train.jsonl` (2,920 examples) and `val.jsonl` (200 examples).

---

## SFT Training

### Model

`Qwen/Qwen3-4B-Instruct-2507` — a small, efficient instruction-tuned model. LoRA (rank 32) is used to keep training cost low.

### First run (2 epochs, batch-size 2048) — `run/sft_filtered_v1`

All 2,920 examples fit into a single batch of size 2048, resulting in only **2 gradient steps total** (1 per epoch). Loss dropped from 1.097 → 0.581 but this was insufficient training.

### Second run (3 epochs, batch-size 32) — `run/sft_filtered_v1_v2`

Reducing the batch size to 32 gave **273 gradient steps** (91 per epoch × 3 epochs). Loss dropped from 1.081 → 0.128 — an 88% reduction.

| Step | Epoch | Loss |
|------|-------|------|
| 0 | 0 | 1.0806 |
| 10 | 0 | 0.3072 |
| 90 | 0 | 0.2041 |
| 120 | 1 | 0.1573 |
| 210 | 2 | 0.1005 |
| 272 | 2 | 0.1279 |

**Training config:**
- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Renderer: `qwen3_instruct`
- LoRA rank: 32
- LR: 4.91e-4 (linear schedule)
- Epochs: 3
- Batch size: 32 sequences
- Max length: 2048 tokens
- Est. cost: $2.31

### Baseline (no SFT)

Running the base Qwen3-4B model directly on the eval set:
- Exact OOM accuracy: **23.3%**
- Parse fail rate: **30%** (model doesn't reliably output valid JSON)

The high parse fail rate is the primary motivation for SFT — the model needs to learn the structured output format before RL can be applied.

---

## Next Steps

1. **Evaluate SFT model** on val set to measure improvement over the 23.3% baseline
2. **RL training** — use the SFT checkpoint (`run/sft_filtered_v1_v2`) as the starting policy and train with GRPO using ground-truth OOM rewards (exact match = 1, otherwise 0)
3. **Final evaluation** on `datasets/final/eval.json` (held out until the end)
