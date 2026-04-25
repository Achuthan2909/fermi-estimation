# Verifier Commands

Commands for the two-stage Fermi verifier in `src/verifier/run_verifier.py`.

The verifier asks a model to plan searches, runs web search in Python, then gives
the retrieved evidence back to the model to compute and verify the answer. The
saved JSON includes the exact `input_messages`, search records, parsed model
responses, and Python verdicts.

## Setup

```bash
set -a
source .env
export SEARCH_PROVIDER=tavily
set +a
```

`set -a` exports every variable loaded from `.env` so child Python processes can
see `TINKER_API_KEY`, `TAVILY_API_KEY`, and any other settings. Do not commit
`.env`; share it out of band if a teammate needs the same keys.

`SEARCH_PROVIDER=tavily` uses Tavily Search. The fallback provider is
DuckDuckGo instant answers, but it often returns empty results for conversion
queries.

Optional Tavily depth:

```bash
export TAVILY_SEARCH_DEPTH=basic
# also supported by Tavily: fast, advanced, ultra-fast
```

## One-Question Smoke Test

```bash
python src/verifier/run_verifier.py \
  --dataset datasets/final/train.json \
  --offset 0 \
  --limit 1 \
  --model-a Qwen/Qwen3-4B-Instruct-2507 \
  --output run/verification_one.json
```

Change `--offset` to test a different row.

## One Model

Run a single verifier model by omitting `--model-b`:

```bash
python src/verifier/run_verifier.py \
  --dataset datasets/final/val.json \
  --limit 10 \
  --model-a Qwen/Qwen3-4B-Instruct-2507 \
  --output run/verification_val_10.json
```

## Two Models

Pass `--model-b` to run two independent verifier loops and compute consensus:

```bash
python src/verifier/run_verifier.py \
  --dataset datasets/final/train.json \
  --offset 0 \
  --limit 1 \
  --model-a Qwen/Qwen3-4B-Instruct-2507 \
  --model-b Qwen/Qwen3-4B-Instruct-2507 \
  --output run/verification_one_two_models.json
```

Each model plans its own searches, receives its own retrieved context, and
computes its own answer. `model_a` is not only for search, and `model_b` is not
only for verification.

## Full Train Split

```bash
python src/verifier/run_verifier.py \
  --dataset datasets/final/train.json \
  --all \
  --model-a Qwen/Qwen3-4B-Instruct-2507 \
  --output run/verification_train.json
```

## Full Validation Split

```bash
python src/verifier/run_verifier.py \
  --dataset datasets/final/val.json \
  --all \
  --model-a Qwen/Qwen3-4B-Instruct-2507 \
  --output run/verification_val.json
```

## Train And Validation Together

```bash
python src/verifier/run_verifier.py \
  --datasets datasets/final/train.json datasets/final/val.json \
  --all \
  --model-a Qwen/Qwen3-4B-Instruct-2507 \
  --output run/verification_train_val.json
```

The output records include `dataset` and `split` fields so train and validation
rows can be separated later.

## Cost Note

Full train plus validation is 4511 questions. With Tavily search and Tinker
sampling, this can take a long time and consume API credits. Run a small
`--limit` first before using `--all`.
