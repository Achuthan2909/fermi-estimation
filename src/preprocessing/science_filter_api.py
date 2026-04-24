"""Classify unified Fermi questions by domain using OpenAI API (default: gpt-4o-mini).

Rationale: local Ollama on a thinking model takes ~12h for 14k items with ~75% accuracy
on a fast model. gpt-4o-mini runs the full dataset in ~20 minutes for ~$0.20 with higher
accuracy and zero local CPU load.

Input:  datasets/processed/unified.jsonl
Output: datasets/processed/classified.jsonl
        datasets/processed/classified_report.json

Resumable: already-tagged ids are skipped if the output file already exists.

Requires: OPENAI_API_KEY env var; `pip install openai`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

REPO = Path(__file__).resolve().parent.parent
PROCESSED = REPO / "datasets" / "processed"

DOMAINS = [
    "physics", "chemistry", "biology", "astronomy",
    "earth_science", "mathematics", "other_science", "non_science",
]
SCIENCE_DOMAINS = {
    "physics", "chemistry", "biology", "astronomy",
    "earth_science", "mathematics", "other_science",
}

SYSTEM_PROMPT = """You classify Fermi-estimation questions by domain.

Output labels from: physics, chemistry, biology, astronomy, earth_science, mathematics, other_science, non_science.

Domain guide:
  physics       — energy, force, motion, thermodynamics, light, pressure, scale from physical principles
  chemistry     — atoms, moles, reactions, compounds, materials
  biology       — cells, organisms, metabolism, blood, DNA, bacteria, ecology, physiology
  astronomy     — planets, stars, galaxies, solar system, space
  earth_science — atmosphere, climate, oceans, volcanoes, geology, earthquakes, geography
  mathematics   — combinatorics, probability, number theory, geometry, computing an exact arithmetic quantity
  other_science — scientific/engineering estimates that cross domains or don't fit neatly
  non_science   — finance, sports, pop-culture, demographics of human behavior, counts of commercial/cultural items, absurd comparisons without physical meaning

Examples (follow strictly):
  Q: How many cells are in the human body?                  -> biology
  Q: How many atoms are in your body?                        -> chemistry
  Q: How many people are alive today?                        -> non_science
  Q: What is the mass of Mars?                               -> astronomy
  Q: How many CO2 molecules in the atmosphere?               -> earth_science
  Q: What is 7^43?                                           -> mathematics
  Q: Compute (10!)!                                          -> mathematics
  Q: How many different sets of 5 can you pick from 50?      -> mathematics
  Q: How many hot dogs at MLB games per year?                -> non_science
  Q: How many miles of blood vessels in a human?             -> biology
  Q: What is the area of Madagascar?                         -> earth_science
  Q: How many songs on Spotify?                              -> non_science
  Q: How much energy in a lightning bolt?                    -> physics
  Q: How much energy would it take to evaporate the oceans?  -> physics
  Q: How many proteins in a cell?                            -> biology
  Q: How long is DNA in all cells of the body?               -> biology
  Q: How many people have lived on Earth?                    -> non_science
  Q: How many Bill Gates can fit in Mars?                    -> non_science
  Q: What is your SSN?                                       -> non_science

Rule of thumb: if answering requires real physical, biological, chemical, astronomical, or mathematical reasoning, pick that domain — even if phrased whimsically."""

USER_TEMPLATE = """Classify these {n} questions. Output ONLY a JSON array of exactly {n} strings, in order.
Valid labels: physics, chemistry, biology, astronomy, earth_science, mathematics, other_science, non_science.

Example output: ["physics","non_science","biology"]

Questions:
{questions}"""


def build_user_msg(batch: list[str]) -> str:
    return USER_TEMPLATE.format(
        n=len(batch),
        questions="\n".join(f"{i+1}. {q}" for i, q in enumerate(batch)),
    )


_JSON_ARRAY_RE = re.compile(r"\[[^\[\]]*\]", re.DOTALL)


def parse_labels(text: str, n: int) -> list[str] | None:
    for candidate in (text.strip(), *_JSON_ARRAY_RE.findall(text)):
        try:
            arr = json.loads(candidate)
        except Exception:
            continue
        if isinstance(arr, list) and len(arr) == n and all(isinstance(x, str) for x in arr):
            return [x.strip().lower() for x in arr]
    return None


def classify_batch(client: OpenAI, model: str, batch: list[str]) -> list[str]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_msg(batch)},
        ],
        temperature=0.0,
        max_tokens=400,
    )
    text = resp.choices[0].message.content or ""
    labels = parse_labels(text, len(batch))
    if labels is None:
        # Fall back to one-by-one if the batch parse failed.
        labels = []
        for q in batch:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Classify this question. Output ONLY one label (no JSON, no quotes).\n\n{q}"},
                ],
                temperature=0.0,
                max_tokens=16,
            )
            t = (r.choices[0].message.content or "").strip().lower()
            labels.append(t.split()[0] if t else "non_science")
    cleaned = []
    for lbl in labels:
        lbl = lbl.strip().lower().strip('"').strip("'").rstrip(".,")
        if lbl not in DOMAINS:
            lbl = "non_science"
        cleaned.append(lbl)
    # Token usage for billing math
    usage = {
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
    }
    return cleaned, usage


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if path.exists():
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    return done


# Pricing ($ per 1M tokens) — adjust as needed.
PRICING = {
    "gpt-4o-mini":   {"input": 0.15, "output": 0.60},
    "gpt-4o":        {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini":  {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano":  {"input": 0.10, "output": 0.40},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--input", default=str(PROCESSED / "unified.jsonl"))
    ap.add_argument("--output", default=str(PROCESSED / "classified.jsonl"))
    ap.add_argument("--budget-usd", type=float, default=5.0,
                    help="Abort if estimated spend exceeds this. Default $5.")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not set")

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = [json.loads(l) for l in in_path.open() if l.strip()]
    if args.sources:
        all_rows = [r for r in all_rows if r["source"] in args.sources]
    if args.limit:
        all_rows = all_rows[: args.limit]

    done = load_done(out_path)
    todo = [r for r in all_rows if r["id"] not in done]
    print(f"Total: {len(all_rows)}  already classified: {len(done)}  todo: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    client = OpenAI()
    price = PRICING.get(args.model, {"input": 1.0, "output": 3.0})

    t0 = time.time()
    written = 0
    prompt_tok = 0
    compl_tok = 0
    spent = 0.0
    with out_path.open("a") as f:
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i : i + args.batch_size]
            qs = [r["question"] for r in batch]
            try:
                labels, usage = classify_batch(client, args.model, qs)
            except Exception as e:
                print(f"[batch {i}] error: {e}; sleeping 5s and retrying once", file=sys.stderr)
                time.sleep(5)
                try:
                    labels, usage = classify_batch(client, args.model, qs)
                except Exception as e2:
                    print(f"[batch {i}] second failure: {e2}; skipping", file=sys.stderr)
                    continue
            prompt_tok += usage["prompt_tokens"]
            compl_tok += usage["completion_tokens"]
            spent = prompt_tok * price["input"] / 1e6 + compl_tok * price["output"] / 1e6
            for row, lbl in zip(batch, labels):
                row_out = dict(row)
                row_out["domain"] = lbl
                row_out["is_science"] = lbl in SCIENCE_DOMAINS
                row_out["classifier_model"] = args.model
                f.write(json.dumps(row_out, ensure_ascii=False) + "\n")
                written += 1
            f.flush()
            if (i // args.batch_size) % 10 == 0:
                elapsed = time.time() - t0
                rate = written / elapsed if elapsed > 0 else 0
                eta = (len(todo) - written) / rate if rate > 0 else float("inf")
                print(f"  {written}/{len(todo)}  rate={rate:.1f}/s  eta={eta/60:.1f} min  spent=${spent:.3f}")
            if spent > args.budget_usd:
                print(f"BUDGET EXCEEDED: ${spent:.3f} > ${args.budget_usd}. Stopping.", file=sys.stderr)
                break

    elapsed = time.time() - t0
    report = {
        "model": args.model,
        "items_classified": written,
        "elapsed_seconds": round(elapsed, 1),
        "prompt_tokens": prompt_tok,
        "completion_tokens": compl_tok,
        "spent_usd": round(spent, 4),
        "pricing_assumed_per_1m": price,
    }
    report_path = out_path.parent / "classified_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nDone. Wrote {written} classifications. Spent ~${spent:.3f}. "
          f"Report -> {report_path.name}")


if __name__ == "__main__":
    main()
