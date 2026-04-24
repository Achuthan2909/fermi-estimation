"""Classify each unified question as science-domain (physics/chem/bio/astro/earth/geo)
or not, using a local Ollama model.

Input:  datasets/processed/unified.jsonl
Output: datasets/processed/classified.jsonl  (same rows + {domain, is_science})
        datasets/processed/science_filter_report.json

Features:
  - Batches N questions per LLM call to cut latency.
  - Resumable: if classified.jsonl exists, already-tagged ids are skipped.
  - Deterministic prompt (temperature=0).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import ollama

REPO = Path(__file__).resolve().parent.parent
PROCESSED = REPO / "datasets" / "processed"

DOMAINS = ["physics", "chemistry", "biology", "astronomy", "earth_science", "mathematics", "other_science", "non_science"]
SCIENCE_DOMAINS = {"physics", "chemistry", "biology", "astronomy", "earth_science", "mathematics", "other_science"}

SYSTEM_PROMPT = """You classify Fermi-estimation questions by domain.

Output labels from: physics, chemistry, biology, astronomy, earth_science, mathematics, other_science, non_science.

Domain guide:
  physics       — energy, force, motion, thermodynamics, light, pressure, scale arguments from physical principles
  chemistry     — atoms, moles, reactions, compounds, materials
  biology       — cells, organisms, metabolism, blood, DNA, bacteria, ecology, physiology
  astronomy     — planets, stars, galaxies, solar system, space
  earth_science — atmosphere, climate, oceans, volcanoes, geology, earthquakes, geography
  mathematics   — combinatorics, probability, number theory, geometry, computing an exact quantity (e.g. "ways to pick 5 from 50", "compute π^19"). Still mathematics even if the result is large.
  other_science — scientific/engineering estimates that cross domains or don't fit neatly (reactor output, bandwidth, medical diagnostics)
  non_science   — finance, sports, pop-culture, demographics of human behavior, counts of commercial/cultural items, absurd comparisons with no physical meaning

Examples (follow strictly):
  Q: How many cells are in the human body?                    -> biology
  Q: How many atoms are in your body?                          -> chemistry
  Q: How many people are alive today?                          -> non_science
  Q: What is the mass of Mars?                                 -> astronomy
  Q: How many CO2 molecules in the atmosphere?                 -> earth_science
  Q: What is 7^43?                                             -> mathematics
  Q: Compute (10!)!                                            -> mathematics
  Q: How many different sets of 5 can you pick from 50?        -> mathematics
  Q: What is pi^19?                                            -> mathematics
  Q: How many hot dogs at MLB games per year?                  -> non_science
  Q: How many miles of blood vessels in a human?               -> biology
  Q: What is the area of Madagascar?                           -> earth_science
  Q: How many songs on Spotify?                                -> non_science
  Q: How much energy in a lightning bolt?                      -> physics
  Q: How much energy would it take to evaporate the oceans?    -> physics
  Q: How many proteins in a cell?                              -> biology
  Q: How long is DNA in all cells of the body?                 -> biology
  Q: How many people have lived on Earth?                      -> non_science
  Q: How many Bill Gates can fit in Mars?                      -> non_science
  Q: What is your SSN?                                         -> non_science
  Q: How many port-a-potties for a million-man march?          -> non_science

Rule of thumb: if answering requires real physical, biological, chemical, astronomical, or mathematical reasoning, pick that domain — even if the phrasing is whimsical or the result is absurd.
"""

USER_TEMPLATE = """Classify these {n} questions. Output ONLY a JSON array of {n} labels, in order.
Valid labels: physics, chemistry, biology, astronomy, earth_science, other_science, non_science.

Example output: ["physics","non_science","biology"]

Questions:
{questions}"""


def build_user_msg(batch: list[str]) -> str:
    lines = []
    for i, q in enumerate(batch, 1):
        lines.append(f"{i}. {q}")
    return USER_TEMPLATE.format(n=len(batch), questions="\n".join(lines))


_JSON_ARRAY_RE = re.compile(r"\[[^\[\]]*\]", re.DOTALL)


def parse_labels(text: str, n: int) -> list[str] | None:
    # Try strict JSON first, then fall back to the first bracketed array.
    for candidate in (text.strip(), *_JSON_ARRAY_RE.findall(text)):
        try:
            arr = json.loads(candidate)
        except Exception:
            continue
        if isinstance(arr, list) and len(arr) == n and all(isinstance(x, str) for x in arr):
            return [x.strip().lower() for x in arr]
    return None


def _content(resp) -> str:
    msg = getattr(resp, "message", None) or (resp.get("message") if isinstance(resp, dict) else None)
    if msg is None:
        return ""
    if hasattr(msg, "content"):
        return msg.content or ""
    return (msg.get("content") or "") if isinstance(msg, dict) else ""


def classify_batch(model: str, batch: list[str], num_predict: int = 2048) -> list[str]:
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_msg(batch)},
        ],
        options={"temperature": 0.0, "num_predict": num_predict},
    )
    text = _content(resp)
    labels = parse_labels(text, len(batch))
    if labels is None:
        # Defensive fallback: one-by-one on this batch.
        labels = []
        for q in batch:
            r = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f'Classify this single question. Output ONLY one label (no JSON).\n\nQuestion: {q}'},
                ],
                options={"temperature": 0.0, "num_predict": num_predict},
            )
            t = _content(r).strip().lower()
            labels.append(t.split()[0] if t else "non_science")
    # Coerce unknowns to non_science so downstream stays sane.
    cleaned = []
    for lbl in labels:
        lbl = lbl.strip().lower().strip('"').strip("'").rstrip(".,")
        if lbl not in DOMAINS:
            lbl = "non_science"
        cleaned.append(lbl)
    return cleaned


def load_done(path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done[row["id"]] = row
            except Exception:
                pass
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5:4b",
                    help="Ollama model. qwen3.5:4b = slower but more accurate (thinking model); qwen2.5:3b-instruct = ~10x faster but ~70%% accurate.")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--num-predict", type=int, default=2048,
                    help="Max tokens per call. Thinking models need 1500+; non-thinking can use 256.")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--sources", nargs="*", default=None,
                    help="Restrict to these sources (e.g. realFP scioly fermiq). Default = all.")
    ap.add_argument("--input", default=str(PROCESSED / "unified.jsonl"))
    ap.add_argument("--output", default=str(PROCESSED / "classified.jsonl"))
    args = ap.parse_args()

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

    t0 = time.time()
    written = 0
    with out_path.open("a") as f:
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i : i + args.batch_size]
            qs = [r["question"] for r in batch]
            try:
                labels = classify_batch(args.model, qs, num_predict=args.num_predict)
            except Exception as e:
                print(f"[batch {i}] error: {e}; skipping", file=sys.stderr)
                continue
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
                print(f"  {written}/{len(todo)}  rate={rate:.1f}/s  eta={eta/60:.1f} min")

    try:
        rel = out_path.relative_to(REPO)
    except ValueError:
        rel = out_path
    print(f"Done. Wrote {written} classifications to {rel}")


if __name__ == "__main__":
    main()
