#!/usr/bin/env python3
"""
Build held-out eval benchmark (~100 problems) from three Fermi problem sources.

Sources:
  1. RealFP  — allenai/fermi (real-world problems with program solutions)
  2. Science Olympiad — landy8697/open-scioly-fermi
  3. EricAndrechek/FermiQuestions

Stratification targets:
  Source:  ~33-34 per source
  Variant: ~60% standard, ~40% distractor (deviations documented; only RealFP
           has distractor variants, so overall distractor rate will be lower)
  Length:  spread across short / medium / long within each source slice

Outputs:
  eval/eval_benchmark.jsonl
  data/realFP/          — cached raw JSON files
  data/synthFP/         — cached raw JSON files
  data/external/        — cached external-source files
"""

import json
import math
import re
import sys
from pathlib import Path

import requests

BASE_DIR = Path(__file__).parent.parent
EVAL_DIR = BASE_DIR / "eval"
DATA_DIR = BASE_DIR / "data"

REALFP_BASE  = "https://raw.githubusercontent.com/allenai/fermi/master/data/realFP"
SYNTHFP_BASE = "https://raw.githubusercontent.com/allenai/fermi/master/data/synthFP"
SCIOLY_URL   = "https://raw.githubusercontent.com/landy8697/open-scioly-fermi/master/data.js"
ERIC_URL     = "https://raw.githubusercontent.com/EricAndrechek/FermiQuestions/main/question-bank.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_json(url: str, local: Path) -> list | dict:
    if local.exists():
        print(f"    cached  {local.relative_to(BASE_DIR)}")
        return json.loads(local.read_text(encoding="utf-8"))
    print(f"    fetch   {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    local.write_text(r.text, encoding="utf-8")
    return json.loads(r.text)


def fetch_raw(url: str, local: Path) -> str:
    if local.exists():
        print(f"    cached  {local.relative_to(BASE_DIR)}")
        return local.read_text(encoding="utf-8")
    print(f"    fetch   {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    local.write_text(r.text, encoding="utf-8")
    return r.text


def parse_answer_log10(answer_str: str) -> float | None:
    """Parse a RealFP answer string like '2.3e+8 kg' → log10 float."""
    m = re.search(r'([-+]?\d+\.?\d*[eE][+-]?\d+|[-+]?\d+\.\d+|[-+]?\d+)', answer_str)
    if not m:
        return None
    try:
        val = float(m.group(1))
        if val > 0:
            return round(math.log10(val), 4)
    except (ValueError, OverflowError):
        pass
    return None


def classify_length(text: str) -> str:
    """Classify question into short / medium / long."""
    text = text.strip()
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    n_sent  = len(sentences)
    n_words = len(text.split())
    if n_sent <= 1 and n_words <= 20:
        return "short"
    if n_sent <= 3 and n_words <= 60:
        return "medium"
    return "long"


def is_math_puzzle(question: str) -> bool:
    """Return True for pure math computations that are not Fermi estimation."""
    q = question.strip()
    patterns = [
        r'^\s*(?:Calculate|Compute|Evaluate|Simplify)\s+[\d\(\[]',
        r'\b\d+\s*!\s*[=\s]',           # factorial: "125! ="
        r'\bWhat\s+is\s+\d+\s*[\^!]',  # "What is 7^43"
        r'^\s*\d[\d\s]*[\^!]',          # starts with "125^"
        r'=\s*$',                        # ends with "="
        r'\b(?:log|ln|sin|cos|tan)\s*\(',  # pure trig/log expressions
    ]
    return any(re.search(p, q, re.IGNORECASE) for p in patterns)


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------

def load_realfp() -> list[dict]:
    print("  Loading RealFP …")
    problems: list[dict] = []

    # Standard: test + val splits (avoid train — save for training)
    for split in ("test", "val"):
        path = DATA_DIR / "realFP" / f"{split}_realfp.json"
        url  = f"{REALFP_BASE}/{split}_realfp.json"
        records = fetch_json(url, path)
        for i, rec in enumerate(records):
            q   = rec.get("question", "").strip()
            log = parse_answer_log10(rec.get("answer", ""))
            if not q or log is None:
                continue
            problems.append({
                "id":           f"realfp_{split}_{i:03d}",
                "source":       "realFP",
                "question":     q,
                "answer_log10": log,
                "answer_raw":   rec.get("answer", "").strip(),
                "has_distractor": False,
                "length_tier":  classify_length(q),
                "notes":        f"allenai/fermi {split} split",
            })

    # Distractor: test_distractor_realfp.json
    dist_path = DATA_DIR / "realFP" / "distractor_setting" / "test_distractor_realfp.json"
    dist_url  = f"{REALFP_BASE}/distractor_setting/test_distractor_realfp.json"
    for i, rec in enumerate(fetch_json(dist_url, dist_path)):
        q   = rec.get("question", "").strip()
        log = parse_answer_log10(rec.get("answer", ""))
        if not q or log is None:
            continue
        problems.append({
            "id":           f"realfp_dist_{i:03d}",
            "source":       "realFP",
            "question":     q,
            "answer_log10": log,
            "answer_raw":   rec.get("answer", "").strip(),
            "has_distractor": True,
            "length_tier":  classify_length(q),
            "notes":        "allenai/fermi realFP distractor setting",
        })

    # Distractor: test_distractor_synthfp.json — counted as realFP source slice
    sdist_path = DATA_DIR / "synthFP" / "distractor_setting" / "test_distractor_synthfp.json"
    sdist_url  = f"{SYNTHFP_BASE}/distractor_setting/test_distractor_synthfp.json"
    for i, rec in enumerate(fetch_json(sdist_url, sdist_path)):
        q   = rec.get("question", "").strip()
        log = parse_answer_log10(rec.get("answer", ""))
        if not q or log is None:
            continue
        problems.append({
            "id":           f"synthfp_dist_{i:03d}",
            "source":       "realFP",  # grouped with allenai/fermi source slice
            "question":     q,
            "answer_log10": log,
            "answer_raw":   rec.get("answer", "").strip(),
            "has_distractor": True,
            "length_tier":  classify_length(q),
            "notes":        "allenai/fermi synthFP distractor setting",
        })

    return problems


def _parse_scioly_js(content: str) -> list[dict]:
    """Robustly parse the Science Olympiad data.js JavaScript file."""
    # Locate the first top-level array
    start = content.find('[')
    if start == -1:
        return []
    depth, end = 0, -1
    for i in range(start, len(content)):
        if content[i] == '[':
            depth += 1
        elif content[i] == ']':
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return []
    arr_str = content[start:end + 1]

    for attempt in range(3):
        try:
            return json.loads(arr_str)
        except json.JSONDecodeError:
            if attempt == 0:
                # Fix unquoted JS keys
                arr_str = re.sub(r'(?<=[{,\[\s])(\w+)\s*(?=:)', r'"\1"', arr_str)
            elif attempt == 1:
                # Remove trailing commas
                arr_str = re.sub(r',(\s*[}\]])', r'\1', arr_str)

    # Fallback: regex per-record extraction
    records = []
    pat = re.compile(
        r'\{[^{}]*?question\s*:\s*"((?:[^"\\]|\\.)*)"\s*,'
        r'[^{}]*?answer\s*:\s*(-?\d+(?:\.\d+)?)'
        r'(?:[^{}]*?source\s*:\s*"((?:[^"\\]|\\.)*)")?[^{}]*?\}',
        re.DOTALL,
    )
    for m in pat.finditer(content):
        records.append({
            "question": m.group(1),
            "answer":   float(m.group(2)),
            "source":   m.group(3) or "",
        })
    return records


def load_scioly() -> list[dict]:
    print("  Loading Science Olympiad …")
    raw_path = DATA_DIR / "external" / "science_olympiad_raw.js"
    js = fetch_raw(SCIOLY_URL, raw_path)
    records = _parse_scioly_js(js)
    print(f"    parsed {len(records)} raw records")

    problems: list[dict] = []
    for i, rec in enumerate(records):
        q = rec.get("question", "").strip()
        if not q or is_math_puzzle(q):
            continue
        try:
            log = float(rec["answer"])
        except (KeyError, TypeError, ValueError):
            continue
        # Sanity-check: log10 values outside −10…40 are almost certainly
        # math-puzzle artefacts or data errors
        if not (-10 <= log < 40):
            continue
        problems.append({
            "id":           f"scioly_{i:04d}",
            "source":       "science_olympiad",
            "question":     q,
            "answer_log10": log,
            "answer_raw":   f"10^{int(log)}",
            "has_distractor": False,
            "length_tier":  classify_length(q),
            "notes":        rec.get("source", ""),
        })
    return problems


def load_eric() -> list[dict]:
    print("  Loading EricAndrechek FermiQuestions …")
    raw_path = DATA_DIR / "external" / "eric_fermi_raw.json"
    data = fetch_json(ERIC_URL, raw_path)

    # Structure: {"questions": {"category": {"question_text": answer_int}}, "sources": {...}}
    problems: list[dict] = []
    idx = 0
    questions_obj = {}
    if isinstance(data, dict):
        questions_obj = data.get("questions", data)

    for cat, q_map in questions_obj.items():
        if not isinstance(q_map, dict):
            idx += 1
            continue
        for q_text, answer in q_map.items():
            # Strip HTML tags (e.g. <sup>43</sup> in some entries)
            q = re.sub(r'<[^>]+>', '', q_text).strip()
            if not q or is_math_puzzle(q):
                idx += 1
                continue
            try:
                log = float(answer)
            except (TypeError, ValueError):
                idx += 1
                continue
            if not (-10 <= log < 40):
                idx += 1
                continue
            problems.append({
                "id":           f"eric_{idx:03d}",
                "source":       "eric_fermi",
                "question":     q,
                "answer_log10": log,
                "answer_raw":   f"10^{int(log)}",
                "has_distractor": False,
                "length_tier":  classify_length(q),
                "notes":        f"category={cat}",
            })
            idx += 1

    return problems


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def _sample_by_length(pool: list[dict], n: int) -> list[dict]:
    """Sample n items from pool with even length-tier coverage."""
    if n <= 0:
        return []
    by_tier: dict[str, list] = {"short": [], "medium": [], "long": []}
    for p in pool:
        by_tier[p["length_tier"]].append(p)

    per_tier = n // 3
    result: list[dict] = []
    leftover = n
    for tier in ("short", "medium", "long"):
        take = min(per_tier, len(by_tier[tier]))
        # Space evenly for reproducibility
        step = max(1, len(by_tier[tier]) // max(take, 1))
        result.extend(by_tier[tier][::step][:take])
        leftover -= take

    # Fill remainder from the largest tier buckets
    if leftover > 0:
        remaining_pool = [p for p in pool if p not in result]
        result.extend(remaining_pool[:leftover])

    return result[:n]


def stratified_sample(
    pool: list[dict],
    n: int,
    distractor_rate: float = 0.4,
) -> list[dict]:
    standard   = [p for p in pool if not p["has_distractor"]]
    distractor = [p for p in pool if p["has_distractor"]]

    n_dist = min(round(n * distractor_rate), len(distractor))
    n_std  = n - n_dist

    return _sample_by_length(standard, n_std) + _sample_by_length(distractor, n_dist)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Building eval benchmark")
    print("=" * 60)

    realfp = load_realfp()
    scioly  = load_scioly()
    eric    = load_eric()

    n_realfp_std  = len([p for p in realfp if not p["has_distractor"]])
    n_realfp_dist = len([p for p in realfp if p["has_distractor"]])
    print(f"\nAvailable after filtering:")
    print(f"  RealFP    : {n_realfp_std} standard  + {n_realfp_dist} distractor")
    print(f"  Sci.Olymp.: {len(scioly)} standard  (no distractor variants)")
    print(f"  Eric FQ   : {len(eric)} standard  (no distractor variants)")

    # Target: ~34 / ~33 / ~33; distractor rate applies only to RealFP slice
    target_realfp = 34
    target_scioly = 33
    target_eric   = 33

    # Adjust if any source is smaller than target
    if len(scioly) < target_scioly:
        deficit = target_scioly - len(scioly)
        target_scioly = len(scioly)
        target_realfp += deficit // 2
        target_eric   += deficit - deficit // 2
    if len(eric) < target_eric:
        deficit = target_eric - len(eric)
        target_eric   = len(eric)
        target_realfp += deficit

    sampled_realfp = stratified_sample(realfp, target_realfp, distractor_rate=0.40)
    sampled_scioly = stratified_sample(scioly, target_scioly, distractor_rate=0.0)
    sampled_eric   = stratified_sample(eric,   target_eric,   distractor_rate=0.0)

    benchmark = sampled_realfp + sampled_scioly + sampled_eric

    # Write output
    EVAL_DIR.mkdir(exist_ok=True)
    out_path = EVAL_DIR / "eval_benchmark.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in benchmark:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- summary ----
    by_src  = {}
    by_dist = {True: 0, False: 0}
    by_len  = {"short": 0, "medium": 0, "long": 0}
    for p in benchmark:
        by_src[p["source"]] = by_src.get(p["source"], 0) + 1
        by_dist[p["has_distractor"]] += 1
        by_len[p["length_tier"]] += 1

    print(f"\n{'='*60}")
    print(f"Wrote {len(benchmark)} problems → {out_path.relative_to(BASE_DIR)}")
    print(f"\nBy source:")
    for src, cnt in sorted(by_src.items()):
        print(f"  {src:<22}: {cnt}")
    print(f"\nBy distractor variant:")
    print(f"  standard (no distractor): {by_dist[False]}")
    print(f"  has distractor          : {by_dist[True]}")
    pct = round(100 * by_dist[True] / len(benchmark))
    if pct < 35:
        print(f"  NOTE: distractor rate {pct}% < 40% target — Science Olympiad and")
        print(f"        EricAndrechek have no distractor variants (documented in eval/README.md)")
    print(f"\nBy length tier:")
    for tier, cnt in sorted(by_len.items()):
        print(f"  {tier:<8}: {cnt}")


if __name__ == "__main__":
    main()
