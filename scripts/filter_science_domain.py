#!/usr/bin/env python3
"""
Science-domain filtering pass (Task 2).

Loads all problems from all three sources EXCLUDING the held-out eval set,
then calls Claude claude-sonnet-4-20250514 to classify each problem with
science domain tags and a relevance tier (core / applied / tangential).

Filter rule:
  KEEP  if at least one tag is not non_science AND relevance is core or applied
  REJECT if relevance is tangential or non_science (logged but excluded)

Outputs:
  data/filtered/science_domain_results.jsonl
    fields: id, source, question, tags (list), relevance, passed_filter (bool)

Prints a summary table:
  - per source: total in, passed, survival rate %
  - per relevance tier: count of core / applied / tangential / non_science
  - per tag: how many problems carry each tag (multi-tag allowed)

Rate limiting: 1 request/sec to stay within API limits.
"""

import json
import os
import re
import time
from pathlib import Path

import anthropic

BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data"
EVAL_FILE   = BASE_DIR / "eval" / "eval_benchmark.jsonl"
OUT_FILE    = DATA_DIR / "filtered" / "science_domain_results.jsonl"
REALFP_BASE = "https://raw.githubusercontent.com/allenai/fermi/master/data/realFP"
SYNTHFP_BASE = "https://raw.githubusercontent.com/allenai/fermi/master/data/synthFP"
SCIOLY_URL  = "https://raw.githubusercontent.com/landy8697/open-scioly-fermi/master/data.js"
ERIC_URL    = "https://raw.githubusercontent.com/EricAndrechek/FermiQuestions/main/question-bank.json"

MODEL = "claude-sonnet-4-20250514"

CLASSIFIER_SYSTEM = """\
You are classifying Fermi estimation problems for a science research benchmark.
For the problem below, do two things:
1. List ALL science domain tags that apply from this set (return as many as fit — do not be restrictive):
   physics, chemistry, biology, astronomy, earth_science, engineering,
   environmental_science, medicine, materials_science, energy, other_science
   If no science domain applies at all, return: non_science
2. Rate the science relevance on a 3-point scale:
   core:       The problem cannot be solved without domain-specific scientific knowledge
               (e.g., ATP hydrolysis, Stefan-Boltzmann law)
   applied:    The problem uses science concepts in a practical or everyday context
               (e.g., energy in food, car fuel efficiency)
   tangential: Science appears peripherally but the estimation is mostly common-sense
               (e.g., "how many steps to walk across the US")
Respond in this exact JSON format with no extra text:
{"tags": ["tag1", "tag2"], "relevance": "core|applied|tangential"}
Problem: """


# ---------------------------------------------------------------------------
# Data loading helpers (reuse from build_eval_benchmark but local copies)
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> list | dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_scioly_js(content: str) -> list[dict]:
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
                arr_str = re.sub(r'(?<=[{,\[\s])(\w+)\s*(?=:)', r'"\1"', arr_str)
            elif attempt == 1:
                arr_str = re.sub(r',(\s*[}\]])', r'\1', arr_str)
    pat = re.compile(
        r'\{[^{}]*?question\s*:\s*"((?:[^"\\]|\\.)*)"\s*,'
        r'[^{}]*?answer\s*:\s*(-?\d+(?:\.\d+)?)'
        r'(?:[^{}]*?source\s*:\s*"((?:[^"\\]|\\.)*)")?[^{}]*?\}',
        re.DOTALL,
    )
    return [
        {"question": m.group(1), "answer": float(m.group(2)), "source": m.group(3) or ""}
        for m in pat.finditer(content)
    ]


def _is_math_puzzle(q: str) -> bool:
    patterns = [
        r'^\s*(?:Calculate|Compute|Evaluate|Simplify)\s+[\d\(\[]',
        r'\b\d+\s*!\s*[=\s]',
        r'\bWhat\s+is\s+\d+\s*[\^!]',
        r'^\s*\d[\d\s]*[\^!]',
        r'=\s*$',
        r'\b(?:log|ln|sin|cos|tan)\s*\(',
    ]
    return any(re.search(p, q, re.IGNORECASE) for p in patterns)


def load_all_training_problems(eval_ids: set[str]) -> list[dict]:
    """Load all problems from all sources, excluding eval IDs."""
    problems: list[dict] = []

    # ---- RealFP train split ----
    for split in ("train",):
        path = DATA_DIR / "realFP" / f"{split}_realfp.json"
        if not path.exists():
            import requests
            url = f"{REALFP_BASE}/{split}_realfp.json"
            print(f"  Fetching {url}")
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            path.write_text(r.text)
        records = _load_json(path)
        for i, rec in enumerate(records):
            pid = f"realfp_{split}_{i:03d}"
            if pid in eval_ids:
                continue
            q = rec.get("question", "").strip()
            if q:
                problems.append({"id": pid, "source": "realFP", "question": q})

    # ---- RealFP train distractor ----
    dist_path = DATA_DIR / "realFP" / "distractor_setting" / "train_distractor_realfp.json"
    if not dist_path.exists():
        import requests
        url = f"{REALFP_BASE}/distractor_setting/train_distractor_realfp.json"
        print(f"  Fetching {url}")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        dist_path.write_text(r.text)
    for i, rec in enumerate(_load_json(dist_path)):
        pid = f"realfp_dist_train_{i:03d}"
        if pid in eval_ids:
            continue
        q = rec.get("question", "").strip()
        if q:
            problems.append({"id": pid, "source": "realFP", "question": q})

    # ---- SynthFP train distractor ----
    sdist_path = DATA_DIR / "synthFP" / "distractor_setting" / "train_distractor_synthfp.json"
    if not sdist_path.exists():
        import requests
        url = f"{SYNTHFP_BASE}/distractor_setting/train_distractor_synthfp.json"
        print(f"  Fetching {url}")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        sdist_path.write_text(r.text)
    for i, rec in enumerate(_load_json(sdist_path)):
        pid = f"synthfp_dist_train_{i:04d}"
        if pid in eval_ids:
            continue
        q = rec.get("question", "").strip()
        if q:
            problems.append({"id": pid, "source": "realFP", "question": q})

    # ---- Science Olympiad (all not in eval) ----
    raw_js_path = DATA_DIR / "external" / "science_olympiad_raw.js"
    if not raw_js_path.exists():
        import requests
        print(f"  Fetching {SCIOLY_URL}")
        r = requests.get(SCIOLY_URL, timeout=60)
        r.raise_for_status()
        raw_js_path.write_text(r.text)
    so_records = _parse_scioly_js(raw_js_path.read_text(encoding="utf-8"))
    for i, rec in enumerate(so_records):
        pid = f"scioly_{i:04d}"
        if pid in eval_ids:
            continue
        q = rec.get("question", "").strip()
        if q and not _is_math_puzzle(q):
            try:
                log = float(rec["answer"])
                if not (-10 <= log < 40):
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            problems.append({"id": pid, "source": "science_olympiad", "question": q})

    # ---- EricAndrechek (all not in eval) ----
    eric_path = DATA_DIR / "external" / "eric_fermi_raw.json"
    if not eric_path.exists():
        import requests
        print(f"  Fetching {ERIC_URL}")
        r = requests.get(ERIC_URL, timeout=60)
        r.raise_for_status()
        eric_path.write_text(r.text)
    eric_data = _load_json(eric_path)
    # Structure: {"questions": {"category": {"question_text": answer_int}}, "sources": {...}}
    questions_obj: dict = {}
    if isinstance(eric_data, dict):
        questions_obj = eric_data.get("questions", eric_data)
    idx_e = 0
    for cat, q_map in questions_obj.items():
        if not isinstance(q_map, dict):
            idx_e += 1
            continue
        for q_text, answer in q_map.items():
            q = re.sub(r'<[^>]+>', '', q_text).strip()
            pid = f"eric_{idx_e:03d}"
            idx_e += 1
            if pid in eval_ids:
                continue
            if not q or _is_math_puzzle(q):
                continue
            try:
                log = float(answer)
                if not (-10 <= log < 40):
                    continue
            except (TypeError, ValueError):
                continue
            problems.append({"id": pid, "source": "eric_fermi", "question": q})

    return problems


# ---------------------------------------------------------------------------
# Claude classification
# ---------------------------------------------------------------------------

def classify_problem(client: anthropic.Anthropic, question: str) -> dict:
    """Call Claude to classify a single problem. Returns tags + relevance."""
    prompt = CLASSIFIER_SYSTEM + question
    try:
        msg = client.messages.create(
            model=MODEL,
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        result = json.loads(raw)
        if "tags" not in result or "relevance" not in result:
            raise ValueError("Missing keys")
        return {"tags": result["tags"], "relevance": result["relevance"], "raw": raw}
    except Exception as exc:
        return {"tags": [], "relevance": "error", "raw": str(exc)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Only classify the first N unclassified problems (useful for test runs)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.", flush=True)
        raise SystemExit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Load eval IDs to exclude
    eval_ids: set[str] = set()
    if EVAL_FILE.exists():
        with EVAL_FILE.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    eval_ids.add(json.loads(line)["id"])
    print(f"Eval set size (excluded): {len(eval_ids)}")

    # Load training problems
    print("Loading training problems …")
    problems = load_all_training_problems(eval_ids)
    print(f"Training problems to classify: {len(problems)}")

    # Resume from existing output
    done_ids: set[str] = set()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if OUT_FILE.exists():
        with OUT_FILE.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)["id"])
    print(f"Already done: {len(done_ids)}  Remaining: {len(problems) - len(done_ids)}")

    out_fh = OUT_FILE.open("a", encoding="utf-8")
    classified_this_run = 0

    try:
        for idx, prob in enumerate(problems):
            if args.limit is not None and classified_this_run >= args.limit:
                print(f"  --limit {args.limit} reached, stopping early.")
                break
            pid = prob["id"]
            if pid in done_ids:
                continue

            result = classify_problem(client, prob["question"])

            tags      = result["tags"]
            relevance = result["relevance"]
            passed    = (
                relevance in ("core", "applied")
                and any(t != "non_science" for t in tags)
            )

            record = {
                "id":           pid,
                "source":       prob["source"],
                "question":     prob["question"],
                "tags":         tags,
                "relevance":    relevance,
                "passed_filter": passed,
            }
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fh.flush()
            done_ids.add(pid)
            classified_this_run += 1

            status = "PASS" if passed else "skip"
            print(f"  [{idx+1}/{len(problems)}] {status}  {pid}  tags={tags}  relevance={relevance}")

            time.sleep(1.0)  # 1 req/sec rate limit

    finally:
        out_fh.close()

    # ---- Summary ----
    results = []
    with OUT_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Per-source stats
    src_stats: dict[str, dict] = {}
    for rec in results:
        s = rec["source"]
        if s not in src_stats:
            src_stats[s] = {"total": 0, "passed": 0}
        src_stats[s]["total"]  += 1
        src_stats[s]["passed"] += int(rec["passed_filter"])

    print("\nPer source:")
    print(f"  {'Source':<25} {'Total':>7} {'Passed':>7} {'Rate':>7}")
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*7}")
    for src, st in sorted(src_stats.items()):
        rate = 100 * st["passed"] / max(st["total"], 1)
        print(f"  {src:<25} {st['total']:>7} {st['passed']:>7} {rate:>6.1f}%")
        if rate < 40:
            print(f"    ⚠ LOW SURVIVAL (<40%) — may not be enough training samples")

    # Per-relevance-tier counts
    tier_counts: dict[str, int] = {}
    for rec in results:
        r = rec["relevance"]
        tier_counts[r] = tier_counts.get(r, 0) + 1

    print("\nPer relevance tier:")
    for tier in ("core", "applied", "tangential", "non_science", "error"):
        print(f"  {tier:<15}: {tier_counts.get(tier, 0)}")

    # Per-tag counts
    tag_counts: dict[str, int] = {}
    for rec in results:
        for t in rec["tags"]:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    print("\nPer tag (problems carrying each tag):")
    for tag, cnt in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"  {tag:<25}: {cnt}")

    total   = len(results)
    passed  = sum(1 for r in results if r["passed_filter"])
    print(f"\nOverall: {passed}/{total} passed ({100*passed//max(total,1)}%)")
    print(f"Output: {OUT_FILE}")


if __name__ == "__main__":
    main()
