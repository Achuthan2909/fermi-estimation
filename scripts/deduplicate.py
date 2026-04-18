#!/usr/bin/env python3
"""
Exact deduplication pass — run this BEFORE filter_science_domain.py.

Flags problems whose question text is identical (after lowercasing and
collapsing whitespace) to an earlier problem in the merged training set.
Differently-worded problems that ask the same thing are kept.

Outputs:
  data/filtered/dedup_manifest.json
    {"<id>": "keep"} or {"<id>": "duplicate_of:<original_id>"}

filter_science_domain.py reads this manifest and skips duplicates.
"""

import json
import re
import unicodedata
from pathlib import Path

BASE_DIR  = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / "data"
EVAL_FILE = BASE_DIR / "eval" / "eval_benchmark.jsonl"
OUT_FILE  = DATA_DIR / "filtered" / "dedup_manifest.json"

REALFP_BASE  = "https://raw.githubusercontent.com/allenai/fermi/master/data/realFP"
SYNTHFP_BASE = "https://raw.githubusercontent.com/allenai/fermi/master/data/synthFP"
SCIOLY_URL   = "https://raw.githubusercontent.com/landy8697/open-scioly-fermi/master/data.js"
ERIC_URL     = "https://raw.githubusercontent.com/EricAndrechek/FermiQuestions/main/question-bank.json"


def _load_json(path: Path) -> list | dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fetch(url: str, path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    import requests
    print(f"  Fetching {url}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    path.write_text(r.text)
    return r.text


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
        r'\{[^{}]*?question\s*:\s*"((?:[^"\\]|\\.)*)"\s*,[^{}]*?answer\s*:\s*(-?\d+(?:\.\d+)?).*?\}',
        re.DOTALL,
    )
    return [{"question": m.group(1), "answer": float(m.group(2))} for m in pat.finditer(content)]


def _is_math_puzzle(q: str) -> bool:
    patterns = [
        r'^\s*(?:Calculate|Compute|Evaluate|Simplify)\s+[\d\(\[]',
        r'\b\d+\s*!\s*[=\s]',
        r'\bWhat\s+is\s+\d+\s*[\^!]',
        r'^\s*\d[\d\s]*[\^!]',
        r'=\s*$',
    ]
    return any(re.search(p, q, re.IGNORECASE) for p in patterns)


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def load_training_problems(eval_ids: set[str]) -> list[dict]:
    problems: list[dict] = []

    for split in ("train",):
        path = DATA_DIR / "realFP" / f"{split}_realfp.json"
        _fetch(f"{REALFP_BASE}/{split}_realfp.json", path)
        for i, rec in enumerate(_load_json(path)):
            pid = f"realfp_{split}_{i:03d}"
            if pid in eval_ids:
                continue
            q = rec.get("question", "").strip()
            if q:
                problems.append({"id": pid, "source": "realFP", "question": q})

    dist_path = DATA_DIR / "realFP" / "distractor_setting" / "train_distractor_realfp.json"
    _fetch(f"{REALFP_BASE}/distractor_setting/train_distractor_realfp.json", dist_path)
    for i, rec in enumerate(_load_json(dist_path)):
        pid = f"realfp_dist_train_{i:03d}"
        if pid in eval_ids:
            continue
        q = rec.get("question", "").strip()
        if q:
            problems.append({"id": pid, "source": "realFP", "question": q})

    sdist_path = DATA_DIR / "synthFP" / "distractor_setting" / "train_distractor_synthfp.json"
    _fetch(f"{SYNTHFP_BASE}/distractor_setting/train_distractor_synthfp.json", sdist_path)
    for i, rec in enumerate(_load_json(sdist_path)):
        pid = f"synthfp_dist_train_{i:04d}"
        if pid in eval_ids:
            continue
        q = rec.get("question", "").strip()
        if q:
            problems.append({"id": pid, "source": "synthFP", "question": q})

    raw_js = DATA_DIR / "external" / "science_olympiad_raw.js"
    _fetch(SCIOLY_URL, raw_js)
    for i, rec in enumerate(_parse_scioly_js(raw_js.read_text(encoding="utf-8"))):
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

    eric_path = DATA_DIR / "external" / "eric_fermi_raw.json"
    _fetch(ERIC_URL, eric_path)
    eric_data = _load_json(eric_path)
    questions_obj = eric_data.get("questions", eric_data) if isinstance(eric_data, dict) else {}
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


def main() -> None:
    eval_ids: set[str] = set()
    if EVAL_FILE.exists():
        with EVAL_FILE.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    eval_ids.add(json.loads(line)["id"])
    print(f"Eval IDs excluded: {len(eval_ids)}")

    print("Loading training problems …")
    problems = load_training_problems(eval_ids)
    print(f"Total problems loaded: {len(problems)}")

    # Exact dedup: first occurrence wins
    manifest: dict[str, str] = {}
    seen: dict[str, str] = {}   # normalized text → first id

    for p in problems:
        key = normalize(p["question"])
        if key in seen:
            manifest[p["id"]] = f"duplicate_of:{seen[key]}"
        else:
            seen[key] = p["id"]
            manifest[p["id"]] = "keep"

    kept  = sum(1 for v in manifest.values() if v == "keep")
    dupes = len(manifest) - kept

    # Per-source breakdown
    src_total: dict[str, int] = {}
    src_kept:  dict[str, int] = {}
    for p in problems:
        s = p["source"]
        src_total[s] = src_total.get(s, 0) + 1
        if manifest[p["id"]] == "keep":
            src_kept[s] = src_kept.get(s, 0) + 1

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(manifest, indent=2))

    print(f"\n{'='*50}")
    print(f"Total   : {len(problems)}")
    print(f"Dupes   : {dupes}")
    print(f"Kept    : {kept}")
    print(f"\nPer source:")
    print(f"  {'Source':<25} {'Total':>7} {'Kept':>7} {'Dupes':>7}")
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*7}")
    for src in sorted(src_total):
        t = src_total[src]
        k = src_kept.get(src, 0)
        print(f"  {src:<25} {t:>7} {k:>7} {t-k:>7}")
    print(f"\nManifest → {OUT_FILE}")
    print("Next: run  python scripts/filter_science_domain.py")


if __name__ == "__main__":
    main()
