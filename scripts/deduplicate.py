#!/usr/bin/env python3
"""
Deduplication pass — run this BEFORE filter_science_domain.py.

Loads all training problems from all three sources (excluding eval IDs),
detects near-duplicates using TF-IDF cosine similarity, and writes a
manifest marking each problem as kept or duplicate.

filter_science_domain.py reads this manifest and skips duplicates,
avoiding wasted API calls on redundant problems.

Algorithm:
  1. Exact dedup  — normalize whitespace/case, exact string match
  2. Near dedup   — TF-IDF cosine similarity > THRESHOLD (default 0.85)
     Greedy: first occurrence in the merged problem list wins.

Outputs:
  data/filtered/dedup_manifest.json
    {id: "keep"} or {id: "duplicate_of:<original_id>"}

  Prints a report: total problems, exact dupes, near dupes, kept count,
  per-source breakdown.
"""

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data"
EVAL_FILE  = BASE_DIR / "eval" / "eval_benchmark.jsonl"
OUT_FILE   = DATA_DIR / "filtered" / "dedup_manifest.json"

REALFP_BASE  = "https://raw.githubusercontent.com/allenai/fermi/master/data/realFP"
SYNTHFP_BASE = "https://raw.githubusercontent.com/allenai/fermi/master/data/synthFP"
SCIOLY_URL   = "https://raw.githubusercontent.com/landy8697/open-scioly-fermi/master/data.js"
ERIC_URL     = "https://raw.githubusercontent.com/EricAndrechek/FermiQuestions/main/question-bank.json"

THRESHOLD = 0.85   # cosine similarity above which two problems are near-duplicates
BATCH     = 500    # problems per batch for similarity computation


# ---------------------------------------------------------------------------
# Helpers (mirrors build_eval_benchmark / filter_science_domain loaders)
# ---------------------------------------------------------------------------

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


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load eval IDs
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
    print(f"Total problems: {len(problems)}")

    ids       = [p["id"]       for p in problems]
    questions = [p["question"] for p in problems]
    normed    = [normalize(q)  for q in questions]

    manifest: dict[str, str] = {}   # id → "keep" or "duplicate_of:<id>"

    # ---- Pass 1: exact dedup ----
    seen_exact: dict[str, str] = {}   # normalized text → first id
    exact_dup = 0
    for pid, norm in zip(ids, normed):
        if norm in seen_exact:
            manifest[pid] = f"duplicate_of:{seen_exact[norm]}"
            exact_dup += 1
        else:
            seen_exact[norm] = pid

    print(f"Exact duplicates: {exact_dup}")

    # ---- Pass 2: near-dedup on non-exact-dup problems ----
    keep_idx  = [i for i, pid in enumerate(ids) if pid not in manifest]
    keep_ids  = [ids[i]   for i in keep_idx]
    keep_text = [normed[i] for i in keep_idx]

    print(f"Running TF-IDF near-dedup on {len(keep_ids)} problems (threshold={THRESHOLD}) …")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    tfidf = vectorizer.fit_transform(keep_text)

    # Greedy: mark later occurrences as duplicates if similarity > threshold
    # Process in batches to avoid memory blowup
    dup_set: set[int] = set()   # indices into keep_ids already marked as dup
    near_dup = 0

    for batch_start in range(0, len(keep_ids), BATCH):
        batch_end = min(batch_start + BATCH, len(keep_ids))
        batch_vec = tfidf[batch_start:batch_end]

        # Compare batch against all following problems (upper triangle only)
        sims = cosine_similarity(batch_vec, tfidf[batch_start:])  # shape: (batch, rest)

        for bi in range(sims.shape[0]):
            gi = batch_start + bi   # global index
            if gi in dup_set:
                continue
            # Find near-duplicate candidates AFTER this problem
            for gj in range(gi + 1, len(keep_ids)):
                if gj in dup_set:
                    continue
                sim_idx = gj - batch_start
                if 0 <= sim_idx < sims.shape[1] and sims[bi, sim_idx] > THRESHOLD:
                    dup_set.add(gj)
                    manifest[keep_ids[gj]] = f"duplicate_of:{keep_ids[gi]}"
                    near_dup += 1

        pct = round(100 * batch_end / len(keep_ids))
        print(f"  {batch_end}/{len(keep_ids)} ({pct}%)  near-dups so far: {near_dup}", end="\r")

    print()

    # Mark all remaining as keep
    for pid in ids:
        if pid not in manifest:
            manifest[pid] = "keep"

    # ---- Write manifest ----
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(manifest, indent=2))

    # ---- Summary ----
    kept  = sum(1 for v in manifest.values() if v == "keep")
    dupes = len(manifest) - kept

    print(f"\n{'='*50}")
    print(f"Total problems : {len(problems)}")
    print(f"Exact dupes    : {exact_dup}")
    print(f"Near dupes     : {near_dup}")
    print(f"Total dupes    : {dupes}  ({round(100*dupes/len(problems))}%)")
    print(f"Kept           : {kept}")

    # Per-source
    src_total: dict[str, int] = {}
    src_kept:  dict[str, int] = {}
    for p in problems:
        s = p["source"]
        src_total[s] = src_total.get(s, 0) + 1
        if manifest[p["id"]] == "keep":
            src_kept[s] = src_kept.get(s, 0) + 1

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
