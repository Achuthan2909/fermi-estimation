#!/usr/bin/env python3
"""
Combine all dataset sources into a single deduplicated master set.

Sources
-------
datasets/Achuthan-Dataset/          JSONL  (train/val/eval splits)
datasets/eval/                      JSON   (eval benchmark + earlier splits)
datasets/train/                     JSON   (filtered + synthetic problems)

Unified schema
--------------
Required:
  question       str    — the Fermi problem text
  fermi_answer   int    — log10 order-of-magnitude answer
  domain         str    — subject domain
  source         str    — provenance tag (see SOURCE TAGS below)

Optional (preserved when present):
  answer_raw     str    — raw numeric answer
  answer_log10   float  — precise log10 value
  id             str    — original record ID
  unit           str    — unit string (Achuthan-Dataset only)
  answer_type    str    — 'numeric' | 'oom_exponent' (Achuthan-Dataset)
  has_distractor bool   — (eval_set_v2 only)
  length_tier    str    — (eval_set_v2 only)
  relevance      str    — (filtered_5 / synth_2 only)
  program        str    — (filtered_5 / synth_2 / Achuthan train only)
  context        str    — (filtered_5 / synth_2 / Achuthan train only)
  file_source_url str   — URL/label for where the question originally came from

SOURCE TAGS
-----------
  achuthan:train          datasets/Achuthan-Dataset/train.jsonl
  achuthan:val            datasets/Achuthan-Dataset/val.jsonl
  achuthan:eval           datasets/Achuthan-Dataset/eval.jsonl  ← sealed eval
  eval_dir:eval_benchmark datasets/eval/eval_set_v2.json        ← sealed eval
  eval_dir:train          datasets/eval/train.json
  eval_dir:val            datasets/eval/val.json
  eval_dir:test           datasets/eval/test.json
  train_dir:filtered:<f>  datasets/train/fermi_problems_filtered_*.json
  train_dir:synth:<f>     datasets/train/fermi_problems_synth_*.json

Dedup key: normalize(question) — lowercase, strip punctuation, collapse
           whitespace (same normalization as scripts/deduplicate.py).
           First-occurrence wins; loading order below determines priority.

Output
------
  datasets/combined_deduped/master.json
  datasets/combined_deduped/dedup_report.txt
  datasets/combined_deduped/duplicate_log.json
"""

import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
ACHUTHAN_DIR = BASE_DIR / "datasets" / "Achuthan-Dataset"
EVAL_DIR     = BASE_DIR / "datasets" / "eval"
TRAIN_DIR    = BASE_DIR / "datasets" / "train"
OUT_DIR      = BASE_DIR / "datasets" / "combined_deduped"

OPTIONAL_FIELDS = (
    "answer_raw", "answer_log10", "id", "unit", "answer_type",
    "has_distractor", "length_tier", "relevance", "program", "context",
    "file_source_url",
)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _make_rec(question: str, fermi_answer, domain: str, source_tag: str,
              extras: dict) -> dict:
    rec = {
        "question":     question.strip(),
        "fermi_answer": fermi_answer,
        "domain":       domain,
        "source":       source_tag,
    }
    for field in OPTIONAL_FIELDS:
        if extras.get(field) is not None:
            rec[field] = extras[field]
    return rec


def load_achuthan_jsonl(path: Path, source_tag: str) -> list[dict]:
    """JSONL: {id, question, gold_numeric, gold_oom, answer_type, unit, ...}"""
    records = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        q = json.loads(line)
        question = q.get("question", "").strip()
        if not question:
            continue

        fermi_answer = q.get("gold_oom")        # log10 int
        gold_numeric = q.get("gold_numeric")     # raw float

        extras = {}
        extras["id"]          = q.get("id")
        extras["unit"]        = q.get("unit")
        extras["answer_type"] = q.get("answer_type")
        if gold_numeric is not None:
            extras["answer_raw"] = str(gold_numeric)
            try:
                if gold_numeric > 0:
                    extras["answer_log10"] = round(math.log10(gold_numeric), 6)
            except (ValueError, TypeError):
                pass
        if q.get("program"):
            extras["program"] = q["program"]
        if q.get("context"):
            extras["context"] = q["context"]
        # orig_source as file_source_url when present
        if q.get("orig_source"):
            extras["file_source_url"] = q["orig_source"]

        records.append(_make_rec(question, fermi_answer, q.get("domain", ""), source_tag, extras))
    return records


def load_science_domain_file(path: Path, source_tag: str) -> list[dict]:
    """JSON: {source?, science_domain_questions: [{question, fermi_answer, domain, ...}]}"""
    data = json.loads(path.read_text(encoding="utf-8"))
    file_url = data.get("source", "")
    records = []
    for q in data.get("science_domain_questions", []):
        question = q.get("question", "").strip()
        if not question:
            continue
        extras = {}
        for field in ("answer_raw", "answer_log10", "id", "has_distractor",
                      "length_tier", "relevance", "program", "context"):
            if q.get(field) is not None:
                extras[field] = q[field]
        # per-question source overrides file-level url
        item_src = q.get("source", file_url)
        if item_src:
            extras["file_source_url"] = item_src
        records.append(_make_rec(question, q.get("fermi_answer"), q.get("domain", ""), source_tag, extras))
    return records


def load_split_json(path: Path, source_tag: str) -> list[dict]:
    """JSON: {dataset, count, questions: [{question, fermi_answer, domain, source?}]}"""
    data = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for q in data.get("questions", []):
        question = q.get("question", "").strip()
        if not question:
            continue
        extras = {}
        for field in ("answer_raw", "answer_log10", "id", "has_distractor",
                      "length_tier", "relevance", "program", "context"):
            if q.get(field) is not None:
                extras[field] = q[field]
        if q.get("source"):
            extras["file_source_url"] = q["source"]
        records.append(_make_rec(question, q.get("fermi_answer"), q.get("domain", ""), source_tag, extras))
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load — order matters: first occurrence wins in dedup.
    #    Train sources load first so train/val from eval_dir and
    #    achuthan don't crowd out synthesized data.
    #    Sealed eval sets load last so we can detect contamination.
    # ------------------------------------------------------------------
    all_records: list[dict] = []
    file_stats: list[dict] = []

    def add(recs: list[dict], path: Path, tag: str) -> None:
        file_stats.append({
            "file":       str(path.relative_to(BASE_DIR)),
            "source_tag": tag,
            "count":      len(recs),
        })
        all_records.extend(recs)

    # ── datasets/train/ ──────────────────────────────────────────────
    for fpath in sorted(TRAIN_DIR.glob("fermi_problems_filtered_*.json")):
        tag = f"train_dir:filtered:{fpath.name}"
        add(load_science_domain_file(fpath, tag), fpath, tag)

    for fpath in sorted(TRAIN_DIR.glob("fermi_problems_synth_*.json")):
        tag = f"train_dir:synth:{fpath.name}"
        add(load_science_domain_file(fpath, tag), fpath, tag)

    # ── datasets/eval/ train/val/test splits ─────────────────────────
    for split_name, tag in [
        ("train.json", "eval_dir:train"),
        ("val.json",   "eval_dir:val"),
        ("test.json",  "eval_dir:test"),
    ]:
        fpath = EVAL_DIR / split_name
        if fpath.exists():
            add(load_split_json(fpath, tag), fpath, tag)

    # ── datasets/Achuthan-Dataset/ train ─────────────────────────────
    fpath = ACHUTHAN_DIR / "train.jsonl"
    if fpath.exists():
        add(load_achuthan_jsonl(fpath, "achuthan:train"), fpath, "achuthan:train")

    # ── datasets/Achuthan-Dataset/ val ───────────────────────────────
    fpath = ACHUTHAN_DIR / "val.jsonl"
    if fpath.exists():
        add(load_achuthan_jsonl(fpath, "achuthan:val"), fpath, "achuthan:val")

    # ── Sealed eval sets — load last to detect contamination ─────────
    fpath = EVAL_DIR / "eval_set_v2.json"
    if fpath.exists():
        add(load_science_domain_file(fpath, "eval_dir:eval_benchmark"), fpath, "eval_dir:eval_benchmark")

    fpath = ACHUTHAN_DIR / "eval.jsonl"
    if fpath.exists():
        add(load_achuthan_jsonl(fpath, "achuthan:eval"), fpath, "achuthan:eval")

    total_before = len(all_records)

    # ------------------------------------------------------------------
    # 2. Deduplicate — first occurrence wins; track provenance
    # ------------------------------------------------------------------
    seen: dict[str, int] = {}       # normalized text → index in kept_records
    kept_records: list[dict] = []
    dup_log: list[dict] = []

    src_total: dict[str, int] = defaultdict(int)
    src_kept:  dict[str, int] = defaultdict(int)

    for rec in all_records:
        src = rec["source"]
        src_total[src] += 1
        key = normalize(rec["question"])
        if key in seen:
            orig_src = kept_records[seen[key]]["source"]
            dup_log.append({
                "dup_question": rec["question"][:120],
                "dup_source":   src,
                "orig_source":  orig_src,
                "cross_source": orig_src != src,
            })
        else:
            seen[key] = len(kept_records)
            kept_records.append(rec)
            src_kept[src] += 1

    total_after = len(kept_records)
    total_dupes = total_before - total_after

    cross_dupes   = sum(1 for d in dup_log if d["cross_source"])
    within_dupes  = total_dupes - cross_dupes

    # Per-source breakdown on cross-source duplicates (who lost to whom)
    cross_dup_by_src: dict[str, int] = defaultdict(int)
    for d in dup_log:
        if d["cross_source"]:
            cross_dup_by_src[d["dup_source"]] += 1

    # ------------------------------------------------------------------
    # 3. Contamination check: eval questions already in non-eval sources
    # ------------------------------------------------------------------
    EVAL_TAGS = {"eval_dir:eval_benchmark", "achuthan:eval"}
    contaminated = []   # eval questions whose first occurrence is a train source
    for d in dup_log:
        if d["dup_source"] in EVAL_TAGS and d["orig_source"] not in EVAL_TAGS:
            contaminated.append(d)

    # ------------------------------------------------------------------
    # 4. Save outputs
    # ------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "master.json").write_text(
        json.dumps(kept_records, indent=2, ensure_ascii=False))
    (OUT_DIR / "duplicate_log.json").write_text(
        json.dumps(dup_log, indent=2, ensure_ascii=False))

    # ------------------------------------------------------------------
    # 5. Report
    # ------------------------------------------------------------------
    W = 62
    sep  = "─" * W
    dsep = "═" * W

    lines = []
    def h(title):
        lines.append(sep)
        lines.append(title)
        lines.append(sep)

    lines.append(dsep)
    lines.append("DATASET COMBINATION & DEDUPLICATION REPORT")
    lines.append(dsep)
    lines.append("")

    h("FILE INVENTORY (load order = dedup priority)")
    lines.append(f"  {'File':<48} {'Count':>5}")
    lines.append(f"  {'─'*48} {'─'*5}")
    for fs in file_stats:
        lines.append(f"  {fs['file']:<48} {fs['count']:>5}")
    lines.append(f"  {'─'*48} {'─'*5}")
    lines.append(f"  {'TOTAL':<48} {total_before:>5}")
    lines.append("")

    h("DEDUPLICATION SUMMARY")
    lines.append(f"  Total samples before dedup   : {total_before:>5}")
    lines.append(f"  Duplicates removed           : {total_dupes:>5}")
    lines.append(f"    — within-source            : {within_dupes:>5}")
    lines.append(f"    — cross-source             : {cross_dupes:>5}")
    lines.append(f"  Unique samples after dedup   : {total_after:>5}")
    lines.append("")

    h("PER-SOURCE BREAKDOWN")
    lines.append(f"  {'Source':<40} {'Total':>5} {'Kept':>5} {'Lost':>5}")
    lines.append(f"  {'─'*40} {'─'*5} {'─'*5} {'─'*5}")
    for src in sorted(src_total.keys()):
        t = src_total[src]
        k = src_kept.get(src, 0)
        lines.append(f"  {src:<40} {t:>5} {k:>5} {t-k:>5}")
    lines.append("")

    h("SCHEMA NORMALIZATION")
    lines.append("  Required fields (all records):")
    lines.append("    question, fermi_answer, domain, source")
    lines.append("")
    lines.append("  Key field mappings:")
    lines.append("    Achuthan-Dataset  gold_oom     → fermi_answer")
    lines.append("    Achuthan-Dataset  gold_numeric → answer_raw (str)")
    lines.append("    Achuthan-Dataset  gold_numeric → answer_log10 (computed)")
    lines.append("    eval/train dirs   fermi_answer → fermi_answer (unchanged)")
    lines.append("")
    lines.append("  Optional fields preserved when present:")
    lines.append("    answer_raw, answer_log10, id, unit, answer_type,")
    lines.append("    has_distractor, length_tier, relevance, program,")
    lines.append("    context, file_source_url")
    lines.append("")

    h("PROVENANCE GUIDE (source column values)")
    lines.append("  achuthan:train         → Achuthan-Dataset/train.jsonl")
    lines.append("  achuthan:val           → Achuthan-Dataset/val.jsonl")
    lines.append("  achuthan:eval          → Achuthan-Dataset/eval.jsonl  [EVAL]")
    lines.append("  eval_dir:eval_benchmark→ eval/eval_set_v2.json        [EVAL]")
    lines.append("  eval_dir:train         → eval/train.json")
    lines.append("  eval_dir:val           → eval/val.json")
    lines.append("  eval_dir:test          → eval/test.json")
    lines.append("  train_dir:filtered:*   → train/fermi_problems_filtered_*.json")
    lines.append("  train_dir:synth:*      → train/fermi_problems_synth_*.json")
    lines.append("")
    lines.append("  [EVAL] = sealed/held-out — must not appear in train/val splits")
    lines.append("")

    h("EVAL CONTAMINATION CHECK")
    if contaminated:
        lines.append(f"  WARNING: {len(contaminated)} eval question(s) are ALREADY")
        lines.append("  present in train-tagged records (first-occurrence wins,")
        lines.append("  so these appear in master.json tagged as train sources).")
        lines.append("")
        lines.append("  These questions are in master.json as train-source records")
        lines.append("  but were intended for eval. Action required before splitting.")
        lines.append("")
        for d in contaminated:
            lines.append(f"  dup_src : {d['dup_source']}")
            lines.append(f"  orig_src: {d['orig_source']}")
            lines.append(f"  q: {d['dup_question'][:80]}")
            lines.append("")
    else:
        lines.append("  CLEAN — no eval questions found in train-tagged records.")
        lines.append("  (Eval sets were loaded last; any overlap was dropped from")
        lines.append("  eval and the train version is the one kept.)")
        lines.append("")
        lines.append("  If 'Lost' > 0 for achuthan:eval or eval_dir:eval_benchmark,")
        lines.append("  those questions already exist in train records — they will")
        lines.append("  NOT appear in master.json with an eval source tag. Ensure")
        lines.append("  you exclude them from training when you create splits.")
        lines.append("")
        # Show which eval questions were dropped (absorbed by train)
        absorbed = [d for d in dup_log if d["dup_source"] in EVAL_TAGS]
        if absorbed:
            lines.append(f"  {len(absorbed)} eval question(s) absorbed by earlier train records:")
            for d in absorbed:
                lines.append(f"    orig_src: {d['orig_source']}")
                lines.append(f"    q: {d['dup_question'][:80]}")
    lines.append("")

    lines.append(dsep)
    lines.append("OUTPUT")
    lines.append(dsep)
    lines.append(f"  datasets/combined_deduped/master.json      ({total_after} records)")
    lines.append(f"  datasets/combined_deduped/duplicate_log.json ({total_dupes} entries)")
    lines.append(f"  datasets/combined_deduped/dedup_report.txt")
    lines.append(dsep)

    report = "\n".join(lines)
    (OUT_DIR / "dedup_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
