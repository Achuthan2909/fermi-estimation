"""Unify, dedupe, and (optionally) filter raw Fermi datasets.

Sources:
  - RealFP       datasets/raw/fermi/data/realFP/{train,val,test}_realfp.json
  - SynthFP      datasets/raw/fermi/data/synthFP/{train,val,test}_synthfp.json
  - FermiQuest.  datasets/raw/FermiQuestions/question-bank.json
  - scioly       datasets/raw/open-scioly-fermi/formatted_test_data/questions.json

Output schema (one JSON object per line in unified.jsonl):
  {
    "id": "<source>:<orig_split|bucket>:<idx>",
    "question": str,
    "gold_numeric": float | None,     # known numeric answer (RealFP, SynthFP)
    "gold_oom": int,                  # base-10 exponent (floor(log10(numeric)) if numeric known)
    "answer_type": "numeric" | "oom_exponent",
    "unit": str | None,               # only RealFP carries units
    "source": "realFP" | "synthFP" | "scioly" | "fermiq",
    "orig_split": str | None,         # train/val/test for fermi sources
    "orig_source": str | None,        # sub-source for scioly/fermiq (e.g. 'BirdSO 20-21')
    "program": str | None,            # RealFP/SynthFP only
    "context": str | None             # RealFP/SynthFP only
  }
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "datasets" / "raw"
PROCESSED = REPO / "datasets" / "processed"

OOM_MIN, OOM_MAX = -30, 40  # clip range; anything outside is almost certainly dirty
MIN_Q_CHARS = 20            # strip obvious non-Fermi trivia ("What is pi?")

# Source priority (lower = kept on dedup)
SOURCE_PRIORITY = {"realFP": 0, "scioly": 1, "fermiq": 2, "synthFP": 3}
# Within a source, prefer test > val > train so eval-split leaks don't silently poison training.
SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2, None: 3}


@dataclass
class Item:
    id: str
    question: str
    gold_numeric: Optional[float]
    gold_oom: int
    answer_type: str
    unit: Optional[str]
    source: str
    orig_split: Optional[str] = None
    orig_source: Optional[str] = None
    program: Optional[str] = None
    context: Optional[str] = None


# ---------- answer parsing ----------

_REALFP_ANS_RE = re.compile(
    r"""
    ^\s*
    (?P<mant>[-+]?\d+(?:\.\d+)?)       # mantissa
    (?:[eE](?P<exp>[-+]?\d+))?         # optional exponent
    (?:\s*(?P<unit>\S.*?))?            # optional unit (rest of string)
    \s*$
    """,
    re.VERBOSE,
)


def parse_scientific(s: str) -> tuple[Optional[float], Optional[str]]:
    """Parse strings like '2.3e+8 in**2' -> (2.3e8, 'in**2')."""
    s = s.strip()
    m = _REALFP_ANS_RE.match(s)
    if not m:
        return None, None
    mant = float(m.group("mant"))
    exp = int(m.group("exp")) if m.group("exp") is not None else 0
    unit = m.group("unit")
    try:
        val = mant * (10 ** exp)
    except OverflowError:
        return None, unit
    return val, unit


def oom_from_numeric(x: float) -> Optional[int]:
    if x is None or x <= 0 or not math.isfinite(x):
        return None
    return int(math.floor(math.log10(x)))


# ---------- loaders ----------

def load_realfp() -> list[Item]:
    out: list[Item] = []
    for split in ("train", "val", "test"):
        path = RAW / "fermi" / "data" / "realFP" / f"{split}_realfp.json"
        raw = json.loads(path.read_text())
        for i, row in enumerate(raw):
            q = (row.get("question") or "").strip()
            ans_str = (row.get("answer") or "").strip()
            numeric, unit = parse_scientific(ans_str)
            if numeric is None:
                continue
            oom = oom_from_numeric(numeric)
            if oom is None:
                continue
            out.append(Item(
                id=f"realFP:{split}:{i}",
                question=q,
                gold_numeric=numeric,
                gold_oom=oom,
                answer_type="numeric",
                unit=unit,
                source="realFP",
                orig_split=split,
                program=row.get("program"),
                context=row.get("context"),
            ))
    return out


def load_synthfp() -> list[Item]:
    out: list[Item] = []
    for split in ("train", "val", "test"):
        path = RAW / "fermi" / "data" / "synthFP" / f"{split}_synthfp.json"
        raw = json.loads(path.read_text())
        for i, row in enumerate(raw):
            q = (row.get("question") or "").strip()
            ans_str = (row.get("answer") or "").strip()
            numeric, _ = parse_scientific(ans_str)  # synthFP has no units
            if numeric is None:
                continue
            oom = oom_from_numeric(numeric)
            if oom is None:
                continue
            out.append(Item(
                id=f"synthFP:{split}:{i}",
                question=q,
                gold_numeric=numeric,
                gold_oom=oom,
                answer_type="numeric",
                unit=None,
                source="synthFP",
                orig_split=split,
                program=row.get("program") or row.get("program_simple"),
                context=row.get("context"),
            ))
    return out


def load_fermiq() -> list[Item]:
    path = RAW / "FermiQuestions" / "question-bank.json"
    data = json.loads(path.read_text())
    out: list[Item] = []
    for bucket, qmap in data["questions"].items():
        for i, (q, ans) in enumerate(qmap.items()):
            q = q.strip()
            try:
                oom = int(ans)
            except (TypeError, ValueError):
                continue
            if not (OOM_MIN <= oom <= OOM_MAX):
                continue
            # strip HTML like <sup>43</sup>
            q_clean = re.sub(r"<[^>]+>", "", q).strip()
            if len(q_clean) < MIN_Q_CHARS:
                continue
            out.append(Item(
                id=f"fermiq:{bucket}:{i}",
                question=q_clean,
                gold_numeric=None,
                gold_oom=oom,
                answer_type="oom_exponent",
                unit=None,
                source="fermiq",
                orig_source=bucket,
            ))
    return out


def load_scioly() -> list[Item]:
    path = RAW / "open-scioly-fermi" / "formatted_test_data" / "questions.json"
    raw = json.loads(path.read_text())
    out: list[Item] = []
    for i, row in enumerate(raw):
        q = (row.get("question") or "").strip()
        oom = row.get("answer")
        if not isinstance(oom, int):
            continue
        if not (OOM_MIN <= oom <= OOM_MAX):
            continue
        if len(q) < MIN_Q_CHARS:
            continue
        out.append(Item(
            id=f"scioly:{i}",
            question=q,
            gold_numeric=None,
            gold_oom=oom,
            answer_type="oom_exponent",
            unit=None,
            source="scioly",
            orig_source=row.get("source"),
        ))
    return out


# ---------- dedupe ----------

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def norm_key(q: str) -> str:
    return _NORMALIZE_RE.sub(" ", q.lower()).strip()


def _priority(it: Item) -> tuple[int, int]:
    return (SOURCE_PRIORITY[it.source], SPLIT_PRIORITY.get(it.orig_split, 3))


def dedupe(items: list[Item]) -> tuple[list[Item], dict[str, int]]:
    """Keep the highest-priority item per normalized-question hash.

    Priority is (source, split) — so within a source, test beats val beats train.
    This preserves eval-set items when the same question appears in multiple splits.
    """
    by_key: dict[str, Item] = {}
    intra = 0  # duplicates within same source
    cross = 0  # duplicates across sources
    for it in items:
        k = norm_key(it.question)
        if not k:
            intra += 1
            continue
        cur = by_key.get(k)
        if cur is None:
            by_key[k] = it
            continue
        if _priority(it) < _priority(cur):
            # new wins
            if cur.source == it.source:
                intra += 1
            else:
                cross += 1
            by_key[k] = it
        else:
            if cur.source == it.source:
                intra += 1
            else:
                cross += 1
    return list(by_key.values()), {"intra_source": intra, "cross_source": cross}


# ---------- writers ----------

def write_jsonl(path: Path, items: Iterable[Item]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
            n += 1
    return n


def counts_by_source(items: list[Item]) -> dict[str, int]:
    c: dict[str, int] = {}
    for it in items:
        c[it.source] = c.get(it.source, 0) + 1
    return c


def main() -> None:
    print("Loading sources...")
    realfp = load_realfp()
    synthfp = load_synthfp()
    fermiq = load_fermiq()
    scioly = load_scioly()

    pre_counts = {
        "realFP": len(realfp),
        "synthFP": len(synthfp),
        "fermiq": len(fermiq),
        "scioly": len(scioly),
    }
    print(f"Raw (post-parse/clip) counts: {pre_counts}  total={sum(pre_counts.values())}")

    all_items = realfp + synthfp + fermiq + scioly
    deduped, dropped = dedupe(all_items)
    print(f"Dedup: kept {len(deduped)}, dropped intra={dropped['intra_source']} cross={dropped['cross_source']}")

    post_counts = counts_by_source(deduped)
    print(f"Post-dedup counts: {post_counts}")

    # Split breakdown for fermi sources (realFP/synthFP only)
    split_counts: dict[str, dict[str, int]] = {}
    for it in deduped:
        if it.orig_split:
            split_counts.setdefault(it.source, {}).setdefault(it.orig_split, 0)
            split_counts[it.source][it.orig_split] += 1

    out_path = PROCESSED / "unified.jsonl"
    n = write_jsonl(out_path, deduped)
    print(f"Wrote {n} items -> {out_path.relative_to(REPO)}")

    report = {
        "pre_dedup_counts": pre_counts,
        "pre_dedup_total": sum(pre_counts.values()),
        "post_dedup_counts": post_counts,
        "post_dedup_total": len(deduped),
        "dedup_dropped": dropped,
        "split_counts_after_dedup": split_counts,
        "source_priority_on_dedup": SOURCE_PRIORITY,
        "split_priority_on_dedup": {str(k): v for k, v in SPLIT_PRIORITY.items()},
        "oom_clip": [OOM_MIN, OOM_MAX],
        "min_question_chars": MIN_Q_CHARS,
    }
    report_path = PROCESSED / "unified_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote report -> {report_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
