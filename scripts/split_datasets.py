#!/usr/bin/env python3
"""
Split datasets/combined_deduped/master.json → train / val / eval.

Target proportions (of 5008 total):
  eval  20% ≈ 1002   (curated first; see construction below)
  train 50% ≈ 2504   (stratified from remainder)
  val   30% ≈ 1502   (stratified from remainder)

Eval construction
─────────────────
1. Mandatory items
   a. 46 contaminated questions  — questions that existed in a sealed eval
      source (eval_dir:eval_benchmark or achuthan:eval) but appear in
      master.json under a train source tag (first-occurrence wins).
      They must be excluded from training; we route them to eval.
   b. All has_distractor=True records (5 records, overlap-checked).
   c. 3 synthetic wrong-assumption questions created here (tagged
      source="synthetic:wrong_assumption").
2. Stratified fill to ≈ 20 % of total, sampling proportionally across
   complexity × length_tier strata from the non-mandatory pool.

Train / Val split
─────────────────
Stratify on domain_group × complexity × length_tier.
Within each stratum split 62.5 % train / 37.5 % val
(= 50 % / 30 % of total, given eval is already 20 %).

Complexity proxy (no explicit labels in data)
──────────────────────────────────────────────
  easy   : word_count ≤ 12  AND  0 ≤ fermi_answer ≤ 8
  hard   : word_count > 35  OR   fermi_answer > 20  OR  fermi_answer < -2
  medium : everything else
"""

import json
import math
import random
import re
import unicodedata
from collections import defaultdict, Counter
from pathlib import Path

BASE_DIR   = Path(__file__).parent.parent
MASTER     = BASE_DIR / "datasets" / "combined_deduped" / "master.json"
DUP_LOG    = BASE_DIR / "datasets" / "combined_deduped" / "duplicate_log.json"
OUT_DIR    = BASE_DIR / "datasets" / "final"

EVAL_SOURCE_TAGS = {"eval_dir:eval_benchmark", "achuthan:eval"}
SEED = 42


# ── helpers ─────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


DOMAIN_MAP: list[tuple[list[str], str]] = [
    (["physics", "optics", "mechanics", "thermodynamics", "electro",
      "quantum", "nuclear", "materials", "engineering", "acoustics"], "physics"),
    (["biology", "medicine", "anatomy", "physiology", "genetics",
      "ecology", "evolution", "microbio", "cell", "biochem"], "biology"),
    (["chemistry", "chemical", "organic", "polymer"], "chemistry"),
    (["astro", "cosmo", "space", "planet", "stellar", "galaxy"], "astronomy"),
    (["earth", "geo", "climate", "ocean", "atmo", "weather",
      "hydro", "soil"], "earth_science"),
    (["math", "stat", "numer", "calculus", "algebra", "combinat",
      "probability", "logic"], "mathematics"),
]

def domain_group(domain: str) -> str:
    d = domain.lower()
    for tokens, group in DOMAIN_MAP:
        if any(t in d for t in tokens):
            return group
    return "other_science"


def length_tier(wc: int) -> str:
    if wc <= 12:
        return "short"
    if wc <= 35:
        return "medium"
    return "long"


def complexity(wc: int, fa) -> str:
    if fa is None:
        fa = 5  # neutral default
    if wc <= 12 and 0 <= fa <= 8:
        return "easy"
    if wc > 35 or fa > 20 or fa < -2:
        return "hard"
    return "medium"


def annotate(rec: dict) -> dict:
    """Add derived fields in-place; return rec."""
    wc = len(rec["question"].split())
    fa = rec.get("fermi_answer")
    rec["_wc"]         = wc
    rec["_lt"]         = length_tier(wc)
    rec["_complexity"] = complexity(wc, fa)
    rec["_dg"]         = domain_group(rec.get("domain", ""))
    return rec


def strat_key(rec: dict) -> str:
    return f"{rec['_dg']}|{rec['_complexity']}|{rec['_lt']}"


def stratified_split(pool: list[dict], frac_a: float, rng: random.Random
                     ) -> tuple[list[dict], list[dict]]:
    """Split pool into two parts at frac_a / (1-frac_a) stratified by strat_key."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        buckets[strat_key(r)].append(r)
    a, b = [], []
    for recs in buckets.values():
        rng.shuffle(recs)
        cut = round(len(recs) * frac_a)
        a.extend(recs[:cut])
        b.extend(recs[cut:])
    return a, b


def strip_private(rec: dict) -> dict:
    """Remove internal _* fields before saving."""
    return {k: v for k, v in rec.items() if not k.startswith("_")}


# ── synthetic wrong-assumption questions ────────────────────────────────────

SYNTHETIC_QS: list[dict] = [
    {
        # WA-1 — wrong per-person wheat acreage (off by ~2.5 OOM)
        "question": (
            "How many acres of land would need to be devoted to cultivating wheat "
            "to sustain a population of one million people for a year? "
            "Assume it takes approximately 100 acres of wheat to feed a single "
            "person annually."
        ),
        "fermi_answer": 6,          # correct answer (~3e5 acres, fa=5.5≈6)
        "domain": "earth_science",
        "source": "synthetic:wrong_assumption",
        "answer_log10": 5.48,
        "answer_raw": "3e5",
        "synthetic_note": (
            "Wrong assumption: 100 acres/person. Reality ≈ 0.3 acres/person "
            "(off by ~333×, ~2.5 OOM). A model following the stated assumption "
            "would answer fa≈8 instead of fa≈6."
        ),
        "wrong_assumption": "100 acres of wheat per person per year",
        "correct_assumption": "≈ 0.3 acres of wheat per person per year",
    },
    {
        # WA-2 — wrong speed of light (off by 100×, 2 OOM)
        "question": (
            "How fast is the speed of light in centimeters per hour? "
            "Assume the speed of light is approximately 3,000 km/s."
        ),
        "fermi_answer": 14,         # correct answer (≈1.08e14 cm/hr, fa=14)
        "domain": "physics",
        "source": "synthetic:wrong_assumption",
        "answer_log10": 14.03,
        "answer_raw": "1.08e14",
        "synthetic_note": (
            "Wrong assumption: c = 3,000 km/s. Reality c ≈ 300,000 km/s "
            "(off by 100×, 2 OOM). A model following the stated assumption "
            "would answer fa≈12 instead of fa≈14."
        ),
        "wrong_assumption": "speed of light = 3,000 km/s",
        "correct_assumption": "speed of light ≈ 300,000 km/s (3×10^8 m/s)",
    },
    {
        # WA-3 — wrong Earth radius (off by 10×, volume off by 3 OOM)
        "question": (
            "What is the volume of the Earth in cubic meters? "
            "Assume that Earth's radius is approximately 640 km."
        ),
        "fermi_answer": 21,         # correct answer (≈1.08e21 m³, fa=21)
        "domain": "earth_science",
        "source": "synthetic:wrong_assumption",
        "answer_log10": 21.03,
        "answer_raw": "1.08e21",
        "synthetic_note": (
            "Wrong assumption: Earth radius = 640 km. Reality ≈ 6,371 km "
            "(off by 10×; volume off by 10³ = 3 OOM). A model following the "
            "stated assumption would answer fa≈18 instead of fa≈21."
        ),
        "wrong_assumption": "Earth radius = 640 km",
        "correct_assumption": "Earth radius ≈ 6,371 km",
    },
]


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────
    master   = [annotate(r) for r in json.loads(MASTER.read_text())]
    dup_log  = json.loads(DUP_LOG.read_text())

    total = len(master)

    # ── 2. Identify contaminated records ─────────────────────────────────
    # From duplicate_log: entries where the dup was from a sealed eval source
    # but was absorbed by an earlier train-tagged record.  The question TEXT
    # in the log matches a record in master.json under the train source.
    cont_norms: set[str] = set()
    for d in dup_log:
        if d["dup_source"] in EVAL_SOURCE_TAGS and \
           d["orig_source"] not in EVAL_SOURCE_TAGS:
            cont_norms.add(normalize(d["dup_question"]))

    master_by_norm = {normalize(r["question"]): r for r in master}
    contaminated_records = [
        master_by_norm[n] for n in cont_norms if n in master_by_norm
    ]
    cont_ids = {id(r) for r in contaminated_records}

    # ── 3. Identify has_distractor records ────────────────────────────────
    distractor_records = [r for r in master if r.get("has_distractor")]
    dist_ids = {id(r) for r in distractor_records}

    # ── 4. Annotate synthetic questions ──────────────────────────────────
    for sq in SYNTHETIC_QS:
        wc = len(sq["question"].split())
        fa = sq.get("fermi_answer")
        sq["_wc"]         = wc
        sq["_lt"]         = length_tier(wc)
        sq["_complexity"] = complexity(wc, fa)
        sq["_dg"]         = domain_group(sq.get("domain", ""))

    # ── 5. Build mandatory eval set ───────────────────────────────────────
    # Merge contaminated + distractor (dedup by id()), then synthetic
    mandatory_ids: set[int] = set()
    mandatory_eval: list[dict] = []
    for r in contaminated_records + distractor_records:
        if id(r) not in mandatory_ids:
            mandatory_ids.add(id(r))
            mandatory_eval.append(r)
    mandatory_eval.extend(SYNTHETIC_QS)   # synthetic have no id() collision

    # ── 6. Remaining (non-mandatory) pool ────────────────────────────────
    remaining_pool = [r for r in master if id(r) not in mandatory_ids]

    # ── 7. Eval stratified fill ───────────────────────────────────────────
    eval_target = 500
    fill_needed = max(0, eval_target - len(mandatory_eval))

    # Sample fill_needed from remaining_pool, stratified by complexity×length_tier
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in remaining_pool:
        buckets[f"{r['_complexity']}|{r['_lt']}"].append(r)

    eval_fill: list[dict] = []
    pool_size = len(remaining_pool)
    for key, recs in buckets.items():
        rng.shuffle(recs)
        # proportional share
        n = round(fill_needed * len(recs) / pool_size)
        eval_fill.extend(recs[:n])

    # top-up or trim to exact fill_needed
    taken_ids = {id(r) for r in eval_fill}
    if len(eval_fill) < fill_needed:
        extras = [r for r in remaining_pool if id(r) not in taken_ids]
        rng.shuffle(extras)
        eval_fill.extend(extras[:fill_needed - len(eval_fill)])
    elif len(eval_fill) > fill_needed:
        eval_fill = eval_fill[:fill_needed]

    eval_fill_ids = {id(r) for r in eval_fill}
    eval_set = mandatory_eval + eval_fill

    # ── 8. Train / val stratified split ──────────────────────────────────
    trainval_pool = [r for r in remaining_pool if id(r) not in eval_fill_ids]

    # 62.5 % of trainval → train  (= 50 % of total)
    # 37.5 % of trainval → val    (= 30 % of total)
    train_set, val_set = stratified_split(trainval_pool, 62.5 / 100, rng)

    # ── 9. Save ───────────────────────────────────────────────────────────
    def save(path: Path, records: list[dict]) -> None:
        clean = [strip_private(r) for r in records]
        path.write_text(json.dumps(clean, indent=2, ensure_ascii=False))

    save(OUT_DIR / "eval.json",  eval_set)
    save(OUT_DIR / "train.json", train_set)
    save(OUT_DIR / "val.json",   val_set)

    # ── 10. Report ────────────────────────────────────────────────────────
    W = 65
    sep  = "─" * W
    dsep = "═" * W

    lines = []
    def section(title: str) -> None:
        lines.append("")
        lines.append(sep)
        lines.append(title)
        lines.append(sep)

    lines.append(dsep)
    lines.append("DATASET SPLIT REPORT")
    lines.append(dsep)

    # ── counts ──
    section("SPLIT COUNTS")
    accounted = len(train_set) + len(val_set) + len(eval_set)
    lines.append(f"  {'Split':<10} {'Count':>6}  {'%':>6}")
    lines.append(f"  {'─'*10} {'─'*6}  {'─'*6}")
    for name, ds in [("train", train_set), ("val", val_set), ("eval", eval_set)]:
        lines.append(f"  {name:<10} {len(ds):>6}  {100*len(ds)/total:>5.1f}%")
    lines.append(f"  {'─'*10} {'─'*6}  {'─'*6}")
    lines.append(f"  {'TOTAL':<10} {accounted:>6}  {100*accounted/total:>5.1f}%")
    lines.append(f"  (master total: {total})")

    # ── eval composition ──
    section("EVAL SET COMPOSITION")
    lines.append(f"  Contaminated questions (mandatory)  : {len(contaminated_records):>4}")
    overlap = len(cont_ids & dist_ids)
    lines.append(f"  Distractor questions (has_distractor): {len(distractor_records):>4}  (overlap with contaminated: {overlap})")
    lines.append(f"  Synthetic wrong-assumption questions :    3")
    lines.append(f"  Stratified fill from non-mandatory  : {len(eval_fill):>4}")
    lines.append(f"  ─────────────────────────────────────────")
    lines.append(f"  Total eval                          : {len(eval_set):>4}")

    # ── complexity distributions ──
    section("COMPLEXITY DISTRIBUTION")
    lines.append(f"  {'Complexity':<10} {'train':>7} {'val':>7} {'eval':>7}")
    lines.append(f"  {'─'*10} {'─'*7} {'─'*7} {'─'*7}")
    for lvl in ("easy", "medium", "hard"):
        nt = sum(1 for r in train_set if r.get("_complexity") == lvl)
        nv = sum(1 for r in val_set   if r.get("_complexity") == lvl)
        ne = sum(1 for r in eval_set  if r.get("_complexity", complexity(r.get("_wc", len(r["question"].split())), r.get("fermi_answer"))) == lvl)
        def pct(n, tot): return f"{100*n/tot:.0f}%" if tot else "-"
        lines.append(f"  {lvl:<10} {nt:>4} {pct(nt,len(train_set)):>3}  {nv:>4} {pct(nv,len(val_set)):>3}  {ne:>4} {pct(ne,len(eval_set)):>3}")

    # ── length tier distributions ──
    section("LENGTH TIER DISTRIBUTION")
    lines.append(f"  {'Length':<10} {'train':>7} {'val':>7} {'eval':>7}")
    lines.append(f"  {'─'*10} {'─'*7} {'─'*7} {'─'*7}")
    for lvl in ("short", "medium", "long"):
        nt = sum(1 for r in train_set if r.get("_lt") == lvl)
        nv = sum(1 for r in val_set   if r.get("_lt") == lvl)
        ne_raw = [r for r in eval_set if r.get("_lt") == lvl]
        ne = len(ne_raw)
        def pct(n, tot): return f"{100*n/tot:.0f}%" if tot else "-"
        lines.append(f"  {lvl:<10} {nt:>4} {pct(nt,len(train_set)):>3}  {nv:>4} {pct(nv,len(val_set)):>3}  {ne:>4} {pct(ne,len(eval_set)):>3}")

    # ── answer_type distributions ──
    section("ANSWER TYPE DISTRIBUTION")
    lines.append(f"  {'Type':<18} {'train':>7} {'val':>7} {'eval':>7}")
    lines.append(f"  {'─'*18} {'─'*7} {'─'*7} {'─'*7}")
    for at in ("numeric", "oom_exponent", "(none)"):
        key = None if at == "(none)" else at
        nt = sum(1 for r in train_set if r.get("answer_type") == key)
        nv = sum(1 for r in val_set   if r.get("answer_type") == key)
        ne = sum(1 for r in eval_set  if r.get("answer_type") == key)
        def pct(n, tot): return f"{100*n/tot:.0f}%" if tot else "-"
        lines.append(f"  {at:<18} {nt:>4} {pct(nt,len(train_set)):>3}  {nv:>4} {pct(nv,len(val_set)):>3}  {ne:>4} {pct(ne,len(eval_set)):>3}")

    # ── domain group distributions ──
    section("DOMAIN GROUP DISTRIBUTION")
    all_dgs = sorted({r.get("_dg","?") for r in master})
    lines.append(f"  {'Domain':<16} {'train':>7} {'val':>7} {'eval':>7}")
    lines.append(f"  {'─'*16} {'─'*7} {'─'*7} {'─'*7}")
    for dg in all_dgs:
        nt = sum(1 for r in train_set if r.get("_dg") == dg)
        nv = sum(1 for r in val_set   if r.get("_dg") == dg)
        ne = sum(1 for r in eval_set  if r.get("_dg") == dg)
        def pct(n, tot): return f"{100*n/tot:.0f}%" if tot else "-"
        lines.append(f"  {dg:<16} {nt:>4} {pct(nt,len(train_set)):>3}  {nv:>4} {pct(nv,len(val_set)):>3}  {ne:>4} {pct(ne,len(eval_set)):>3}")

    # ── special eval items ──
    section("DISTRACTOR QUESTIONS IN EVAL")
    for r in eval_set:
        if r.get("has_distractor") or (r in distractor_records):
            lines.append(f"  [{r.get('_complexity','?'):6}] {r['question'][:90]}")

    section("WRONG-ASSUMPTION QUESTIONS IN EVAL")
    for r in eval_set:
        if r.get("source") == "synthetic:wrong_assumption":
            lines.append(f"  Q  : {r['question'][:100]}")
            lines.append(f"  Note: {r.get('synthetic_note','')[:100]}")
            lines.append(f"  Wrong: {r.get('wrong_assumption','')}")
            lines.append(f"  Correct: {r.get('correct_assumption','')}")
            lines.append(f"  fermi_answer (correct): {r.get('fermi_answer')}")
            lines.append("")

    section("CONTAMINATED QUESTIONS DISPOSITION")
    lines.append(f"  {len(contaminated_records)} questions were originally in a sealed eval source")
    lines.append(f"  but absorbed by train-tagged records in master.json.")
    lines.append(f"  All {len(contaminated_records)} are routed to eval (not train or val).")
    lines.append("")
    lines.append(f"  {'Original master source':<48} {'→ Split'}")
    lines.append(f"  {'─'*48} {'─'*7}")
    src_counts: dict[str, int] = Counter(r["source"] for r in contaminated_records)
    for src, cnt in sorted(src_counts.items()):
        lines.append(f"  {src:<48} eval  ({cnt})")

    # ── source distribution in each split ──
    section("SOURCE TAG DISTRIBUTION (top sources)")
    for split_name, ds in [("train", train_set), ("val", val_set), ("eval", eval_set)]:
        src_c = Counter(r.get("source","?").split(":")[0] for r in ds)
        lines.append(f"  {split_name}:")
        for s, c in src_c.most_common():
            lines.append(f"    {s:<30} {c:>5}  ({100*c/len(ds):.0f}%)")
        lines.append("")

    # ── zero-overlap verification ──
    section("ZERO-OVERLAP VERIFICATION")
    train_norms = {normalize(r["question"]) for r in train_set}
    val_norms   = {normalize(r["question"]) for r in val_set}
    eval_norms  = {normalize(r["question"]) for r in eval_set}
    tv = len(train_norms & val_norms)
    te = len(train_norms & eval_norms)
    ve = len(val_norms   & eval_norms)
    lines.append(f"  train ∩ val  : {tv} overlapping questions")
    lines.append(f"  train ∩ eval : {te} overlapping questions")
    lines.append(f"  val   ∩ eval : {ve} overlapping questions")
    if tv == te == ve == 0:
        lines.append("  ✓ CLEAN — zero overlap across all three splits")
    else:
        lines.append("  ✗ WARNING — overlap detected (investigate)")

    lines.append("")
    lines.append(dsep)
    lines.append("OUTPUT FILES")
    lines.append(dsep)
    for name, ds in [("train.json", train_set), ("val.json", val_set), ("eval.json", eval_set)]:
        p = OUT_DIR / name
        lines.append(f"  {p}  ({len(ds)} records)")
    lines.append(dsep)

    report = "\n".join(lines)
    (OUT_DIR / "split_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
