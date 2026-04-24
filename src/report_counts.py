"""Build a markdown summary of per-source dataset counts before/after filtering.

Reads:
  - datasets/processed/unified.jsonl
  - datasets/processed/unified_report.json
  - datasets/processed/classified.jsonl      (if present)

Writes:
  - datasets/processed/dataset_counts.md
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROCESSED = REPO / "datasets" / "processed"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open() if l.strip()]


def main() -> None:
    unified = load_jsonl(PROCESSED / "unified.jsonl")
    classified = load_jsonl(PROCESSED / "classified.jsonl")
    report = json.loads((PROCESSED / "unified_report.json").read_text())

    lines: list[str] = []
    ap = lines.append

    ap("# Dataset counts")
    ap("")
    ap("Counts reported at each stage of the build pipeline.")
    ap("")

    # --- Stage A: raw vs post-parse/dedup ---
    ap("## Stage A — raw → parsed → deduped")
    ap("")
    ap("| source | raw (from disk) | after parse+oom-clip | after dedup |")
    ap("|---|---:|---:|---:|")
    raw_counts = {
        "realFP": 185 + 125 + 557,
        "synthFP": 8000 + 1000 + 1000,
        "fermiq": 279,
        "scioly": 4008,
    }
    parse_counts = report["pre_dedup_counts"]
    dedup_counts = report["post_dedup_counts"]
    total_raw = total_parse = total_dedup = 0
    for src in ("realFP", "synthFP", "fermiq", "scioly"):
        ap(f"| {src} | {raw_counts[src]} | {parse_counts[src]} | {dedup_counts[src]} |")
        total_raw += raw_counts[src]
        total_parse += parse_counts[src]
        total_dedup += dedup_counts[src]
    ap(f"| **total** | **{total_raw}** | **{total_parse}** | **{total_dedup}** |")
    ap("")
    ap(f"- Dropped by parse/clip: **{total_raw - total_parse}** (answers unparseable, OoM outside [-30, 40], question < 20 chars).")
    dd = report["dedup_dropped"]
    ap(f"- Dropped by dedup: **{dd['intra_source'] + dd['cross_source']}** total "
       f"({dd['intra_source']} intra-source, {dd['cross_source']} cross-source).")
    ap("- Dedup priority: realFP > scioly > fermiq > synthFP; within source: test > val > train.")
    ap("  This preserves eval-split items when the same question appears in multiple splits.")
    ap("")
    ap("### Split breakdown (post-dedup) for fermi sources")
    ap("")
    ap("| source | train | val | test |")
    ap("|---|---:|---:|---:|")
    for src, splits in report["split_counts_after_dedup"].items():
        ap(f"| {src} | {splits.get('train',0)} | {splits.get('val',0)} | {splits.get('test',0)} |")
    ap("")

    # --- Stage B: science filter (optional, if classified.jsonl exists) ---
    if classified:
        ap("## Stage B — science-domain filter")
        ap("")
        # Best-effort metadata
        n_class = len(classified)
        n_unified = len(unified)
        ap(f"Classified **{n_class}/{n_unified}** items so far ({100*n_class/max(1,n_unified):.0f}%).")
        ap("")
        ap("### Per-source science vs non-science")
        ap("")
        ap("| source | total | is_science | non_science | % science |")
        ap("|---|---:|---:|---:|---:|")
        by_src: dict[str, dict[str, int]] = defaultdict(lambda: {"total":0,"sci":0,"non":0})
        for r in classified:
            s = r["source"]
            by_src[s]["total"] += 1
            if r.get("is_science"):
                by_src[s]["sci"] += 1
            else:
                by_src[s]["non"] += 1
        tot = sci_tot = non_tot = 0
        for src in ("realFP","synthFP","fermiq","scioly"):
            b = by_src.get(src, {"total":0,"sci":0,"non":0})
            pct = 100*b['sci']/b['total'] if b['total'] else 0
            ap(f"| {src} | {b['total']} | {b['sci']} | {b['non']} | {pct:.0f}% |")
            tot += b['total']; sci_tot += b['sci']; non_tot += b['non']
        pct = 100*sci_tot/tot if tot else 0
        ap(f"| **total** | **{tot}** | **{sci_tot}** | **{non_tot}** | **{pct:.0f}%** |")
        ap("")
        ap("### Domain distribution among science-flagged items")
        ap("")
        ap("| domain | count |")
        ap("|---|---:|")
        dom_counts = Counter(r["domain"] for r in classified if r.get("is_science"))
        for d, n in dom_counts.most_common():
            ap(f"| {d} | {n} |")
        ap("")
        # Try to detect model from first row if present
        model_note = classified[0].get("classifier_model", "qwen3.5:4b") if classified else "qwen3.5:4b"
        ap("### Notes on filter quality")
        ap(f"- Classifier: `{model_note}` via local Ollama, few-shot prompt, temperature 0.")
        ap("- Science domains: physics, chemistry, biology, astronomy, earth_science, mathematics, other_science.")
        ap("- non_science: finance, sports, pop-culture, demographic trivia, counts of commercial/cultural items.")
    else:
        ap("## Stage B — science-domain filter")
        ap("")
        ap("_No classified.jsonl found yet. Run `python3 src/science_filter.py`._")
    ap("")

    # --- Stage C: answer type distribution ---
    ap("## Stage C — answer-type distribution (post-dedup)")
    ap("")
    at_by_src = defaultdict(Counter)
    for r in unified:
        at_by_src[r["source"]][r["answer_type"]] += 1
    ap("| source | numeric | oom_exponent |")
    ap("|---|---:|---:|")
    for src in ("realFP","synthFP","fermiq","scioly"):
        c = at_by_src.get(src, Counter())
        ap(f"| {src} | {c.get('numeric',0)} | {c.get('oom_exponent',0)} |")
    ap("")

    # --- OoM distribution ---
    ap("## Stage D — OoM (log10 scale) distribution, post-dedup")
    ap("")
    ap("| OoM bucket | realFP | synthFP | fermiq | scioly |")
    ap("|---|---:|---:|---:|---:|")
    buckets = [(-30,-10,"[-30,-10)"),(-10,0,"[-10,0)"),(0,5,"[0,5)"),(5,10,"[5,10)"),(10,15,"[10,15)"),(15,25,"[15,25)"),(25,40,"[25,40]")]
    for lo, hi, label in buckets:
        row = [label]
        for src in ("realFP","synthFP","fermiq","scioly"):
            n = sum(1 for r in unified if r["source"]==src and lo <= r["gold_oom"] < hi)
            row.append(str(n))
        ap("| " + " | ".join(row) + " |")
    ap("")

    out = PROCESSED / "dataset_counts.md"
    out.write_text("\n".join(lines))
    try:
        rel = out.relative_to(REPO)
    except ValueError:
        rel = out
    print(f"Wrote report -> {rel}")


if __name__ == "__main__":
    main()
