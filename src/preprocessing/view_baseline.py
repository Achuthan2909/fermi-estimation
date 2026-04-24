"""Browse a baseline run's per-item reasoning for error analysis.

Examples:
    # worst 10 items overall
    python3 src/view_baseline.py baseline_Qwen_Qwen3-4B-Instruct-2507_eval_v1 --worst 10

    # every unit_fail
    python3 src/view_baseline.py baseline_Qwen_Qwen3-4B-Instruct-2507_eval_v1 --status unit_fail

    # one specific item
    python3 src/view_baseline.py baseline_Qwen_Qwen3-4B-Instruct-2507_eval_v1 --id realFP:test:120

    # summary table only
    python3 src/view_baseline.py baseline_Qwen_Qwen3-4B-Instruct-2507_eval_v1 --summary
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from prompt_contract import parse_response, ParseError  # noqa: E402


def load(run: str) -> list[dict[str, Any]]:
    path = os.path.join(ROOT, "datasets", "eval", run + ".jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"not found: {path}")
    return [json.loads(l) for l in open(path)]


def fmt_item(r: dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 88)
    lines.append(f"{r['id']}   [{r.get('source','')}/{r.get('domain','')}]   "
                 f"status={r['status']}  score={r['score']:.3f}")
    lines.append(f"Q: {r['question']}")
    gold = r.get('gold_numeric')
    if gold is not None:
        lines.append(f"GOLD: {gold}  {r.get('gold_unit') or ''}   "
                     f"(oom={r.get('gold_oom')})")
    else:
        lines.append(f"GOLD: oom={r.get('gold_oom')} (oom_exponent)")
    lines.append(f"PRED: {r.get('pred_value')}  {r.get('pred_unit') or ''}   "
                 f"(oom={r.get('pred_oom')})")
    if r.get("reason"):
        lines.append(f"NOTE: {r['reason']}")

    # Pretty-print the reasoning JSON if parseable
    try:
        obj = parse_response(r["response_text"])
        lines.append(f"  restatement: {obj.get('restatement','')}")
        for a in obj.get("assumptions", [])[:6]:
            lines.append(f"    - {a}")
        lines.append(f"  formula: {obj.get('formula','')}")
        lines.append("  factors:")
        for fac in obj.get("factors", []):
            lines.append(f"    {fac.get('symbol','?'):>14} = {fac.get('point','?')} "
                         f"{fac.get('unit','') or '':<8}  "
                         f"[{fac.get('low','?')} .. {fac.get('high','?')}]   "
                         f"— {fac.get('why','')}")
        lines.append(f"  calculation: {obj.get('calculation','')}")
        fin = obj.get("final", {})
        lines.append(f"  final: value={fin.get('value')} unit={fin.get('unit')} "
                     f"oom={fin.get('oom')} confidence={obj.get('confidence')}")
    except ParseError as e:
        lines.append(f"  (unparseable response — {e})")
        lines.append("  raw response (first 500 chars):")
        lines.append("    " + (r["response_text"] or "")[:500].replace("\n", "\n    "))
    return "\n".join(lines)


def summary(rows: list[dict[str, Any]]) -> str:
    from collections import Counter
    by_stat = Counter(r["status"] for r in rows)
    by_src = Counter(r.get("source", "?") for r in rows)
    ok = [r for r in rows if r["status"] == "ok"]
    mean_all = sum(r["score"] for r in rows) / max(1, len(rows))
    mean_ok = sum(r["score"] for r in ok) / max(1, len(ok))

    # per-source mean
    src_scores: dict[str, list[float]] = {}
    for r in rows:
        src_scores.setdefault(r.get("source", "?"), []).append(r["score"])
    per_src = {s: sum(v) / len(v) for s, v in src_scores.items()}

    # per-status examples
    out = []
    out.append(f"n={len(rows)}   mean_score={mean_all:.3f}   mean_ok={mean_ok:.3f}")
    out.append(f"by status: {dict(by_stat)}")
    out.append(f"by source: {dict(by_src)}")
    out.append(f"per-source mean score: "
               + ", ".join(f"{k}={v:.3f}" for k, v in per_src.items()))
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="basename of baseline file (no .jsonl)")
    p.add_argument("--worst", type=int, default=0, help="show N lowest-scoring items")
    p.add_argument("--best", type=int, default=0, help="show N highest-scoring items")
    p.add_argument("--status", default=None, help="filter to one status")
    p.add_argument("--id", default=None, help="show a specific item by id")
    p.add_argument("--summary", action="store_true")
    args = p.parse_args()

    rows = load(args.run)
    print(summary(rows))
    print()

    if args.summary and not (args.worst or args.best or args.status or args.id):
        return

    if args.id:
        picked = [r for r in rows if r["id"] == args.id]
    elif args.status:
        picked = [r for r in rows if r["status"] == args.status]
    elif args.worst:
        picked = sorted(rows, key=lambda r: r["score"])[:args.worst]
    elif args.best:
        picked = sorted(rows, key=lambda r: -r["score"])[:args.best]
    else:
        picked = sorted(rows, key=lambda r: r["score"])[:10]  # default: worst 10

    for r in picked:
        print(fmt_item(r))


if __name__ == "__main__":
    main()
