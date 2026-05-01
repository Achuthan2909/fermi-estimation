#!/usr/bin/env python3
"""Summarize verification run files in run/ (or a custom glob).

Per file: n, consensus verdict counts (correct/incorrect/uncertain), stage
status counts, and among incorrect items the share within |oom_error| <= K
(K=2 by default; the closest verifier's oom_error is used when multiple
verifications exist).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

OOM_RE = re.compile(r"oom_error=(-?\d+(?:\.\d+)?)")


def oom_err(reason: str | None) -> float | None:
    if not reason:
        return None
    m = OOM_RE.search(reason)
    return float(m.group(1)) if m else None


def summarize(path: str, oom_threshold: float) -> dict:
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", [])
    verdicts = Counter()
    stages = Counter()
    inc = inc_within = inc_no_oom = 0
    for r in results:
        v = r.get("consensus_verdict")
        if v:
            verdicts[v] += 1
        for ver in r.get("verifications", []):
            for stage in ("search_plan", "computation"):
                st = (ver.get(stage) or {}).get("status")
                if st:
                    stages[f"{stage}:{st}"] += 1
        if v == "incorrect":
            inc += 1
            ooms = []
            for ver in r.get("verifications", []):
                comp = ver.get("comparison") or {}
                oe = oom_err(comp.get("reason", ""))
                if oe is not None:
                    ooms.append(abs(oe))
            if not ooms:
                inc_no_oom += 1
            elif min(ooms) <= oom_threshold:
                inc_within += 1
    return {
        "file": path,
        "n": len(results),
        "verdicts": dict(verdicts),
        "stages": dict(stages),
        "incorrect": inc,
        "incorrect_within_oom": inc_within,
        "incorrect_no_oom_parsed": inc_no_oom,
        "oom_threshold": oom_threshold,
    }


def fmt_table(rows: list[dict]) -> str:
    hdr = (
        f"{'file':<48} {'n':>5} {'corr':>5} {'inc':>5} {'unc':>5} "
        f"{'inc≤OOM':>8} {'inc≤OOM%':>9}"
    )
    out = [hdr, "-" * len(hdr)]
    tot = Counter()
    tot_inc = tot_within = tot_n = 0
    for s in rows:
        v = s["verdicts"]
        c, i, u = v.get("correct", 0), v.get("incorrect", 0), v.get("uncertain", 0)
        within = s["incorrect_within_oom"]
        pct = (100 * within / i) if i else 0.0
        out.append(
            f"{os.path.basename(s['file']):<48} {s['n']:>5} {c:>5} {i:>5} {u:>5} "
            f"{within:>8} {pct:>8.1f}%"
        )
        tot["correct"] += c
        tot["incorrect"] += i
        tot["uncertain"] += u
        tot_inc += i
        tot_within += within
        tot_n += s["n"]
    out.append("-" * len(hdr))
    pct = (100 * tot_within / tot_inc) if tot_inc else 0.0
    out.append(
        f"{'TOTAL':<48} {tot_n:>5} {tot['correct']:>5} {tot['incorrect']:>5} "
        f"{tot['uncertain']:>5} {tot_within:>8} {pct:>8.1f}%"
    )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--glob",
        default="run/verification_*.json",
        help="Glob (relative to repo root or absolute). Default: run/verification_*.json",
    )
    ap.add_argument("--oom-threshold", type=float, default=2.0)
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of table.")
    args = ap.parse_args()

    files = sorted(f for f in glob.glob(args.glob) if not f.endswith(".partial"))
    if not files:
        print(f"No files match {args.glob!r}", file=sys.stderr)
        return 1

    rows = [summarize(f, args.oom_threshold) for f in files]
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(fmt_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
