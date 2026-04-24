"""Re-score a saved baseline jsonl against the latest fp_score without re-sampling.

Usage:
    python3 src/rescore.py baseline_Qwen_Qwen3-4B-Instruct-2507_eval_v1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from prompt_contract import CONTRACT_VERSION  # noqa: E402
from fp_score import score_row  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="basename of baseline jsonl (no extension)")
    args = p.parse_args()

    path = os.path.join(ROOT, "datasets", "eval", args.run + ".jsonl")
    rows = [json.loads(l) for l in open(path)]

    updated = []
    for r in rows:
        new = score_row(
            r.get("response_text") or "",
            answer_type=r.get("answer_type", "numeric"),
            gold_numeric=r.get("gold_numeric"),
            gold_oom=r.get("gold_oom"),
            gold_unit=r.get("gold_unit"),
        )
        r.update({k: new[k] for k in ("score", "status", "reason",
                                      "pred_value", "pred_unit", "pred_oom")})
        updated.append(r)

    # Write back
    with open(path, "w") as f:
        for r in updated:
            f.write(json.dumps(r) + "\n")

    from collections import Counter
    by_stat = Counter(r["status"] for r in updated)
    scores = [r["score"] for r in updated]
    ok = [r for r in updated if r["status"] in ("ok", "unit_converted")]
    summary = {
        "run": args.run + "_rescored_" + CONTRACT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "source_file": os.path.relpath(path, ROOT),
        "n": len(updated),
        "mean_score": round(sum(scores) / len(updated), 4) if updated else 0.0,
        "mean_score_ok_or_converted": (
            round(sum(r["score"] for r in ok) / len(ok), 4) if ok else 0.0
        ),
        "status_counts": dict(by_stat),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(json.dumps(summary, indent=2))

    runs_path = os.path.join(ROOT, "experiments", "runs.jsonl")
    with open(runs_path, "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"Appended → {runs_path}")


if __name__ == "__main__":
    main()
