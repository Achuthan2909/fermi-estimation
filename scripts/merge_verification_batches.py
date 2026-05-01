"""Merge multiple verification batch JSON files into one.

Usage:
    # After Ria and Achuthan both finish and commit/push their run/ files:
    python scripts/merge_verification_batches.py \
        --inputs run/verification_train_ria_b*.json run/verification_train_achuthan_b*.json \
        --output run/verification_train_all.json

    python scripts/merge_verification_batches.py \
        --inputs run/verification_val_ria_b*.json run/verification_val_achuthan_b*.json \
        --output run/verification_val_all.json
"""
from __future__ import annotations

import argparse
import json
import os
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    all_records: list[dict] = []
    first_metadata: dict = {}

    for path in sorted(args.inputs):
        if not os.path.exists(path):
            print(f"  SKIP (not found): {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        records = data.get("results", [])
        all_records.extend(records)
        if not first_metadata:
            first_metadata = data.get("metadata", {})
        print(f"  loaded {len(records):4d} records from {path}")

    all_records.sort(key=lambda r: r["index"])

    verdict_counts: dict[str, int] = {}
    stage_counts: dict[str, int] = {}
    total_matches = 0
    for rec in all_records:
        cv = rec.get("consensus_verdict", "unknown")
        verdict_counts[cv] = verdict_counts.get(cv, 0) + 1
        for v in rec.get("verifications", []):
            total_matches += v.get("fermi_facts_matches", 0)
            for stage in ("search_plan", "computation"):
                key = f"{stage}:{v.get(stage, {}).get('status', 'unknown')}"
                stage_counts[key] = stage_counts.get(key, 0) + 1

    payload = {
        "metadata": {
            **first_metadata,
            "merged_from": sorted(args.inputs),
            "merged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_records": len(all_records),
        },
        "summary": {
            "n": len(all_records),
            "consensus_verdict_counts": verdict_counts,
            "stage_status_counts": stage_counts,
            "total_fermi_matches": total_matches,
        },
        "results": all_records,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"\nMerged {len(all_records)} records → {args.output}")
    print(f"Verdicts: {verdict_counts}")


if __name__ == "__main__":
    main()
