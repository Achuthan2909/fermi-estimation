"""End-to-end: merge per-batch verification files and filter+relabel a dataset.

Wraps `scripts/merge_verification_batches.py` and `scripts/filter_relabel_dataset.py`
so a single command produces:
  - merged verifications JSON
  - filtered dataset JSON (correct + relabeled high/medium confidence)
  - review JSON (relabel candidates flagged low confidence)

Example:
    python scripts/build_filtered_dataset.py \
        --dataset datasets/final/train.json \
        --batches 'run/verification_train_achuthan_b*.json' \
        --merged run/verification_train_all.json \
        --output datasets/final/train_filtered.json \
        --review datasets/final/train_review.json

Pass --batches multiple times to combine globs (e.g. Ria's + Achuthan's).
"""
from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MERGE = REPO_ROOT / "scripts" / "merge_verification_batches.py"
FILTER = REPO_ROOT / "scripts" / "filter_relabel_dataset.py"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument(
        "--batches",
        action="append",
        required=True,
        help="Glob for per-batch verification files. Can be passed multiple times.",
    )
    ap.add_argument("--merged", required=True, help="Path to write merged verification JSON.")
    ap.add_argument("--output", required=True, help="Filtered dataset out.")
    ap.add_argument("--review", required=True, help="Low-confidence review out.")
    ap.add_argument("--oom-threshold", type=float, default=2.0)
    ap.add_argument("--keep-unverified", action="store_true")
    args = ap.parse_args()

    inputs: list[str] = []
    for pat in args.batches:
        matched = sorted(f for f in glob.glob(pat) if not f.endswith(".partial"))
        if not matched:
            print(f"warning: no files matched {pat!r}", file=sys.stderr)
        inputs.extend(matched)
    if not inputs:
        print("error: no batch files found", file=sys.stderr)
        return 1

    print(f"=== merge ({len(inputs)} files) -> {args.merged} ===")
    rc = subprocess.call(
        [sys.executable, str(MERGE), "--inputs", *inputs, "--output", args.merged]
    )
    if rc != 0:
        return rc

    print(f"\n=== filter -> {args.output} (review: {args.review}) ===")
    cmd = [
        sys.executable,
        str(FILTER),
        "--dataset", args.dataset,
        "--verifications", args.merged,
        "--output", args.output,
        "--review", args.review,
        "--oom-threshold", str(args.oom_threshold),
    ]
    if args.keep_unverified:
        cmd.append("--keep-unverified")
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
