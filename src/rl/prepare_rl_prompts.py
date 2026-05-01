"""Prepare RL prompt set from datasets/final/train.json.

Filters out questions already used for SFT (pass --exclude-ids-file),
applies stratified sampling by domain, and writes datasets/rl/train.jsonl
where each line has the exact same system+user prompt as SFT plus metadata
for the reward function.

Usage:
    python src/rl/prepare_rl_prompts.py
    python src/rl/prepare_rl_prompts.py --max-items 400 --exclude-ids-file run/sft_ids.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src" / "preprocessing"))

from prompt_contract import SYSTEM_PROMPT, USER_TEMPLATE  # noqa: E402

_DEFAULT_INPUT = ROOT / "datasets" / "final" / "train.json"
_DEFAULT_OUTPUT = ROOT / "datasets" / "rl" / "train.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-file", default=str(_DEFAULT_INPUT))
    ap.add_argument("--output-file", default=str(_DEFAULT_OUTPUT))
    ap.add_argument(
        "--exclude-ids-file",
        default=None,
        help="JSON file containing a list of question IDs to exclude (already used in SFT).",
    )
    ap.add_argument("--max-items", type=int, default=400)
    ap.add_argument("--tail-fraction", type=float, default=0.5,
                    help="Fraction of items from tail OOMs (|OOM|>9). Default 0.5 oversamples "
                         "relative to natural ~36%% distribution.")
    ap.add_argument("--val-output-file", default=None,
                    help="Optional path for a held-out val JSONL (sampled from rows not in train).")
    ap.add_argument("--val-items", type=int, default=200,
                    help="Number of val items when --val-output-file is set.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.input_file) as f:
        rows = json.load(f)
    print(f"Loaded {len(rows)} rows from {args.input_file}")

    exclude_ids: set = set()
    if args.exclude_ids_file:
        with open(args.exclude_ids_file) as f:
            data = json.load(f)
        if isinstance(data, list):
            exclude_ids = set(data)
        elif isinstance(data, dict):
            exclude_ids = set(data.keys())
        print(f"Excluding {len(exclude_ids)} SFT IDs from {args.exclude_ids_file}")

    valid: list[dict] = []
    for row in rows:
        if row.get("id") in exclude_ids:
            continue
        gt = row.get("fermi_answer") if row.get("fermi_answer") is not None else row.get("provided_ground_truth")
        if not isinstance(gt, (int, float)):
            continue
        if not row.get("question"):
            continue
        row = dict(row)
        row["_gold_oom"] = int(gt)
        valid.append(row)
    print(f"Valid rows after filtering: {len(valid)}")

    # Split into tail (|OOM|>9) and mid pools
    def _is_tail(oom: int) -> bool:
        return oom <= -4 or oom >= 10  # Low/Very Low and High/Very High buckets

    tail_rows = [r for r in valid if _is_tail(r["_gold_oom"])]
    mid_rows  = [r for r in valid if not _is_tail(r["_gold_oom"])]
    print(f"Tail OOM (<=−4 or >=10): {len(tail_rows)}  Mid OOM (−3 to 9): {len(mid_rows)}")

    rng = random.Random(args.seed)

    def _stratified_sample(pool: list[dict], quota: int) -> list[dict]:
        """Sample `quota` items from pool, stratified by domain."""
        by_domain: dict[str, list] = defaultdict(list)
        for row in pool:
            by_domain[row.get("domain", "unknown")].append(row)
        total = len(pool)
        out: list[dict] = []
        for domain_rows in by_domain.values():
            n = max(1, round(quota * len(domain_rows) / total))
            rng.shuffle(domain_rows)
            out.extend(domain_rows[:n])
        rng.shuffle(out)
        return out[:quota]

    n_tail = round(args.max_items * args.tail_fraction)
    n_mid  = args.max_items - n_tail
    sampled = _stratified_sample(tail_rows, n_tail) + _stratified_sample(mid_rows, n_mid)
    rng.shuffle(sampled)
    sampled = sampled[: args.max_items]
    print(f"Sampled {len(sampled)} items — tail: {sum(1 for r in sampled if _is_tail(r['_gold_oom']))}, "
          f"mid: {sum(1 for r in sampled if not _is_tail(r['_gold_oom']))}  (tail_fraction={args.tail_fraction})")

    # Build val pool from rows not selected for train
    sampled_ids = {r.get("id") for r in sampled}
    val_tail_rows = [r for r in tail_rows if r.get("id") not in sampled_ids]
    val_mid_rows  = [r for r in mid_rows  if r.get("id") not in sampled_ids]

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    domain_counts: dict[str, int] = defaultdict(int)
    with output_path.open("w") as f:
        for row in sampled:
            question = row["question"]
            unit = row.get("unit") or "any sensible SI unit — state it in final.unit"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(
                    question=question, unit=unit,
                )},
            ]
            record = {
                "messages": messages,
                "metadata": {
                    "id": row.get("id"),
                    "question": question,
                    "unit": unit,
                    "ground_truth_oom": row["_gold_oom"],
                    "answer_type": row.get("answer_type", "oom_exponent"),
                    "domain": row.get("domain", "unknown"),
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            domain_counts[row.get("domain", "unknown")] += 1  # type: ignore[index]

    print(f"\nWritten {len(sampled)} records → {output_path}")
    print("Domain breakdown:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {domain:<30} {count}")

    if args.val_output_file:
        n_val_tail = round(args.val_items * args.tail_fraction)
        n_val_mid  = args.val_items - n_val_tail
        val_sampled = _stratified_sample(val_tail_rows, n_val_tail) + _stratified_sample(val_mid_rows, n_val_mid)
        rng.shuffle(val_sampled)
        val_sampled = val_sampled[: args.val_items]

        val_path = Path(args.val_output_file)
        val_path.parent.mkdir(parents=True, exist_ok=True)
        with val_path.open("w") as f:
            for row in val_sampled:
                question = row["question"]
                unit = row.get("unit") or "any sensible SI unit — state it in final.unit"
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(
                        question=question, unit=unit,
                    )},
                ]
                record = {
                    "messages": messages,
                    "metadata": {
                        "id": row.get("id"),
                        "question": question,
                        "unit": unit,
                        "ground_truth_oom": row["_gold_oom"],
                        "answer_type": row.get("answer_type", "oom_exponent"),
                        "domain": row.get("domain", "unknown"),
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"\nWritten {len(val_sampled)} val records → {val_path}")
        print(f"  tail: {sum(1 for r in val_sampled if _is_tail(r['_gold_oom']))}, "
              f"mid: {sum(1 for r in val_sampled if not _is_tail(r['_gold_oom']))}")


if __name__ == "__main__":
    main()
