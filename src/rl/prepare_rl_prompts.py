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

    # Stratified sampling by domain — preserve domain distribution
    rng = random.Random(args.seed)
    by_domain: dict[str, list] = defaultdict(list)
    for row in valid:
        by_domain[row.get("domain", "unknown")].append(row)

    total = len(valid)
    sampled: list[dict] = []
    for domain, domain_rows in by_domain.items():
        quota = max(1, round(args.max_items * len(domain_rows) / total))
        rng.shuffle(domain_rows)
        sampled.extend(domain_rows[:quota])

    rng.shuffle(sampled)
    sampled = sampled[: args.max_items]
    print(f"Sampled {len(sampled)} items (max_items={args.max_items})")

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


if __name__ == "__main__":
    main()
