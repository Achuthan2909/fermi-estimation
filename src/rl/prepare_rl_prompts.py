"""Build the RL prompt set as JSONL of {messages, metadata} entries.

Reads a Fermi dataset (default ``datasets/final/train.json``), excludes any
IDs already used for SFT (passed via ``--exclude-ids-file``), then takes a
domain-stratified sample of up to ``--max-items`` rows.

The prompts are built from the SAME ``SYSTEM_PROMPT`` and ``USER_TEMPLATE``
used by SFT (``src/preprocessing/prompt_contract.py``) so the model sees
the same input distribution at RL time.

Output line shape::

    {
      "messages": [
        {"role": "system", "content": "<SYSTEM_PROMPT>"},
        {"role": "user",   "content": "<USER_TEMPLATE.format(...)>"}
      ],
      "metadata": {
        "id": "<question_id>",
        "ground_truth_oom": <int>,
        "answer_type": "oom_exponent",
        "domain": "<domain>"
      }
    }

Example::

    python src/rl/prepare_rl_prompts.py \\
        --input-file datasets/final/train.json \\
        --output-file datasets/rl/train.jsonl \\
        --exclude-ids-file datasets/sft/filtered_v1/used_ids.txt \\
        --max-items 400
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src" / "preprocessing"))

from prompt_contract import SYSTEM_PROMPT, USER_TEMPLATE  # noqa: E402

_DEFAULT_INPUT = ROOT / "datasets" / "final" / "train.json"
_DEFAULT_OUTPUT = ROOT / "datasets" / "rl" / "train.jsonl"


def _load_exclude_ids(path: Path | None) -> set[str]:
    if not path:
        return set()
    if not path.exists():
        sys.exit(f"--exclude-ids-file not found: {path}")
    ids: set[str] = set()
    text = path.read_text()
    if path.suffix == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    ids.add(item)
                elif isinstance(item, dict) and isinstance(item.get("id"), str):
                    ids.add(item["id"])
        elif isinstance(data, dict) and "ids" in data:
            ids.update(str(x) for x in data["ids"])
        return ids
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            ids.add(s)
    return ids


def _ground_truth_oom(row: dict[str, Any]) -> int | None:
    for key in ("fermi_answer", "provided_ground_truth"):
        v = row.get(key)
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            return v
        if isinstance(v, float) and float(v).is_integer():
            return int(v)
    return None


def _stratified_sample(
    rows: list[dict[str, Any]], max_items: int, seed: int
) -> list[dict[str, Any]]:
    if max_items <= 0 or len(rows) <= max_items:
        return rows
    rng = random.Random(seed)
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_domain.setdefault(str(r.get("domain") or "unknown"), []).append(r)
    for bucket in by_domain.values():
        rng.shuffle(bucket)

    total = len(rows)
    quotas: dict[str, int] = {}
    for d, bucket in by_domain.items():
        quotas[d] = max(1, math.floor(max_items * len(bucket) / total))
    deficit = max_items - sum(quotas.values())
    if deficit:
        ordered = sorted(by_domain.keys(), key=lambda d: -len(by_domain[d]))
        i = 0
        while deficit != 0 and ordered:
            d = ordered[i % len(ordered)]
            if deficit > 0 and quotas[d] < len(by_domain[d]):
                quotas[d] += 1
                deficit -= 1
            elif deficit < 0 and quotas[d] > 1:
                quotas[d] -= 1
                deficit += 1
            i += 1

    out: list[dict[str, Any]] = []
    for d, bucket in by_domain.items():
        out.extend(bucket[: quotas[d]])
    rng.shuffle(out)
    return out[:max_items]


def _build_messages(question: str, unit: str | None) -> list[dict[str, str]]:
    unit_str = unit or "any sensible SI unit — state it in final.unit"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(question=question, unit=unit_str)},
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-file", default=str(_DEFAULT_INPUT),
                    help="JSON list of rows with question/fermi_answer/id/domain/answer_type.")
    ap.add_argument("--output-file", default=str(_DEFAULT_OUTPUT),
                    help="JSONL output (one prompt per line).")
    ap.add_argument("--exclude-ids-file", default=None,
                    help="Optional file of question IDs to skip (one per line, or JSON list).")
    ap.add_argument("--max-items", type=int, default=400,
                    help="Cap on output rows after stratification (default 400).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for stratified sampling and shuffle (default 42).")
    args = ap.parse_args()

    in_path = Path(args.input_file)
    out_path = Path(args.output_file)
    excl_path = Path(args.exclude_ids_file) if args.exclude_ids_file else None

    if not in_path.exists():
        sys.exit(f"input file not found: {in_path}")
    rows: list[dict[str, Any]] = json.loads(in_path.read_text())
    if not isinstance(rows, list):
        sys.exit(f"expected a JSON list in {in_path}")

    exclude_ids = _load_exclude_ids(excl_path)

    eligible: list[dict[str, Any]] = []
    skipped_no_question = skipped_no_oom = skipped_no_id = skipped_excluded = 0
    for r in rows:
        rid = r.get("id")
        if not isinstance(rid, str) or not rid:
            skipped_no_id += 1
            continue
        if rid in exclude_ids:
            skipped_excluded += 1
            continue
        q = r.get("question")
        if not isinstance(q, str) or not q.strip():
            skipped_no_question += 1
            continue
        oom = _ground_truth_oom(r)
        if oom is None:
            skipped_no_oom += 1
            continue
        eligible.append({**r, "_gt_oom": oom})

    sampled = _stratified_sample(eligible, args.max_items, args.seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    domain_counts: dict[str, int] = {}
    with out_path.open("w") as f:
        for r in sampled:
            messages = _build_messages(r["question"], r.get("unit"))
            metadata = {
                "id": r["id"],
                "ground_truth_oom": int(r["_gt_oom"]),
                "answer_type": r.get("answer_type") or "oom_exponent",
                "domain": str(r.get("domain") or "unknown"),
            }
            f.write(json.dumps({"messages": messages, "metadata": metadata},
                               ensure_ascii=False) + "\n")
            domain_counts[metadata["domain"]] = domain_counts.get(metadata["domain"], 0) + 1
            n_written += 1

    print(f"input rows         : {len(rows)}")
    print(f"  no id            : {skipped_no_id}")
    print(f"  excluded (sft)   : {skipped_excluded}")
    print(f"  no question      : {skipped_no_question}")
    print(f"  no fermi_answer  : {skipped_no_oom}")
    print(f"eligible           : {len(eligible)}")
    print(f"written            : {n_written} → {out_path}")
    print("domain breakdown   :")
    for d, c in sorted(domain_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {d:<24s} {c}")


if __name__ == "__main__":
    main()
