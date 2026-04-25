"""Convert student_traces JSONL to messages JSONL for SFT training.

Reads all JSONL files from datasets/student_traces/ (output of run_verifier.py),
maps each record to {"messages": [...]} format expected by
FromConversationFileBuilder, and writes train/val splits.

The target field from the verifier uses `final_answer` and `assumptions` — this
script normalises them to the baseline prompt_contract schema (`final`, `factors`)
so the SFT model's output is directly parseable by fp_score.py.

Usage:
    python sft/prepare_traces.py
    python sft/prepare_traces.py --traces-dir datasets/student_traces \
        --output-dir datasets/sft_messages --val-size 200 --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src" / "preprocessing"))

from prompt_contract import SYSTEM_PROMPT, USER_TEMPLATE  # noqa: E402

_DEFAULT_TRACES_DIR = ROOT / "datasets" / "student_traces"
_DEFAULT_OUTPUT_DIR = ROOT / "datasets" / "sft_messages"


def _load_traces(traces_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(traces_dir.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def _target_to_response(target: dict[str, Any]) -> dict[str, Any]:
    """Normalise verifier target → baseline prompt_contract response schema.

    Verifier target keys: question, assumptions (list of dicts), calculation,
    final_answer (dict: value, unit, oom).

    Baseline parse_response() requires: final.value, final.unit, final.oom,
    and confidence ∈ {low, medium, high}.
    """
    raw_assumptions = target.get("assumptions") or []

    # Map assumption dicts → factor dicts (baseline schema)
    factors = []
    for a in raw_assumptions:
        if not isinstance(a, dict):
            continue
        point = a.get("value") or a.get("point")
        factors.append({
            "name": a.get("name", ""),
            "symbol": a.get("name", "").replace(" ", "_")[:12],
            "low": a.get("low", point),
            "high": a.get("high", point),
            "point": point,
            "unit": a.get("unit") or "",
            "why": a.get("reasoning") or a.get("why") or "",
        })

    fa = target.get("final_answer") or {}
    final_value = fa.get("value")
    final_unit = fa.get("unit") or ""
    final_oom = fa.get("oom")
    if final_oom is None and final_value is not None:
        try:
            final_oom = int(math.floor(math.log10(abs(float(final_value)))))
        except (ValueError, TypeError):
            final_oom = 0

    return {
        "restatement": target.get("question") or "",
        "assumptions": [f["why"] for f in factors if f["why"]],
        "factors": factors,
        "formula": target.get("formula") or "",
        "calculation": target.get("calculation") or "",
        "final": {
            "value": final_value,
            "unit": final_unit,
            "oom": final_oom,
        },
        "confidence": "high",
    }


def _build_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    question = record["question"]
    unit = record.get("unit") or "any sensible SI unit — state it in final.unit"
    target = record.get("target") or {}
    response = _target_to_response(target)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            question=question, unit=unit,
        )},
        {"role": "assistant", "content": json.dumps(response, ensure_ascii=False)},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-dir", default=str(_DEFAULT_TRACES_DIR))
    ap.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT_DIR))
    ap.add_argument("--val-size", type=int, default=200,
                    help="Number of examples held out for val. 0 = no val split.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    traces_dir = Path(args.traces_dir)
    output_dir = Path(args.output_dir)

    if not traces_dir.exists():
        sys.exit(f"traces-dir not found: {traces_dir}")

    records = _load_traces(traces_dir)
    if not records:
        sys.exit(f"No JSONL records found in {traces_dir}")
    print(f"Loaded {len(records)} records from {traces_dir}")

    # Shuffle and split
    rng = random.Random(args.seed)
    rng.shuffle(records)
    val_size = min(args.val_size, len(records) // 5)  # cap at 20%
    val_records = records[:val_size] if val_size else []
    train_records = records[val_size:]

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_records in [("train", train_records), ("val", val_records)]:
        if not split_records:
            continue
        out_path = output_dir / f"{split_name}.jsonl"
        skipped = 0
        with out_path.open("w") as f:
            for rec in split_records:
                try:
                    messages = _build_messages(rec)
                    f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                except Exception as e:
                    print(f"  skip {rec.get('id','?')}: {e}", file=sys.stderr)
                    skipped += 1
        written = len(split_records) - skipped
        print(f"  {split_name}: {written} written → {out_path}"
              + (f" ({skipped} skipped)" if skipped else ""))

    print("Done.")


if __name__ == "__main__":
    main()
