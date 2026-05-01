"""Convert verifier outputs into student-facing sampling traces.

The verifier can use search tools. The student model should not learn to cite
search references directly, so this script distills verified computations into
the existing sampling schema: question, assumptions, calculation, final_answer.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
import os
import re
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(SRC_ROOT)
sys.path.insert(0, SRC_ROOT)

from sampling.prompt_contract import parse_response  # noqa: E402

DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "datasets", "student_traces", "from_verifier.jsonl")
CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


def load_verification_payload(path: str) -> dict[str, Any]:
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def coerce_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if not isinstance(value, str):
        return None

    text = value.strip().replace(",", "")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        pass
    else:
        return int(number) if number.is_integer() else number

    if re.fullmatch(r"[+-]?\d+(\.\d+)?/[+-]?\d+(\.\d+)?", text):
        try:
            number = float(Fraction(text))
        except (ValueError, ZeroDivisionError):
            return None
        return int(number) if number.is_integer() else number
    return None


def default_bounds(value: int | float, *, exact: bool) -> tuple[int | float, int | float]:
    if exact or value == 0:
        return value, value
    if value > 0:
        return value / 2, value * 2
    return value * 2, value / 2


def clean_reasoning(text: Any) -> str:
    reasoning = " ".join(str(text or "").split())
    replacements = [
        (r"\b[Tt]he reference (context )?states that ", ""),
        (r"\b[Rr]eference context (states|provides|says) ", ""),
        (r"\b[Dd]irectly states that ", ""),
        (r"\b[Ee]stablishes that ", ""),
        (r"\b[Cc]onfirms that ", ""),
    ]
    for pattern, repl in replacements:
        reasoning = re.sub(pattern, repl, reasoning)
    return reasoning.strip() or "This value is needed for the calculation."


def assumption_from_value(value_record: dict[str, Any], *, exact: bool) -> dict[str, Any] | None:
    numeric_value = coerce_number(value_record.get("value"))
    if numeric_value is None:
        return None
    low, high = default_bounds(numeric_value, exact=exact)
    return {
        "name": str(value_record.get("name") or "assumption"),
        "value": numeric_value,
        "unit": str(value_record.get("unit") or ""),
        "low": low,
        "high": high,
        "reasoning": clean_reasoning(value_record.get("reasoning") or value_record.get("plausibility_check")),
        "plausibility_check": clean_reasoning(value_record.get("plausibility_check")),
    }


def assumptions_from_computation(computation: dict[str, Any]) -> list[dict[str, Any]]:
    assumptions: list[dict[str, Any]] = []
    for value_record in computation.get("reference_values") or []:
        if isinstance(value_record, dict):
            assumption = assumption_from_value(value_record, exact=True)
            if assumption is not None:
                assumptions.append(assumption)
    for value_record in computation.get("memory_values") or []:
        if isinstance(value_record, dict):
            assumption = assumption_from_value(value_record, exact=False)
            if assumption is not None:
                assumptions.append(assumption)
    return assumptions


def is_eligible_verification(
    verification: dict[str, Any],
    *,
    min_confidence: str,
) -> bool:
    computation_stage = verification.get("computation") or {}
    if verification.get("search_plan", {}).get("status") != "ok":
        return False
    if computation_stage.get("status") != "ok":
        return False
    if verification.get("comparison", {}).get("verdict") != "correct":
        return False
    parsed = computation_stage.get("parsed_response")
    if not isinstance(parsed, dict):
        return False
    confidence = parsed.get("confidence", "low")
    return CONFIDENCE_ORDER.get(confidence, -1) >= CONFIDENCE_ORDER[min_confidence]


def build_student_target(record: dict[str, Any], verification: dict[str, Any]) -> dict[str, Any] | None:
    computation = verification["computation"]["parsed_response"]
    target = {
        "question": record["question"],
        "assumptions": assumptions_from_computation(computation),
        "calculation": computation.get("calculation", ""),
        "final_answer": computation.get("final_answer"),
    }
    if not target["assumptions"]:
        return None
    parse_response(json.dumps(target))
    return target


def convert_record(
    record: dict[str, Any],
    *,
    min_confidence: str,
    require_consensus_correct: bool,
) -> dict[str, Any] | None:
    if require_consensus_correct and record.get("consensus_verdict") != "correct":
        return None
    for verification in record.get("verifications") or []:
        if not isinstance(verification, dict):
            continue
        if not is_eligible_verification(verification, min_confidence=min_confidence):
            continue
        target = build_student_target(record, verification)
        if target is None:
            continue
        return {
            "id": record.get("id"),
            "question": record["question"],
            "unit": record.get("unit"),
            "answer_type": record.get("answer_type", "oom_exponent"),
            "split": record.get("split"),
            "dataset": record.get("dataset"),
            "source_verifier_model": verification.get("model"),
            "target": target,
        }
    return None


def convert_payloads(
    payloads: list[dict[str, Any]],
    *,
    min_confidence: str,
    require_consensus_correct: bool,
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for payload in payloads:
        for record in payload.get("results") or []:
            if not isinstance(record, dict):
                continue
            converted_record = convert_record(
                record,
                min_confidence=min_confidence,
                require_consensus_correct=require_consensus_correct,
            )
            if converted_record is not None:
                converted.append(converted_record)
    return converted


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build student sampling traces from verifier outputs.")
    parser.add_argument("--input", nargs="+", required=True, help="Verifier JSON file(s)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--min-confidence", choices=["low", "medium", "high"], default="medium")
    parser.add_argument(
        "--allow-non-consensus",
        action="store_true",
        help="Keep correct individual verifications even when consensus is not correct.",
    )
    args = parser.parse_args()

    payloads = [load_verification_payload(path) for path in args.input]
    rows = convert_payloads(
        payloads,
        min_confidence=args.min_confidence,
        require_consensus_correct=not args.allow_non_consensus,
    )
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} student traces to {args.output}")


if __name__ == "__main__":
    main()
