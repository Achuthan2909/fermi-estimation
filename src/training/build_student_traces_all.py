"""Convert verifier outputs into student traces, keeping correct/incorrect within
an OOM tolerance and routing uncertain ones to a separate "unordered" file.

Unlike build_student_traces_from_verifier, this script does not filter on
verdict/confidence/consensus. The student target is built from the verifier's
own computation regardless of whether the verifier judged itself correct.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(SRC_ROOT)
sys.path.insert(0, SRC_ROOT)

from training.build_student_traces_from_verifier import (  # noqa: E402
    assumptions_from_computation,
    load_verification_payload,
)
from sampling.prompt_contract import ParseError, parse_response  # noqa: E402

DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "datasets", "student_traces")
OOM_REASON_RE = re.compile(r"oom_error=(-?\d+)")


def parse_oom_error(comparison: dict[str, Any]) -> int | None:
    reason = str(comparison.get("reason") or "")
    match = OOM_REASON_RE.search(reason)
    if match:
        return int(match.group(1))
    return None


def build_target(computation: dict[str, Any]) -> dict[str, Any] | None:
    target = {
        "question": computation.get("__question__", ""),
        "assumptions": assumptions_from_computation(computation),
        "calculation": computation.get("calculation", ""),
        "final_answer": computation.get("final_answer"),
    }
    try:
        parse_response(json.dumps(target))
    except ParseError:
        return None
    return target


def make_row(
    record: dict[str, Any],
    verification: dict[str, Any],
    *,
    verdict: str,
    oom_error: int | None,
) -> dict[str, Any] | None:
    parsed = (verification.get("computation") or {}).get("parsed_response")
    if not isinstance(parsed, dict):
        return None
    parsed = dict(parsed)
    parsed["__question__"] = record["question"]
    target = build_target(parsed)
    if target is None:
        return None
    return {
        "id": record.get("id"),
        "question": record["question"],
        "unit": record.get("unit"),
        "answer_type": record.get("answer_type", "oom_exponent"),
        "split": record.get("split"),
        "dataset": record.get("dataset"),
        "source_verifier_model": verification.get("model"),
        "verdict": verdict,
        "oom_error": oom_error,
        "provided_ground_truth": record.get("provided_ground_truth"),
        "confidence": parsed.get("confidence"),
        "target": target,
    }


def make_unordered_row(
    record: dict[str, Any],
    verification: dict[str, Any],
    *,
    oom_error: int | None,
) -> dict[str, Any] | None:
    computation = verification.get("computation") or {}
    response_text = computation.get("response_text")
    parsed = computation.get("parsed_response") if isinstance(computation.get("parsed_response"), dict) else None
    if not response_text and not parsed:
        return None
    return {
        "id": record.get("id"),
        "question": record["question"],
        "unit": record.get("unit"),
        "answer_type": record.get("answer_type", "oom_exponent"),
        "split": record.get("split"),
        "dataset": record.get("dataset"),
        "source_verifier_model": verification.get("model"),
        "verdict": "uncertain",
        "oom_error": oom_error,
        "provided_ground_truth": record.get("provided_ground_truth"),
        "computation_status": computation.get("status"),
        "computation_error": computation.get("error"),
        "response_text": response_text,
        "parsed_response": parsed,
    }


def convert(
    payload: dict[str, Any],
    *,
    oom_tolerance: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    main_rows: list[dict[str, Any]] = []
    unordered_rows: list[dict[str, Any]] = []
    for record in payload.get("results") or []:
        if not isinstance(record, dict):
            continue
        for verification in record.get("verifications") or []:
            if not isinstance(verification, dict):
                continue
            comparison = verification.get("comparison") or {}
            verdict = comparison.get("verdict")
            oom_error = parse_oom_error(comparison)
            if verdict in ("correct", "incorrect"):
                if oom_error is None or abs(oom_error) > oom_tolerance:
                    continue
                row = make_row(record, verification, verdict=verdict, oom_error=oom_error)
                if row is not None:
                    main_rows.append(row)
            elif verdict == "uncertain":
                row = make_unordered_row(record, verification, oom_error=oom_error)
                if row is not None:
                    unordered_rows.append(row)
    return main_rows, unordered_rows


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Verifier JSON file")
    parser.add_argument(
        "--main-output",
        default=os.path.join(DEFAULT_OUT_DIR, "train_25_1024.jsonl"),
        help="Output JSONL for correct+incorrect verifications within OOM tolerance.",
    )
    parser.add_argument(
        "--unordered-output",
        default=os.path.join(DEFAULT_OUT_DIR, "train_25_1024_unordered.jsonl"),
        help="Output JSONL for uncertain verifications.",
    )
    parser.add_argument(
        "--oom-tolerance",
        type=int,
        default=2,
        help="Max |oom_error| accepted for correct/incorrect rows (default 2).",
    )
    args = parser.parse_args()

    payload = load_verification_payload(args.input)
    main_rows, unordered_rows = convert(payload, oom_tolerance=args.oom_tolerance)
    write_jsonl(args.main_output, main_rows)
    write_jsonl(args.unordered_output, unordered_rows)
    print(f"Wrote {len(main_rows)} traces to {args.main_output}")
    print(f"Wrote {len(unordered_rows)} unordered traces to {args.unordered_output}")


if __name__ == "__main__":
    main()
