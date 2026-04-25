"""Prompt + JSON schema contract for Tinker sampling traces.

This sampling contract is intentionally separate from preprocessing so prompt
experiments can evolve without changing old eval runs. The model should not see
the dataset ground truth; sampling scripts can print that beside the response.
"""
from __future__ import annotations

import json
import math
from typing import Any

CONTRACT_VERSION = "fp-sampling-trace-v1"

SYSTEM_PROMPT = (
    "You are a careful Fermi estimation solver. Use first-principles reasoning "
    "where possible. When a quantity cannot be derived from first principles, "
    "use a transparent common-sense approximate prior, give plausible low/high "
    "bounds, and explain why the central value is reasonable. Do not use "
    "lookups, search, or claim exact facts unless the problem is pure math. "
    "Keep assumptions physically plausible, dimensionally consistent, and not "
    "arbitrarily wide. Sanity-check the final order of magnitude.\n\n"
    "Respond with ONE JSON object. No prose before or after. No code fences."
)

USER_TEMPLATE = """Question: {question}
Required unit: {unit}

Return JSON matching this schema exactly:
{{
  "question": "<copy the question>",
  "assumptions": [
    {{
      "name": "<short assumption label>",
      "value": <central numeric value>,
      "unit": "<unit or ''>",
      "low": <plausible lower bound>,
      "high": <plausible upper bound>,
      "reasoning": "<why this assumption and range are plausible>"
    }}
  ],
  "calculation": "<compact formula and arithmetic substitution using assumption values>",
  "final_answer": {{
    "value": <numeric answer>,
    "unit": "<unit>",
    "oom": <integer base-10 order of magnitude>
  }}
}}

Rules:
- Do not include ground truth in the response.
- Prefer a few strong assumptions over many tiny ones.
- Ensure low <= value <= high for every assumption.
- Put the estimated quantity in final_answer.value and its base-10 exponent in
  final_answer.oom. For dimensionless answers, use "dimensionless" as the unit.
"""

# Renderer stop sequences normally end generation; max_tokens is just a backstop.
SAMPLING_PARAMS: dict[str, float | int] = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 512,
    "seed": 0,
}


def build_user_message(question: str, unit: str | None) -> str:
    unit_str = (
        unit
        if unit
        else "infer the most natural unit; state it in final_answer.unit"
    )
    return USER_TEMPLATE.format(question=question, unit=unit_str)


def build_messages(question: str, unit: str | None) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(question, unit)},
    ]


class ParseError(Exception):
    pass


def parse_response(text: str) -> dict[str, Any]:
    """Strict-ish JSON parse for the sampling trace schema."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    try:
        obj, _end = json.JSONDecoder().raw_decode(s)
    except json.JSONDecodeError as e:
        raise ParseError(f"json decode failed: {e}") from e
    if not isinstance(obj, dict):
        raise ParseError("top-level is not an object")
    assumptions = obj.get("assumptions")
    if not isinstance(assumptions, list):
        raise ParseError("missing 'assumptions' list")
    if "calculation" not in obj:
        raise ParseError("missing 'calculation'")
    final = obj.get("final_answer")
    if not isinstance(final, dict):
        raise ParseError("missing 'final_answer' object")
    for k in ("value", "unit", "oom"):
        if k not in final:
            raise ParseError(f"missing final_answer.{k}")
    if not isinstance(final["value"], (int, float)):
        raise ParseError("final_answer.value not numeric")
    return obj


def compute_oom(value: float) -> int:
    if value == 0 or not math.isfinite(value):
        return 0
    return int(math.floor(math.log10(abs(value))))
