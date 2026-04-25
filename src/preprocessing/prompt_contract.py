"""Frozen prompt + schema contract for Fermi estimation runs.

Bump CONTRACT_VERSION on any change; write it into every output row so old
runs remain interpretable.
"""
from __future__ import annotations

import json
import math
from typing import Any

CONTRACT_VERSION = "fp-v1"

SYSTEM_PROMPT = (
    "You are a Fermi estimation solver. Decompose the problem into a small "
    "number of factors, bound each factor with low/high/point estimates from "
    "first principles (no lookups, no search), combine them with an explicit "
    "formula, and return a single numeric answer in the requested unit.\n\n"
    "Respond with ONE JSON object. No prose before or after. No code fences."
)

USER_TEMPLATE = """Question: {question}
Required unit: {unit}

Return JSON matching this schema exactly:
{{
  "restatement": "<one sentence, your own words>",
  "assumptions": ["<short bullet>", "..."],
  "factors": [
    {{
      "name": "<short label>",
      "symbol": "<e.g. N_people>",
      "low":   <number>,
      "high":  <number>,
      "point": <number>,
      "unit":  "<unit or ''>",
      "why":   "<one sentence justification>"
    }}
  ],
  "formula": "<e.g. 'N_people * liters_per_day * 365'>",
  "calculation": "<one-line arithmetic substitution using point values>",
  "final": {{
    "value": <number>,
    "unit":  "<unit>",
    "oom":   <integer>
  }},
  "confidence": "low" | "medium" | "high"
}}
"""

SAMPLING_PARAMS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 1024,
    "seed": 0,
}


def build_user_message(question: str, unit: str | None) -> str:
    unit_str = unit if unit else 'any sensible SI unit — state it in final.unit'
    return USER_TEMPLATE.format(question=question, unit=unit_str)


def build_messages(question: str, unit: str | None) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(question, unit)},
    ]


class ParseError(Exception):
    pass


def parse_response(text: str) -> dict[str, Any]:
    """Strict-ish JSON parse. Strips a single surrounding code fence if present."""
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
    final = obj.get("final")
    if not isinstance(final, dict):
        raise ParseError("missing 'final' object")
    for k in ("value", "unit", "oom"):
        if k not in final:
            raise ParseError(f"missing final.{k}")
    if not isinstance(final["value"], (int, float)):
        raise ParseError("final.value not numeric")
    return obj


def compute_oom(value: float) -> int:
    if value == 0 or not math.isfinite(value):
        return 0
    return int(math.floor(math.log10(abs(value))))
