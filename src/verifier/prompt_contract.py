"""Two-stage verifier prompt contract.

Stage 1 asks a model what facts/constants should be searched. Stage 2 presents
the found facts as neutral reference context and asks it to separate cited facts
from memory-derived estimates before computing an answer. The model never sees
the provided ground truth; Python compares afterward.
"""
from __future__ import annotations

import json
from typing import Any

CONTRACT_VERSION = "fp-verifier-search-loop-v2"

SEARCH_PLANNER_SYSTEM_PROMPT = (
    "You plan web searches for a Fermi-problem verifier. Do not solve the "
    "problem. Identify the constants, definitions, conversion factors, or "
    "background quantities needed to compute the answer. Return concise search "
    "queries that can retrieve those facts."
)

SEARCH_PLANNER_TEMPLATE = """Question:
{question}

Answer type:
{answer_type}

Return JSON matching this schema exactly:
{{
  "constants_needed": [
    {{
      "name": "<constant or fact needed>",
      "unit": "<desired unit or ''>",
      "why": "<why this fact is needed>"
    }}
  ],
  "search_queries": [
    "<specific search query for one needed fact>",
    "..."
  ]
}}

Rules:
- Do not compute the final answer.
- Do not search for a final answer; search reusable facts, definitions, or constants.
- Prefer searches for reusable facts, definitions, constants, and unit conversions.
- Keep search_queries to at most {max_search_queries} queries.
"""

COMPUTE_SYSTEM_PROMPT = (
    "You are a careful Fermi-problem verifier. You receive reference evidence "
    "as context. First separate values supported by the reference context from "
    "values you are estimating from memory. For every memory value, explain why "
    "it is plausible before using it. Then compute the answer with dimensional "
    "and order-of-magnitude checks. Return only one JSON object."
)

COMPUTE_TEMPLATE = """Question:
{question}

Answer type:
{answer_type}

Reference context:
{search_evidence}

Return JSON matching this schema exactly:
{{
  "reference_values": [
    {{
      "name": "<fact or value used>",
      "value": <number or string>,
      "unit": "<unit or ''>",
      "source": "<short reference label>",
      "reasoning": "<why this value is relevant>"
    }}
  ],
  "memory_values": [
    {{
      "name": "<fact or value estimated from memory>",
      "value": <number or string>,
      "unit": "<unit or ''>",
      "reasoning": "<why you remember or assume this value>",
      "plausibility_check": "<why the value is reasonable for this problem>"
    }}
  ],
  "calculation": "<formula and arithmetic substitution>",
  "final_answer": {{
    "value": <number>,
    "unit": "<unit>",
    "oom": <integer base-10 order of magnitude>
  }},
  "confidence": "low" | "medium" | "high",
  "reasoning": "<concise explanation of how the answer was computed>"
}}

Rules:
- Use reference_values for facts directly supported by the reference context.
- Use memory_values for values not directly stated in the reference context.
- It is OK for memory_values to be empty if the reference context provides all
  needed quantities.
- When using a memory value, explicitly reason why it makes physical or common
  sense before using it in the calculation.
- For answer_type="oom_exponent", final_answer.value should be the estimated
  quantity and final_answer.oom should be the exponent to compare.
- If evidence is insufficient, still make the best estimate and use low confidence.
- Do not mention how the context was retrieved.
- Do not include prose outside the JSON.
"""


def build_search_plan_messages(
    question: str,
    answer_type: str,
    max_search_queries: int = 5,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SEARCH_PLANNER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": SEARCH_PLANNER_TEMPLATE.format(
                question=question,
                answer_type=answer_type,
                max_search_queries=max_search_queries,
            ),
        },
    ]


def build_compute_messages(
    question: str,
    answer_type: str,
    search_evidence: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": COMPUTE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": COMPUTE_TEMPLATE.format(
                question=question,
                answer_type=answer_type,
                search_evidence=search_evidence,
            ),
        },
    ]


class ParseError(Exception):
    pass


def _parse_json_object(text: str) -> dict[str, Any]:
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
    return obj


def parse_search_plan_response(text: str) -> dict[str, Any]:
    obj = _parse_json_object(text)
    constants_needed = obj.get("constants_needed")
    search_queries = obj.get("search_queries")
    if not isinstance(constants_needed, list):
        raise ParseError("constants_needed must be a list")
    if not isinstance(search_queries, list):
        raise ParseError("search_queries must be a list")
    if not all(isinstance(query, str) for query in search_queries):
        raise ParseError("search_queries must contain only strings")
    return obj


def parse_compute_response(text: str) -> dict[str, Any]:
    obj = _parse_json_object(text)
    final = obj.get("final_answer")
    if not isinstance(final, dict):
        raise ParseError("missing 'final_answer' object")
    for key in ("value", "unit", "oom"):
        if key not in final:
            raise ParseError(f"missing final_answer.{key}")
    if not isinstance(final["value"], (int, float)):
        raise ParseError("final_answer.value not numeric")
    if obj.get("confidence") not in {"low", "medium", "high"}:
        raise ParseError("confidence must be low, medium, or high")
    for key in ("reference_values", "memory_values", "calculation", "reasoning"):
        if key not in obj:
            raise ParseError(f"missing {key!r}")
    if not isinstance(obj["reference_values"], list):
        raise ParseError("reference_values must be a list")
    if not isinstance(obj["memory_values"], list):
        raise ParseError("memory_values must be a list")
    return obj
