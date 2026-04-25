from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_contract():
    root = Path(__file__).resolve().parents[1]
    path = root / "src" / "verifier" / "prompt_contract.py"
    spec = importlib.util.spec_from_file_location("verifier_prompt_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_search_plan_messages_hide_ground_truth():
    contract = load_contract()

    messages = contract.build_search_plan_messages(
        question="How many femtojoules are in a petajoule?",
        answer_type="oom_exponent",
        max_search_queries=3,
    )

    assert messages[0]["role"] == "system"
    assert "30" not in messages[1]["content"]
    assert "ground truth" not in messages[0]["content"].lower()
    assert "ground truth" not in messages[1]["content"].lower()
    assert "search_queries" in messages[1]["content"]
    assert "at most 3 queries" in messages[1]["content"]


def test_compute_messages_include_context_but_not_ground_truth():
    contract = load_contract()

    messages = contract.build_compute_messages(
        question="How many femtojoules are in a petajoule?",
        answer_type="oom_exponent",
        search_evidence="femto = 1e-15, peta = 1e15",
    )

    assert "femto = 1e-15" in messages[1]["content"]
    assert "30" not in messages[1]["content"]
    assert "ground truth" not in messages[0]["content"].lower()
    assert "ground truth" not in messages[1]["content"].lower()
    assert "search" not in messages[0]["content"].lower()
    assert "search" not in messages[1]["content"].lower()
    assert "final_answer" in messages[1]["content"]
    assert "memory_values" in messages[1]["content"]
    assert "plausibility_check" in messages[1]["content"]


def test_parse_search_plan_response_accepts_schema():
    contract = load_contract()
    response = {
        "constants_needed": [
            {"name": "femto prefix", "unit": "", "why": "Needed for conversion."}
        ],
        "search_queries": ["femto prefix value", "peta prefix value"],
    }

    parsed = contract.parse_search_plan_response(json.dumps(response))

    assert parsed["search_queries"][0] == "femto prefix value"


def test_parse_compute_response_accepts_schema():
    contract = load_contract()
    response = {
        "reference_values": [
            {
                "name": "femto prefix",
                "value": 1e-15,
                "unit": "",
                "source": "reference 1",
                "reasoning": "SI prefix.",
            }
        ],
        "memory_values": [
            {
                "name": "peta prefix",
                "value": 1e15,
                "unit": "",
                "reasoning": "I remember peta means 10^15 in SI.",
                "plausibility_check": "It is the SI prefix paired with femto across 30 powers of ten.",
            }
        ],
        "calculation": "1e15 / 1e-15 = 1e30",
        "final_answer": {"value": 1e30, "unit": "femtojoules", "oom": 30},
        "confidence": "high",
        "reasoning": "A petajoule is 1e15 J and a femtojoule is 1e-15 J.",
    }

    parsed = contract.parse_compute_response(json.dumps(response))

    assert parsed["final_answer"]["oom"] == 30
    assert parsed["memory_values"][0]["name"] == "peta prefix"
