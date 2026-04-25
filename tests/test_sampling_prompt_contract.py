from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_sampling_contract():
    root = Path(__file__).resolve().parents[1]
    path = root / "src" / "sampling" / "prompt_contract.py"
    spec = importlib.util.spec_from_file_location("sampling_prompt_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_user_message_omits_ground_truth_instruction():
    contract = load_sampling_contract()

    message = contract.build_user_message("How many piano tuners are in Chicago?", None)

    assert "Ground Truth" not in message
    assert "Do not include ground truth" in message
    assert "final_answer" in message


def test_parse_response_accepts_sampling_trace_schema():
    contract = load_sampling_contract()
    response = {
        "question": "How many piano tuners are in Chicago?",
        "assumptions": [
            {
                "name": "population",
                "value": 3_000_000,
                "unit": "people",
                "low": 1_000_000,
                "high": 10_000_000,
                "reasoning": "Large US city scale.",
            }
        ],
        "calculation": (
            "3e6 people * 1 piano / 100 people * "
            "1 tuner / 1000 pianos = 30"
        ),
        "final_answer": {"value": 30, "unit": "tuners", "oom": 1},
    }

    parsed = contract.parse_response(json.dumps(response))

    assert parsed["final_answer"]["value"] == 30
