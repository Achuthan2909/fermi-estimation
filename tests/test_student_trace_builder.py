from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_builder():
    root = Path(__file__).resolve().parents[1]
    path = root / "src" / "training" / "build_student_traces_from_verifier.py"
    spec = importlib.util.spec_from_file_location("student_trace_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_record(*, verdict: str = "correct", confidence: str = "high") -> dict:
    return {
        "id": "fermiq:test",
        "question": "What fraction of a cubic meter is a milliliter?",
        "answer_type": "oom_exponent",
        "split": "train",
        "dataset": "datasets/final/train.json",
        "consensus_verdict": verdict,
        "verifications": [
            {
                "model": "teacher-model",
                "search_plan": {"status": "ok"},
                "computation": {
                    "status": "ok",
                    "parsed_response": {
                        "reference_values": [
                            {
                                "name": "milliliter to liter conversion",
                                "value": "0.001",
                                "unit": "liter per milliliter",
                                "source": "Reference 1.1",
                                "reasoning": "Directly states that a milliliter is equal to 1/1000 of a liter.",
                            },
                            {
                                "name": "cubic meter to liter conversion",
                                "value": "1000",
                                "unit": "liters per cubic meter",
                                "source": "Reference 2.1",
                                "reasoning": "The reference states that 1 cubic meter equals 1000 liters.",
                            },
                        ],
                        "memory_values": [
                            {
                                "name": "sanity-check scale",
                                "value": "1/1000000",
                                "unit": "cubic meter per milliliter",
                                "reasoning": "I remember metric volume prefixes differ by powers of ten.",
                                "plausibility_check": "Milliliters are tiny compared with cubic meters.",
                            }
                        ],
                        "calculation": "0.001 / 1000 = 1e-6",
                        "final_answer": {
                            "value": 1e-6,
                            "unit": "cubic meters per milliliter",
                            "oom": -6,
                        },
                        "confidence": confidence,
                        "reasoning": "Use exact metric conversions.",
                    },
                },
                "comparison": {"verdict": verdict},
            }
        ],
    }


def test_convert_record_builds_sampling_schema_without_reference_sources():
    builder = load_builder()

    converted = builder.convert_record(
        make_record(),
        min_confidence="medium",
        require_consensus_correct=True,
    )

    assert converted is not None
    target = converted["target"]
    builder.parse_response(json.dumps(target))
    assert target["question"] == "What fraction of a cubic meter is a milliliter?"
    assert target["final_answer"]["oom"] == -6
    assert len(target["assumptions"]) == 3
    assert all("source" not in assumption for assumption in target["assumptions"])
    assert "Reference" not in json.dumps(target["assumptions"])
    assert target["assumptions"][0]["low"] == 0.001
    assert target["assumptions"][0]["high"] == 0.001
    assert target["assumptions"][2]["value"] == 1e-6
    assert target["assumptions"][2]["low"] == 5e-7
    assert target["assumptions"][2]["high"] == 2e-6


def test_convert_record_filters_low_confidence_by_default():
    builder = load_builder()

    converted = builder.convert_record(
        make_record(confidence="low"),
        min_confidence="medium",
        require_consensus_correct=True,
    )

    assert converted is None


def test_convert_record_filters_incorrect_consensus():
    builder = load_builder()

    converted = builder.convert_record(
        make_record(verdict="incorrect"),
        min_confidence="medium",
        require_consensus_correct=True,
    )

    assert converted is None
