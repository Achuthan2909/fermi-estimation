from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def load_runner():
    root = Path(__file__).resolve().parents[1]
    path = root / "src" / "verifier" / "run_verifier.py"
    spec = importlib.util.spec_from_file_location("run_verifier", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_consensus_verdict_agreement():
    runner = load_runner()
    verifications = [
        {"comparison": {"verdict": "correct"}},
        {"comparison": {"verdict": "correct"}},
    ]

    assert runner.consensus_verdict(verifications) == "correct"


def test_consensus_verdict_disagreement():
    runner = load_runner()
    verifications = [
        {"comparison": {"verdict": "correct"}},
        {"comparison": {"verdict": "incorrect"}},
    ]

    assert runner.consensus_verdict(verifications) == "disagreement"


def test_verdict_from_computation_exact_oom_match():
    runner = load_runner()
    row = {"answer_type": "oom_exponent", "fermi_answer": 30}
    parsed = {
        "final_answer": {"value": 1e30, "unit": "femtojoules", "oom": 30}
    }

    comparison = runner.verdict_from_computation(row, parsed)

    assert comparison["verdict"] == "correct"
    assert comparison["pred_oom"] == 30


def test_verdict_from_computation_wrong_oom():
    runner = load_runner()
    row = {"answer_type": "oom_exponent", "fermi_answer": -6}
    parsed = {
        "final_answer": {"value": 1e-3, "unit": "dimensionless", "oom": -3}
    }

    comparison = runner.verdict_from_computation(row, parsed)

    assert comparison["verdict"] == "incorrect"
    assert comparison["reason"] == "oom_error=3"


def test_format_model_search_context_is_neutral():
    runner = load_runner()
    plan = {
        "constants_needed": [
            {
                "name": "femto prefix",
                "unit": "",
                "why": "Needed to convert femtojoules.",
            }
        ]
    }
    records = [
        {
            "query": "femto prefix value",
            "status": "ok",
            "results": [
                {
                    "title": "Femto",
                    "url": "https://example.com/femto",
                    "snippet": "Femto denotes a factor of 1e-15.",
                }
            ],
        }
    ]

    context = runner.format_model_search_context(plan, records)

    assert "Femto" in context
    assert "1e-15" in context
    assert "search" not in context.lower()
    assert "query" not in context.lower()
    assert "https://" not in context


def test_run_model_verifier_saves_input_messages():
    runner = load_runner()
    calls = []

    async def fake_call_tinker_model(*, model, messages, max_tokens):
        calls.append(messages)
        if len(calls) == 1:
            return (
                json.dumps({"constants_needed": [], "search_queries": []}),
                "fake_renderer",
                0.1,
            )
        return (
            json.dumps(
                {
                    "reference_values": [],
                    "memory_values": [
                        {
                            "name": "peta prefix",
                            "value": 1e15,
                            "unit": "",
                            "reasoning": "I remember peta means 10^15.",
                            "plausibility_check": "It pairs with femto across 30 powers of ten.",
                        }
                    ],
                    "calculation": "1e15 / 1e-15 = 1e30",
                    "final_answer": {"value": 1e30, "unit": "femtojoules", "oom": 30},
                    "confidence": "high",
                    "reasoning": "A petajoule to femtojoule conversion spans 30 powers of ten.",
                }
            ),
            "fake_renderer",
            0.2,
        )

    runner.call_tinker_model = fake_call_tinker_model
    record = asyncio.run(
        runner.run_model_verifier(
            model="fake-model",
            row={
                "question": "How many femtojoules are in a petajoule?",
                "answer_type": "oom_exponent",
                "fermi_answer": 30,
            },
            max_tokens=512,
            max_search_queries=3,
            search_results=5,
            search_timeout_s=1.0,
        )
    )

    assert record["search_plan"]["input_messages"] == calls[0]
    assert record["computation"]["input_messages"] == calls[1]
    assert "memory_values" in record["computation"]["input_messages"][1]["content"]
    assert record["comparison"]["verdict"] == "correct"


def test_verify_row_runs_single_model_when_model_b_is_none():
    runner = load_runner()
    models = []

    async def fake_run_model_verifier(
        *,
        model,
        row,
        max_tokens,
        max_search_queries,
        search_results,
        search_timeout_s,
    ):
        del row, max_tokens, max_search_queries, search_results, search_timeout_s
        models.append(model)
        return {
            "model": model,
            "search_plan": {"status": "ok"},
            "computation": {"status": "ok"},
            "comparison": {"verdict": "correct"},
        }

    runner.run_model_verifier = fake_run_model_verifier
    record = asyncio.run(
        runner.verify_row(
            {
                "id": "one",
                "question": "How many milliliters are in a liter?",
                "answer_type": "oom_exponent",
                "fermi_answer": 3,
            },
            index=0,
            model_a="model-a",
            model_b=None,
            max_tokens=512,
            max_search_queries=3,
            search_results=5,
            search_timeout_s=1.0,
        )
    )

    assert models == ["model-a"]
    assert len(record["verifications"]) == 1
    assert record["consensus_verdict"] == "correct"


def test_run_processes_multiple_datasets_when_requested(monkeypatch, tmp_path):
    runner = load_runner()
    train_path = tmp_path / "train.json"
    val_path = tmp_path / "val.json"
    output_path = tmp_path / "verification.json"
    train_path.write_text(
        json.dumps([{"id": "train-1", "question": "train question"}])
    )
    val_path.write_text(json.dumps([{"id": "val-1", "question": "val question"}]))

    async def fake_verify_row(
        row,
        *,
        index,
        model_a,
        model_b,
        max_tokens,
        max_search_queries,
        search_results,
        search_timeout_s,
    ):
        del model_a, model_b, max_tokens, max_search_queries, search_results, search_timeout_s
        return {
            "index": index,
            "id": row["id"],
            "question": row["question"],
            "verifications": [
                {
                    "search_plan": {"status": "ok"},
                    "computation": {"status": "ok"},
                    "comparison": {"verdict": "correct"},
                }
            ],
            "consensus_verdict": "correct",
        }

    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    monkeypatch.setattr(runner, "verify_row", fake_verify_row)

    payload = asyncio.run(
        runner.run(
            SimpleNamespace(
                dataset=str(train_path),
                datasets=[str(train_path), str(val_path)],
                output=str(output_path),
                offset=0,
                limit=5,
                all=True,
                model_a="model-a",
                model_b=None,
                max_tokens=512,
                max_search_queries=3,
                search_results=5,
                search_timeout=1.0,
            )
        )
    )

    assert [record["id"] for record in payload["results"]] == ["train-1", "val-1"]
    assert [record["split"] for record in payload["results"]] == ["train", "val"]
    assert payload["metadata"]["limit"] is None
    assert len(payload["metadata"]["datasets"]) == 2
