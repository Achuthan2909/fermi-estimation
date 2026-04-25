from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_evaluator():
    root = Path(__file__).resolve().parents[1]
    path = root / "src" / "sampling" / "evaluate_sampling.py"
    spec = importlib.util.spec_from_file_location("sampling_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_record(pred_oom: int, gold_oom: int) -> dict:
    response = {
        "question": "How long is a footlong in furlongs?",
        "assumptions": [
            {
                "name": "footlong",
                "value": 1,
                "unit": "ft",
                "low": 0.9,
                "high": 1.1,
                "reasoning": "A footlong is about one foot.",
            }
        ],
        "calculation": "1 / 660 = 0.0015",
        "final_answer": {"value": 0.0015, "unit": "furlongs", "oom": pred_oom},
    }
    return {
        "sample_status": "ok",
        "answer_type": "oom_exponent",
        "ground_truth_answer": gold_oom,
        "response_text": json.dumps(response),
    }


def test_score_record_exact_oom_match():
    evaluator = load_evaluator()

    scored = evaluator.score_record(make_record(pred_oom=-3, gold_oom=-3))

    assert scored["score"] == 1.0
    assert scored["pred_oom"] == -3
    assert scored["eval_status"] == "ok"


def test_score_record_wrong_oom_scores_zero():
    evaluator = load_evaluator()

    scored = evaluator.score_record(make_record(pred_oom=-2, gold_oom=-3))

    assert scored["score"] == 0.0
    assert scored["eval_reason"] == "oom_error=1"


def test_score_record_parse_fail():
    evaluator = load_evaluator()
    record = {
        "sample_status": "ok",
        "answer_type": "oom_exponent",
        "ground_truth_answer": -3,
        "response_text": "not json",
    }

    scored = evaluator.score_record(record)

    assert scored["score"] == 0.0
    assert scored["eval_status"] == "parse_fail"
