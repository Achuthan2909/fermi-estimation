"""Reward function for Fermi estimation RL.

No Tinker imports — fully unit-testable in isolation.

reward_fn(rollout_text, ground_truth) -> float in roughly [-1, 1]:
  - JSON parse fail          → -1.0
  - Missing final.oom        → -0.5
  - FP accuracy   (w=0.7):  max(0, 1 - (1/3) * |log10(predicted / actual)|)
      uses final.value if available, else falls back to 10^final.oom
  - Structural quality (w=0.3): factors/calculation/formula each +1/3

Adversarial defences baked in:
  - Dummy factors (all zeros / empty strings) don't count
  - Calculation with no digits doesn't pass the length check
  - final.value must be a real finite number (not None, not string)
"""
from __future__ import annotations

import json
import math
import re


def reward_fn(rollout_text: str, ground_truth: dict) -> float:
    """Return a reward in roughly [-1, 1] for one rollout.

    Args:
        rollout_text:  Raw text output from the model.
        ground_truth:  Dict with key "provided_ground_truth" (int OOM exponent).
    """
    # Strip <think>...</think> blocks (hybrid models may emit them)
    text = re.sub(r"<think>.*?</think>", "", rollout_text, flags=re.DOTALL).strip()

    # ── JSON parse ──────────────────────────────────────────────────────────
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return -1.0
    if not isinstance(parsed, dict):
        return -1.0

    # ── Schema check: final.oom must be an integer ──────────────────────────
    final = parsed.get("final")
    if not isinstance(final, dict) or "oom" not in final:
        return -0.5
    pred_oom = final.get("oom")
    if not isinstance(pred_oom, int):
        return -0.5

    # final.value must be a real finite number (not None, not string)
    pred_value = final.get("value")
    if pred_value is not None:
        if not isinstance(pred_value, (int, float)):
            pred_value = None
        elif isinstance(pred_value, float) and (math.isnan(pred_value) or math.isinf(pred_value)):
            pred_value = None

    gold_oom = ground_truth.get("provided_ground_truth")
    if not isinstance(gold_oom, int):
        return -0.5

    # ── FP accuracy (weight 0.7) ─────────────────────────────────────────────
    # fp_score = max(0, 1 - (1/3) * |log10(predicted / actual)|)
    # Use final.value when available (more precise); fall back to 10^pred_oom.
    actual = 10.0 ** gold_oom
    predicted = float(pred_value) if pred_value is not None and pred_value > 0 else 10.0 ** pred_oom
    try:
        delta = abs(math.log10(predicted / actual))
    except (ValueError, ZeroDivisionError):
        delta = float("inf")
    oom_score = max(0.0, 1.0 - delta / 3.0)

    # ── Structural quality (weight 0.3) ─────────────────────────────────────
    struct_score = 0.0

    # 1. Has at least one non-dummy factor
    factors = parsed.get("factors")
    has_real_factor = False
    if isinstance(factors, list):
        for f in factors:
            if not isinstance(f, dict):
                continue
            low = f.get("low", 0)
            high = f.get("high", 0)
            point = f.get("point", 0)
            # Reject if all numeric fields are 0
            if low == 0 and high == 0 and point == 0:
                continue
            # Reject if both name and why are empty
            name = str(f.get("name", "")).strip()
            why = str(f.get("why", "")).strip()
            if not name and not why:
                continue
            has_real_factor = True
            break
    if has_real_factor:
        struct_score += 1 / 3

    # 2. Calculation has >30 alnum chars AND at least one digit
    #    (guards against "x" * 50 padding)
    calculation = str(parsed.get("calculation", ""))
    alnum_count = sum(1 for c in calculation if c.isalnum())
    has_digit = any(c.isdigit() for c in calculation)
    if alnum_count >= 30 and has_digit:
        struct_score += 1 / 3

    # 3. Formula field is non-empty
    formula = str(parsed.get("formula", "")).strip()
    if formula:
        struct_score += 1 / 3

    return 0.7 * oom_score + 0.3 * struct_score


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    def _run(label: str, text: str, gt: dict, expected_range: tuple[float, float]) -> bool:
        r = reward_fn(text, gt)
        lo, hi = expected_range
        ok = lo <= r <= hi
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label}: reward={r:.3f}  (expected {lo:.2f}–{hi:.2f})")
        return ok

    correct_json = json.dumps({
        "restatement": "How many liters of water does a person drink per year?",
        "assumptions": ["Average daily consumption is about 2 liters"],
        "factors": [
            {
                "name": "daily water intake",
                "symbol": "W_day",
                "low": 1.5,
                "high": 2.5,
                "point": 2.0,
                "unit": "liters/day",
                "why": "Standard daily water intake recommendation for adults.",
            }
        ],
        "formula": "W_day * 365",
        "calculation": "2.0 liters/day * 365 days/year = 730 liters/year, log10(730) ~ 2.86",
        "final": {"value": 730.0, "unit": "liters", "oom": 2},
        "confidence": "high",
    })

    hack_json = json.dumps({
        "restatement": "x",
        "assumptions": [],
        "factors": [
            {"name": "", "symbol": "", "low": 0, "high": 0, "point": 0, "unit": "", "why": ""}
        ],
        "formula": "x",
        "calculation": "x" * 50,  # no digits — adversarial padding
        "final": {"value": 730.0, "unit": "liters", "oom": 2},
        "confidence": "high",
    })

    # correct: pred_value=730, gold_oom=2 → delta=|log10(730/100)|=0.863
    #   fp_score = 1 - 0.863/3 = 0.712; struct=1.0 → reward = 0.7*0.712 + 0.3 = 0.798
    # hack (gold=5): pred_value=730, actual=1e5 → delta=|log10(730/1e5)|=2.137
    #   fp_score = 1 - 2.137/3 = 0.288; struct=1/3 → reward = 0.7*0.288 + 0.1 = 0.302
    results = [
        _run("correct (gold=2)",        correct_json,  {"provided_ground_truth": 2},  (0.75, 0.85)),
        _run("malformed JSON",          "not json {{", {"provided_ground_truth": 2},  (-1.0, -1.0)),
        _run("reward-hack (wrong OOM)", hack_json,     {"provided_ground_truth": 5},  (0.0,  0.35)),
    ]

    sys.exit(0 if all(results) else 1)
