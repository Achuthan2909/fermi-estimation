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
    return reward_breakdown(rollout_text, ground_truth)["reward"]


def reward_breakdown(rollout_text: str, ground_truth: dict) -> dict:
    """Same scoring as ``reward_fn`` but returns components for logging.

    Returns a dict with:
      - reward       : final scalar (same as reward_fn)
      - status       : "ok" | "parse_fail" | "schema_fail"
      - oom_score    : in [0, 1] (or 0 on schema_fail / parse_fail)
      - struct_score : in [0, 1] (factors+calc+formula, 1/3 each)
      - delta        : |log10(pred) - gold_oom|  (None on parse/schema fail)
      - parse_ok     : bool, did JSON parse?
      - has_factor   : bool
      - has_calc     : bool
      - has_formula  : bool
    """
    fail = lambda reward, status: {  # noqa: E731
        "reward": reward, "status": status,
        "oom_score": 0.0, "struct_score": 0.0, "delta": None,
        "parse_ok": status != "parse_fail",
        "has_factor": False, "has_calc": False, "has_formula": False,
    }

    text = re.sub(r"<think>.*?</think>", "", rollout_text, flags=re.DOTALL).strip()

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return fail(-1.0, "parse_fail")
    if not isinstance(parsed, dict):
        return fail(-1.0, "parse_fail")

    final = parsed.get("final")
    if not isinstance(final, dict) or "oom" not in final:
        return fail(-0.5, "schema_fail")
    pred_oom = final.get("oom")
    if not isinstance(pred_oom, int):
        return fail(-0.5, "schema_fail")

    pred_value = final.get("value")
    if pred_value is not None:
        if not isinstance(pred_value, (int, float)):
            pred_value = None
        elif isinstance(pred_value, float) and (math.isnan(pred_value) or math.isinf(pred_value)):
            pred_value = None

    gold_oom = ground_truth.get("provided_ground_truth")
    if not isinstance(gold_oom, int):
        return fail(-0.5, "schema_fail")

    # FP accuracy (log-space; safe for wild pred_oom).
    try:
        if isinstance(pred_value, (int, float)) and not isinstance(pred_value, bool) \
                and math.isfinite(pred_value) and pred_value > 0:
            log_pred = math.log10(pred_value)
        else:
            log_pred = float(pred_oom)
        delta = abs(log_pred - float(gold_oom))
    except (ValueError, TypeError):
        delta = float("inf")
    # Sharper than validate_sft (which uses /3.0): RL needs a steeper
    # gradient near the target so the policy actually pushes |err| down.
    oom_score = max(0.0, 1.0 - delta / 1.5)

    # Structural quality.
    has_factor = False
    factors = parsed.get("factors")
    if isinstance(factors, list):
        for f in factors:
            if not isinstance(f, dict):
                continue
            low, high, point = f.get("low", 0), f.get("high", 0), f.get("point", 0)
            if low == 0 and high == 0 and point == 0:
                continue
            name = str(f.get("name", "")).strip()
            why = str(f.get("why", "")).strip()
            if not name and not why:
                continue
            has_factor = True
            break

    calculation = str(parsed.get("calculation", ""))
    has_calc = (sum(1 for c in calculation if c.isalnum()) >= 30
                and any(c.isdigit() for c in calculation))
    has_formula = bool(str(parsed.get("formula", "")).strip())

    struct_score = (1 / 3) * (int(has_factor) + int(has_calc) + int(has_formula))
    reward = 0.7 * oom_score + 0.3 * struct_score
    return {
        "reward": reward,
        "status": "ok",
        "oom_score": oom_score,
        "struct_score": struct_score,
        "delta": delta,
        "parse_ok": True,
        "has_factor": has_factor,
        "has_calc": has_calc,
        "has_formula": has_formula,
    }


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

    # correct: pred_value=730, gold_oom=2 → delta=0.863
    #   fp_score = max(0, 1 - 0.863/1.5) = 0.425; struct=1.0
    #   reward = 0.7*0.425 + 0.3 = 0.598
    # hack (gold=5): delta=2.137, beyond /1.5 cutoff → fp_score=0
    #   struct=1/3 → reward = 0.7*0 + 0.1 = 0.100
    results = [
        _run("correct (gold=2)",        correct_json,  {"provided_ground_truth": 2},  (0.55, 0.65)),
        _run("malformed JSON",          "not json {{", {"provided_ground_truth": 2},  (-1.0, -1.0)),
        _run("reward-hack (wrong OOM)", hack_json,     {"provided_ground_truth": 5},  (0.0,  0.20)),
    ]

    sys.exit(0 if all(results) else 1)
