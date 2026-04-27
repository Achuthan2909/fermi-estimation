"""Reward function for the Fermi-estimation RL phase.

Pure-Python; no Tinker imports here so the function is unit-testable in
isolation. The training loop in ``src/rl/train_rl.py`` wraps a call to
``reward_fn`` inside an ``Env.step`` and returns the float as the
trajectory reward.

Output range is roughly [-1, 1]:
  * JSON parse failure        → -1.0   (hard penalty; no reasoning to score)
  * Schema check failure      → -0.5   (parsed JSON but final.oom missing)
  * Otherwise                 → 0.7 * oom_score + 0.3 * structural_score
                                with both sub-scores in [0, 1].

OOM accuracy buckets (err = |pred_oom - gold_oom|):
  err == 0  → 1.0
  err == 1  → 0.7
  err == 2  → 0.3
  err >= 3  → 0.0

Structural quality (each adds 1/3, max 1.0):
  * factors[] has at least one MEANINGFUL entry
  * calculation has > 30 chars of substantive content
  * formula is non-empty after strip()

A "meaningful" factor is not the all-zero / all-empty hack pattern: at
least one of low/high/point is non-zero, and at least one of
name/why is a non-empty string. "Substantive" calculation requires
alphanumeric content and at least 5 distinct characters (defends
against ``"x" * 50``).

We also require ``final.value`` to be a real number — if it isn't,
structural credit is reduced to discourage outputting only an `oom`
without a quantity.
"""
from __future__ import annotations

import json
import re
from typing import Any

OOM_BUCKETS: dict[int, float] = {0: 1.0, 1: 0.7, 2: 0.3}
OOM_FAR = 0.0

OOM_WEIGHT = 0.7
STRUCT_WEIGHT = 0.3

PARSE_FAIL = -1.0
SCHEMA_FAIL = -0.5

_MIN_CALC_LEN = 30
_MIN_CALC_UNIQUE_CHARS = 5
_ALNUM_RE = re.compile(r"[A-Za-z0-9]")


def _is_real_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _factor_is_meaningful(f: Any) -> bool:
    if not isinstance(f, dict):
        return False
    name = f.get("name") or ""
    why = f.get("why") or ""
    if not (isinstance(name, str) and name.strip()) and not (
        isinstance(why, str) and why.strip()
    ):
        return False
    nums = []
    for k in ("low", "high", "point"):
        v = f.get(k)
        if _is_real_number(v):
            nums.append(float(v))
    # All-zero pattern (or no numbers at all) is the dummy hack.
    if not nums or all(n == 0.0 for n in nums):
        return False
    return True


def _calc_is_substantive(calc: Any) -> bool:
    if not isinstance(calc, str):
        return False
    s = calc.strip()
    if len(s) <= _MIN_CALC_LEN:
        return False
    if not _ALNUM_RE.search(s):
        return False
    if len(set(s)) < _MIN_CALC_UNIQUE_CHARS:
        return False
    return True


def _formula_is_present(formula: Any) -> bool:
    return isinstance(formula, str) and bool(formula.strip())


def _oom_score(pred_oom: int, gold_oom: int) -> float:
    err = abs(int(pred_oom) - int(gold_oom))
    if err in OOM_BUCKETS:
        return OOM_BUCKETS[err]
    return OOM_FAR


def reward_fn(rollout_text: str, ground_truth: dict[str, Any]) -> float:
    """Score a rollout against the gold OOM target.

    Args:
        rollout_text: Raw model output (expected to be a single JSON object
            matching the SFT schema).
        ground_truth: Dict with at least ``oom`` (int).

    Returns:
        A float in roughly [-1, 1].
    """
    try:
        obj = json.loads(rollout_text)
    except (json.JSONDecodeError, TypeError):
        return PARSE_FAIL
    if not isinstance(obj, dict):
        return PARSE_FAIL

    final = obj.get("final")
    if not isinstance(final, dict):
        return SCHEMA_FAIL
    pred_oom = final.get("oom")
    if not isinstance(pred_oom, int) or isinstance(pred_oom, bool):
        return SCHEMA_FAIL

    gold_oom = ground_truth.get("oom")
    if not isinstance(gold_oom, int) or isinstance(gold_oom, bool):
        # Caller error — treat as parse_fail so we don't silently inflate.
        return PARSE_FAIL

    oom = _oom_score(pred_oom, gold_oom)

    struct = 0.0
    factors = obj.get("factors")
    if isinstance(factors, list) and any(_factor_is_meaningful(f) for f in factors):
        struct += 1.0 / 3.0
    if _calc_is_substantive(obj.get("calculation")):
        struct += 1.0 / 3.0
    if _formula_is_present(obj.get("formula")):
        struct += 1.0 / 3.0

    # Defense: if final.value is not a real number, drop one structural unit.
    if not _is_real_number(final.get("value")):
        struct = max(0.0, struct - 1.0 / 3.0)

    # Defense: zero-reasoning answers (no factors, no formula, no calc)
    # don't earn full OOM credit even if the exponent happens to match.
    # Halves OOM when struct == 0 so the lucky-guess hack can't reach the
    # OOM ceiling.
    if struct == 0.0:
        oom *= 0.5

    return OOM_WEIGHT * oom + STRUCT_WEIGHT * struct


def _print_case(label: str, rollout: str, gt: dict[str, Any]) -> None:
    score = reward_fn(rollout, gt)
    print(f"{label:25s}  reward = {score:+.3f}")


if __name__ == "__main__":
    correct = json.dumps({
        "restatement": "How many drops in a bathtub?",
        "assumptions": ["bathtub holds ~150 L", "drop ~ 0.05 mL"],
        "factors": [
            {"name": "bathtub_volume", "symbol": "V", "low": 100, "high": 200,
             "point": 150, "unit": "L", "why": "typical residential bathtub"},
            {"name": "drop_volume", "symbol": "v", "low": 0.025, "high": 0.1,
             "point": 0.05, "unit": "mL", "why": "20 drops per mL"},
        ],
        "formula": "V / v",
        "calculation": "150 L = 150000 mL; 150000 / 0.05 = 3,000,000",
        "final": {"value": 3000000, "unit": "drops", "oom": 6},
        "confidence": "high",
    })
    _print_case("correct (gold oom=6)", correct, {"oom": 6})
    _print_case("off-by-one (gold=5)", correct, {"oom": 5})
    _print_case("off-by-three (gold=3)", correct, {"oom": 3})

    _print_case("malformed JSON", "not json {{{", {"oom": 6})

    schema_only = json.dumps({"final": {"value": 1.0, "unit": "x"}})
    _print_case("missing final.oom", schema_only, {"oom": 6})

    hack = json.dumps({
        "restatement": "",
        "assumptions": [],
        "factors": [
            {"name": "", "symbol": "", "low": 0, "high": 0, "point": 0,
             "unit": "", "why": ""}
        ],
        "formula": "",
        "calculation": "x" * 50,
        "final": {"value": 1, "unit": "", "oom": 6},
        "confidence": "high",
    })
    _print_case("reward-hack attempt", hack, {"oom": 6})

    bad_value = json.dumps({
        "factors": [{"name": "n", "low": 1, "high": 2, "point": 1.5, "why": "ok"}],
        "formula": "n",
        "calculation": "n = 1.5; result is 1.5 widgets per problem unit",
        "final": {"value": "not a number", "oom": 6},
    })
    _print_case("non-numeric final.value", bad_value, {"oom": 6})
