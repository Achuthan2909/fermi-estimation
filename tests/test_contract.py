"""Tests for prompt_contract + fp_score."""
from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
sys.path.insert(0, SRC)

from prompt_contract import (  # noqa: E402
    CONTRACT_VERSION,
    ParseError,
    SAMPLING_PARAMS,
    build_messages,
    compute_oom,
    parse_response,
)
from fp_score import fp_score, fp_score_oom, norm_unit, score_row, try_convert  # noqa: E402
from eval_loader import load_split, load_suspect_ids  # noqa: E402


def _good_response(value=1.0e9, unit="ton", oom=9):
    return json.dumps({
        "restatement": "x",
        "assumptions": ["a"],
        "factors": [{
            "name": "n", "symbol": "N", "low": 1, "high": 10,
            "point": 3, "unit": "", "why": "x",
        }],
        "formula": "N",
        "calculation": "N = 3",
        "final": {"value": value, "unit": unit, "oom": oom},
        "confidence": "medium",
    })


def test_contract_version():
    assert CONTRACT_VERSION == "fp-v1"


def test_sampling_params_deterministic():
    assert SAMPLING_PARAMS["temperature"] == 0.0
    assert SAMPLING_PARAMS["seed"] == 0


def test_build_messages_shape():
    msgs = build_messages("How many X?", "kg")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "How many X?" in msgs[1]["content"]
    assert "kg" in msgs[1]["content"]


def test_build_messages_unit_none():
    msgs = build_messages("q?", None)
    assert "SI unit" in msgs[1]["content"]


def test_parse_good_json():
    obj = parse_response(_good_response())
    assert obj["final"]["value"] == 1.0e9


def test_parse_strips_code_fence():
    fenced = "```json\n" + _good_response() + "\n```"
    obj = parse_response(fenced)
    assert obj["final"]["unit"] == "ton"


def test_parse_missing_final():
    with pytest.raises(ParseError):
        parse_response('{"restatement": "x"}')


def test_parse_non_numeric_value():
    bad = json.dumps({"final": {"value": "lots", "unit": "kg", "oom": 1}})
    with pytest.raises(ParseError):
        parse_response(bad)


def test_parse_garbage():
    with pytest.raises(ParseError):
        parse_response("not json at all")


def test_compute_oom():
    assert compute_oom(1) == 0
    assert compute_oom(1e9) == 9
    assert compute_oom(0.01) == -2
    assert compute_oom(0) == 0


# fp_score ---------------------------------------------------------------

def test_fp_score_exact():
    assert fp_score(1000, 1000) == pytest.approx(1.0)


def test_fp_score_factor_of_2():
    # log10(2) ≈ 0.301 → score ≈ 0.699
    s = fp_score(2000, 1000)
    assert 0.65 < s < 0.75


def test_fp_score_one_decade_off():
    assert fp_score(10_000, 1000) == pytest.approx(0.0, abs=1e-9)


def test_fp_score_two_decades_off_clamped():
    assert fp_score(100_000, 1000) == 0.0


def test_fp_score_under_by_decade():
    assert fp_score(100, 1000) == pytest.approx(0.0, abs=1e-9)


def test_fp_score_zero_or_negative():
    assert fp_score(0, 1000) == 0.0
    assert fp_score(1000, 0) == 0.0


# units ------------------------------------------------------------------

def test_norm_unit_aliases():
    assert norm_unit("Acres") == norm_unit("acre")
    assert norm_unit("km²") == norm_unit("km**2")
    assert norm_unit("tonnes") == norm_unit("ton")


def test_norm_unit_cm3_family():
    # the bug the baseline uncovered
    assert norm_unit("cm³") == norm_unit("cm**3")
    assert norm_unit("cm^3") == norm_unit("cm**3")
    assert norm_unit("cc") == norm_unit("cm**3")


def test_try_convert_kg_to_ton():
    # 1e11 kg = 1e8 ton
    assert try_convert(1e11, "kg", "ton") == pytest.approx(1e8)


def test_try_convert_meters_to_miles():
    assert try_convert(1609.344, "meters", "mile") == pytest.approx(1.0)


def test_try_convert_same_dim_via_aliases():
    assert try_convert(1000, "grams", "kg") == pytest.approx(1.0)


def test_try_convert_incompatible():
    assert try_convert(1.0, "kg", "meter") is None


def test_try_convert_unknown_unit():
    assert try_convert(1.0, "flarbles", "kg") is None


def test_score_row_unit_converted_scores_magnitude():
    # 1e11 kg converts to 1e8 ton, gold 9e8 ton → log10 diff ~0.95 → score ~0.05
    r = score_row(_good_response(value=1e11, unit="kg", oom=11),
                  answer_type="numeric", gold_numeric=9e8, gold_unit="ton")
    assert r["status"] == "unit_converted"
    assert r["score"] > 0.0
    assert "converted" in r["reason"]


def test_score_row_unit_converted_exact():
    # 1 ton = 1000 kg; pred in kg should get full score against ton gold
    r = score_row(_good_response(value=1000, unit="kg", oom=3),
                  answer_type="numeric", gold_numeric=1.0, gold_unit="ton")
    assert r["status"] == "unit_converted"
    assert r["score"] == pytest.approx(1.0)


def test_score_row_incompatible_units_still_unit_fail():
    r = score_row(_good_response(value=5, unit="kg", oom=0),
                  answer_type="numeric", gold_numeric=5, gold_unit="meter")
    assert r["status"] == "unit_fail"


# score_row --------------------------------------------------------------

def test_score_row_ok():
    r = score_row(_good_response(value=1.2e9, unit="ton", oom=9),
                  answer_type="numeric", gold_numeric=1.43e9, gold_unit="ton")
    assert r["status"] == "ok"
    assert r["score"] > 0.9


def test_score_row_parse_fail():
    r = score_row("nope", answer_type="numeric", gold_numeric=1.0, gold_unit="kg")
    assert r["status"] == "parse_fail"
    assert r["score"] == 0.0


def test_score_row_unit_fail():
    # Dimensionally incompatible → true unit fail (kg vs ton converts now)
    r = score_row(_good_response(value=1.0, unit="kg", oom=0),
                  answer_type="numeric", gold_numeric=1.0, gold_unit="meter")
    assert r["status"] == "unit_fail"
    assert r["score"] == 0.0


def test_score_row_gold_unit_none_skips_check():
    r = score_row(_good_response(value=1.0, unit="whatever", oom=0),
                  answer_type="numeric", gold_numeric=1.0, gold_unit=None)
    assert r["status"] == "ok"


# oom_exponent scoring ---------------------------------------------------

def test_fp_score_oom_exact():
    assert fp_score_oom(3, 3) == 1.0


def test_fp_score_oom_one_off():
    assert fp_score_oom(4, 3) == 0.0
    assert fp_score_oom(2, 3) == 0.0


def test_fp_score_oom_none():
    assert fp_score_oom(None, 3) == 0.0


def test_score_row_oom_exponent_ok():
    r = score_row(_good_response(value=120, unit="", oom=2),
                  answer_type="oom_exponent", gold_oom=2)
    assert r["status"] == "ok"
    assert r["score"] == 1.0


def test_score_row_oom_exponent_ignores_unit():
    # scioly rows usually have unit=None; we must not unit-fail them
    r = score_row(_good_response(value=5e3, unit="cars", oom=3),
                  answer_type="oom_exponent", gold_oom=3, gold_unit=None)
    assert r["status"] == "ok"
    assert r["score"] == 1.0


def test_score_row_oom_exponent_missing_gold():
    r = score_row(_good_response(), answer_type="oom_exponent", gold_oom=None)
    assert r["status"] == "gold_missing"


def test_score_row_numeric_missing_gold():
    r = score_row(_good_response(), answer_type="numeric", gold_numeric=None)
    assert r["status"] == "gold_missing"


# eval_loader ------------------------------------------------------------

def test_load_suspect_ids_has_flagged():
    s = load_suspect_ids("eval_v1")
    assert "realFP:test:295" in s
    assert "realFP:test:165" in s
    assert "realFP:test:48" in s
    assert "realFP:test:121" in s


def test_load_split_drops_suspect():
    rows, dropped = load_split("eval_v1", drop_suspect=True)
    assert len(rows) == 100 - len(dropped)
    assert dropped == {"realFP:test:295", "realFP:test:165",
                       "realFP:test:48", "realFP:test:121"}
    assert all(r["id"] not in dropped for r in rows)


def test_load_split_keep_suspect():
    rows, dropped = load_split("eval_v1", drop_suspect=False)
    assert len(rows) == 100
    assert dropped == set()
