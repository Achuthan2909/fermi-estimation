"""Fermi scoring: log10-tolerance on the point estimate.

score(pred, gold) = max(0, 1 - |log10(pred) - log10(gold)| / TOL_DECADES)
    TOL_DECADES = 1.0  → exact OOM match scores 1.0, off by one decade scores 0.0,
                        off by 0.3 decades (factor ~2) scores 0.7.

Returns a dict with score + failure category so aggregation can break out parse
failures, unit failures, and bad-value failures.
"""
from __future__ import annotations

import math
from typing import Any

from prompt_contract import ParseError, parse_response

TOL_DECADES = 1.0


_UNIT_ALIASES = {
    "acre": "acre", "acres": "acre",
    "ton": "ton", "tons": "ton", "tonne": "ton", "tonnes": "ton", "metric_ton": "ton",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "g": "g", "gram": "g", "grams": "g",
    "mg": "mg", "milligram": "mg", "milligrams": "mg",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
    "s": "s", "sec": "s", "second": "s", "seconds": "s",
    "min": "min", "minute": "min", "minutes": "min",
    "hour": "hour", "hours": "hour", "hr": "hour",
    "day": "day", "days": "day",
    "year": "year", "years": "year", "yr": "year", "yrs": "year",
    "m": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "cm": "cm", "centimeter": "cm", "centimeters": "cm",
    "mm": "mm", "millimeter": "mm", "millimeters": "mm",
    "km": "km", "kilometer": "km", "kilometers": "km",
    "mile": "mile", "miles": "mile", "mi": "mile",
    "foot": "ft", "feet": "ft", "ft": "ft",
    "inch": "in", "inches": "in", "in": "in",
    "ly": "ly", "lightyear": "ly", "light_year": "ly", "light-year": "ly",
    "au": "au", "astronomical_unit": "au",
    "m^2": "m**2", "m²": "m**2", "m**2": "m**2",
    "cm^2": "cm**2", "cm²": "cm**2", "cm**2": "cm**2",
    "km^2": "km**2", "km²": "km**2", "km**2": "km**2",
    "m^3": "m**3", "m³": "m**3", "m**3": "m**3",
    "cc": "cm**3", "cubic_centimeter": "cm**3",
    "cm^3": "cm**3", "cm³": "cm**3", "cm**3": "cm**3",
    "km^3": "km**3", "km³": "km**3", "km**3": "km**3",
    "mm^3": "mm**3", "mm³": "mm**3", "mm**3": "mm**3",
    "ly^-3": "ly**-3", "ly**-3": "ly**-3", "per_cubic_light_year": "ly**-3",
    "l": "liter", "liter": "liter", "liters": "liter", "litre": "liter", "litres": "liter",
    "ml": "ml", "milliliter": "ml",
    "gallon": "gallon", "gallons": "gallon", "gal": "gallon",
    "j": "J", "joule": "J", "joules": "J",
    "kj": "kJ", "kilojoule": "kJ",
    "mj": "MJ", "megajoule": "MJ",
    "gj": "GJ", "gigajoule": "GJ",
    "kwh": "kWh", "kilowatt_hour": "kWh", "kilowatt-hour": "kWh",
    "w": "W", "watt": "W", "watts": "W",
    "kw": "kW", "kilowatt": "kW",
    "mw": "MW", "megawatt": "MW",
    "n": "N", "newton": "N", "newtons": "N",
    "pa": "Pa", "pascal": "Pa",
    "dimensionless": "", "fraction": "", "ratio": "", "unitless": "",
    "": "",
}


def norm_unit(u: str | None) -> str:
    if u is None:
        return ""
    key = u.strip().lower().replace(" ", "_")
    return _UNIT_ALIASES.get(key, key)


# Conversion factors to a canonical "dimension unit" (SI base where possible).
# Units that share the same dimension key are interconvertible.
# Format: unit -> (dimension, factor_to_canonical).
_UNIT_TO_DIM: dict[str, tuple[str, float]] = {
    # mass → kg
    "kg": ("mass", 1.0), "g": ("mass", 1e-3), "mg": ("mass", 1e-6),
    "ton": ("mass", 1000.0), "lb": ("mass", 0.45359237),
    # length → m
    "m": ("length", 1.0), "cm": ("length", 1e-2), "mm": ("length", 1e-3),
    "km": ("length", 1e3), "mile": ("length", 1609.344),
    "ft": ("length", 0.3048), "in": ("length", 0.0254),
    "ly": ("length", 9.461e15), "au": ("length", 1.496e11),
    # time → s
    "s": ("time", 1.0), "min": ("time", 60.0), "hour": ("time", 3600.0),
    "day": ("time", 86400.0), "year": ("time", 365.25 * 86400.0),
    # area → m**2
    "m**2": ("area", 1.0), "cm**2": ("area", 1e-4), "km**2": ("area", 1e6),
    "acre": ("area", 4046.8564224),
    # volume → m**3
    "m**3": ("volume", 1.0), "cm**3": ("volume", 1e-6), "mm**3": ("volume", 1e-9),
    "km**3": ("volume", 1e9),
    "liter": ("volume", 1e-3), "ml": ("volume", 1e-6),
    "gallon": ("volume", 3.785411784e-3),
    # energy → J
    "J": ("energy", 1.0), "kJ": ("energy", 1e3), "MJ": ("energy", 1e6),
    "GJ": ("energy", 1e9), "kWh": ("energy", 3.6e6),
    # power → W
    "W": ("power", 1.0), "kW": ("power", 1e3), "MW": ("power", 1e6),
    # force → N
    "N": ("force", 1.0),
    # number density (stars per volume of light years)
    "ly**-3": ("per_ly3", 1.0),
    # pressure → Pa
    "Pa": ("pressure", 1.0),
}


def try_convert(pred_value: float, pred_unit: str, gold_unit: str) -> float | None:
    """If pred_unit and gold_unit share a dimension, return pred_value converted
    into gold_unit. Otherwise return None."""
    p = _UNIT_TO_DIM.get(norm_unit(pred_unit))
    g = _UNIT_TO_DIM.get(norm_unit(gold_unit))
    if not p or not g or p[0] != g[0]:
        return None
    return pred_value * (p[1] / g[1])


def fp_score_oom(pred_oom: int | None, gold_oom: int) -> float:
    """Score for answer_type='oom_exponent': 1 decade off = 0.0, exact = 1.0."""
    if pred_oom is None:
        return 0.0
    return max(0.0, 1.0 - abs(int(pred_oom) - int(gold_oom)))


def fp_score(pred_value: float, gold_value: float) -> float:
    if pred_value is None or gold_value is None:
        return 0.0
    if not (math.isfinite(pred_value) and math.isfinite(gold_value)):
        return 0.0
    if pred_value <= 0 or gold_value <= 0:
        # Fermi answers should be positive magnitudes; handle sign by abs.
        pred_value = abs(pred_value)
        gold_value = abs(gold_value)
        if pred_value == 0 or gold_value == 0:
            return 0.0
    delta = abs(math.log10(pred_value) - math.log10(gold_value))
    return max(0.0, 1.0 - delta / TOL_DECADES)


def score_row(
    response_text: str,
    *,
    answer_type: str = "numeric",
    gold_numeric: float | None = None,
    gold_oom: int | None = None,
    gold_unit: str | None = None,
) -> dict[str, Any]:
    """Parse a model response and score it against the gold row.

    answer_type:
      - "numeric"       → score against gold_numeric (log10-tolerance) with unit check.
      - "oom_exponent"  → score against gold_oom (decade distance). Unit ignored.
    """
    try:
        obj = parse_response(response_text)
    except ParseError as e:
        return {"score": 0.0, "status": "parse_fail", "reason": str(e),
                "pred_value": None, "pred_unit": None, "pred_oom": None}

    final = obj["final"]
    pred_value = float(final["value"])
    pred_unit = final.get("unit") or ""
    pred_oom = final.get("oom")
    # Derive pred_oom from value if model omitted/mis-set it.
    if not isinstance(pred_oom, int):
        try:
            pred_oom = int(pred_oom)
        except (TypeError, ValueError):
            import math as _m
            pred_oom = int(_m.floor(_m.log10(abs(pred_value)))) if pred_value else 0

    if answer_type == "oom_exponent":
        if gold_oom is None:
            return {"score": 0.0, "status": "gold_missing",
                    "reason": "gold_oom required for oom_exponent rows",
                    "pred_value": pred_value, "pred_unit": pred_unit, "pred_oom": pred_oom}
        s = fp_score_oom(pred_oom, gold_oom)
        return {"score": s, "status": "ok", "reason": "",
                "pred_value": pred_value, "pred_unit": pred_unit, "pred_oom": pred_oom}

    # numeric path
    if gold_numeric is None:
        return {"score": 0.0, "status": "gold_missing",
                "reason": "gold_numeric required for numeric rows",
                "pred_value": pred_value, "pred_unit": pred_unit, "pred_oom": pred_oom}
    scored_value = pred_value
    status = "ok"
    reason = ""
    if gold_unit and norm_unit(pred_unit) != norm_unit(gold_unit):
        converted = try_convert(pred_value, pred_unit, gold_unit)
        if converted is None:
            return {"score": 0.0, "status": "unit_fail",
                    "reason": f"pred_unit={pred_unit!r} vs gold_unit={gold_unit!r}",
                    "pred_value": pred_value, "pred_unit": pred_unit, "pred_oom": pred_oom}
        scored_value = converted
        status = "unit_converted"
        reason = f"converted {pred_value} {pred_unit} -> {converted:.3g} {gold_unit}"
    s = fp_score(scored_value, gold_numeric)
    return {"score": s, "status": status, "reason": reason,
            "pred_value": pred_value, "pred_unit": pred_unit, "pred_oom": pred_oom}
