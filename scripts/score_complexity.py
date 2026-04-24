#!/usr/bin/env python3
"""
Complexity scoring pass (Task 3).

Reads science_domain_results.jsonl (Task 2 output), keeps only problems
that passed the filter, computes five complexity metrics, assigns a
complexity_tier (low / medium / high) via tertile splits, and writes
science_domain_scored.jsonl.

Also generates data/filtered/complexity_distribution.png — a two-panel
grouped bar chart showing complexity tier counts broken down by
(a) source and (b) relevance tier.

Inputs:
  data/filtered/science_domain_results.jsonl

Outputs:
  data/filtered/science_domain_scored.jsonl
  data/filtered/complexity_distribution.png
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
FILTER_DIR = DATA_DIR / "filtered"
IN_FILE    = FILTER_DIR / "science_domain_results.jsonl"
OUT_FILE   = FILTER_DIR / "science_domain_scored.jsonl"
PLOT_FILE  = FILTER_DIR / "complexity_distribution.png"

# Physical unit patterns — both full names and abbreviations
UNIT_PATTERN = re.compile(
    r'\b(?:'
    r'kg|kilogram|gram|g(?=\b)|milligram|mg|tonne|ton|pound|lb|ounce|oz'
    r'|meter|metre|m(?=\b)|km|kilometer|kilometre|cm|centimeter|mm|millimeter|nm'
    r'|inch|foot|feet|ft|mile|mi|yard|yd'
    r'|joule|J(?=\b)|kJ|kilojoule|MJ|megajoule|GJ|gigajoule|calorie|cal|kcal|eV|keV|MeV|GeV'
    r'|watt|W(?=\b)|kW|kilowatt|MW|megawatt|GW|gigawatt|horsepower|hp'
    r'|kelvin|K(?=\b)|celsius|centigrade|fahrenheit|°C|°F'
    r'|mol|mole|mmol|μmol|nanomol|nmol'
    r'|pascal|Pa(?=\b)|kPa|MPa|bar|atm|psi'
    r'|hertz|Hz|kHz|MHz|GHz|THz'
    r'|second|sec|s(?=\b)|minute|min|hour|hr|day|week|month|year|yr|century|millennium'
    r'|liter|litre|L(?=\b)|mL|milliliter|μL|microliter|gallon|gal|pint|quart'
    r'|ampere|amp|A(?=\b)|mA|milliamp|volt|V(?=\b)|kV|coulomb|C(?=\b)|ohm|Ω|farad|F(?=\b)'
    r'|newton|N(?=\b)|dyne|pound-force|lbf'
    r'|bit|byte|KB|MB|GB|TB|PB'
    r'|parsec|pc|light.year|AU|astronomical.unit|angstrom|Å'
    r')\b',
    re.IGNORECASE,
)


def count_units(text: str) -> int:
    """Count distinct physical unit tokens in the question."""
    matches = UNIT_PATTERN.findall(text)
    return len(set(m.lower() for m in matches))


def count_sentences(text: str) -> int:
    """Count sentences in question text."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return len([p for p in parts if p.strip()])


def tertile_thresholds(values: list[float]) -> tuple[float, float]:
    """Return (p33, p67) thresholds for tertile splits."""
    arr = sorted(values)
    n = len(arr)
    lo = arr[n // 3]
    hi = arr[(2 * n) // 3]
    return lo, hi


def assign_tier(value: float, lo: float, hi: float) -> str:
    if value <= lo:
        return "low"
    if value <= hi:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not IN_FILE.exists():
        print(f"ERROR: {IN_FILE} not found. Run filter_science_domain.py first.")
        raise SystemExit(1)

    # Load filtered results
    filtered: list[dict] = []
    with IN_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if rec.get("passed_filter"):
                    filtered.append(rec)

    print(f"Loaded {len(filtered)} problems that passed the science filter.")

    if not filtered:
        print("No problems to score.")
        return

    # Compute raw metrics
    for rec in filtered:
        q = rec["question"]
        rec["word_count"]     = len(q.split())
        rec["unit_count"]     = count_units(q)
        rec["sentence_count"] = count_sentences(q)
        rec["tag_count"]      = len(rec.get("tags", []))

    # Compute complexity score = word_count + 3*unit_count + 2*sentence_count + tag_count
    # (weighted sum used to define tertiles; not stored, only tier is stored)
    scores = [
        r["word_count"] + 3 * r["unit_count"] + 2 * r["sentence_count"] + r["tag_count"]
        for r in filtered
    ]
    lo, hi = tertile_thresholds(scores)
    print(f"Complexity tertile thresholds: low ≤ {lo:.1f}, medium ≤ {hi:.1f}, high > {hi:.1f}")

    for rec, sc in zip(filtered, scores):
        rec["complexity_tier"] = assign_tier(sc, lo, hi)

    # Write output
    FILTER_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for rec in filtered:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(filtered)} records → {OUT_FILE}")

    # ---- summary ----
    tier_counts = {"low": 0, "medium": 0, "high": 0}
    for rec in filtered:
        tier_counts[rec["complexity_tier"]] += 1
    print(f"\nComplexity tiers: low={tier_counts['low']}  "
          f"medium={tier_counts['medium']}  high={tier_counts['high']}")

    # ---- plot ----
    _plot(filtered)
    print(f"Plot saved → {PLOT_FILE}")


def _plot(records: list[dict]) -> None:
    """Two-panel grouped bar chart: complexity tier × source and × relevance."""
    tiers = ["low", "medium", "high"]
    sources     = sorted({r["source"] for r in records})
    rel_tiers   = [rt for rt in ("core", "applied") if any(r["relevance"] == rt for r in records)]

    # Count matrices
    src_mat = {s: [0, 0, 0] for s in sources}
    rel_mat = {rt: [0, 0, 0] for rt in rel_tiers}
    for rec in records:
        ti = tiers.index(rec["complexity_tier"])
        src_mat[rec["source"]][ti] += 1
        if rec["relevance"] in rel_mat:
            rel_mat[rec["relevance"]][ti] += 1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Complexity Tier Distribution of Filtered Training Problems", fontsize=13)

    def grouped_bars(ax, matrix: dict, title: str, palette: list[str]) -> None:
        keys  = list(matrix.keys())
        vals  = [matrix[k] for k in keys]
        x     = np.arange(len(tiers))
        width = 0.8 / max(len(keys), 1)
        for i, (key, v) in enumerate(zip(keys, vals)):
            offset = (i - len(keys) / 2 + 0.5) * width
            bars = ax.bar(x + offset, v, width=width, label=key, color=palette[i % len(palette)])
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, str(int(h)),
                            ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(tiers)
        ax.set_xlabel("Complexity Tier")
        ax.set_ylabel("Problem Count")
        ax.set_title(title)
        ax.legend(fontsize=8)

    src_palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    rel_palette = ["#2196F3", "#FF9800"]

    grouped_bars(axes[0], src_mat, "by Source",         src_palette)
    grouped_bars(axes[1], rel_mat, "by Relevance Tier", rel_palette)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
