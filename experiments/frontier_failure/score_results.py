"""
Score frontier failure results using the FP score metric.

Extracts numerical answers from model responses and computes:
    fp_score = max(0, 1 - (1/3) * |log10(predicted / actual)|)

Addresses known flaws from the reference paper:
  - Uses consistent extraction across all models (structured regex,
    then fallback to last-number, both applied identically)
  - Extraction failures are flagged as None, NOT substituted with 0
  - Handles edge cases: predicted=0, negative values, unparseable text
  - Stores extraction metadata (method used, raw extracted string)
    so you can audit whether extraction caused a bad score

Usage:
    python score_results.py
"""

import json
import math
import re
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
PROBLEMS_PATH = Path(__file__).parent / "fermi_problems.json"


# ---------------------------------------------------------------------------
# Numerical extraction
# ---------------------------------------------------------------------------

def extract_final_answer(response: str) -> dict:
    """
    Extract a numerical answer from model response text.
    Returns dict with keys: value (float|None), method (str), raw (str).

    Strategy (applied identically to ALL models):
      1. Look for "FINAL ANSWER: <number>" pattern we asked for
      2. Look for scientific notation patterns (e.g., 3.2e19, 3.2 × 10^19)
      3. Fallback: last number in the response

    If nothing works, returns value=None.
    """
    if not isinstance(response, str) or response.startswith("ERROR:"):
        return {"value": None, "method": "error", "raw": response[:100] if response else ""}

    text = response.strip()

    # --- Method 1: Our requested format "FINAL ANSWER: <number>" ---
    final_pattern = r'FINAL\s+ANSWER\s*:\s*([^\n]+)'
    match = re.search(final_pattern, text, re.IGNORECASE)
    if match:
        num = parse_number_from_string(match.group(1))
        if num is not None:
            return {"value": num, "method": "final_answer_tag", "raw": match.group(1).strip()}

    # --- Method 2: Scientific notation (3.2e19, 3.2 × 10^19, 3.2 x 10^19) ---
    sci_patterns = [
        r'(\d+\.?\d*)\s*[×x\*]\s*10\s*\^\s*\{?\s*([+-]?\d+)\s*\}?',  # 3.2 × 10^19
        r'(\d+\.?\d*)[eE]([+-]?\d+)',                                   # 3.2e19
    ]
    sci_matches = []
    for pat in sci_patterns:
        for m in re.finditer(pat, text):
            mantissa = float(m.group(1))
            exponent = int(m.group(2))
            sci_matches.append((m.start(), mantissa * 10**exponent, m.group(0)))

    if sci_matches:
        # Take the LAST scientific notation number (most likely the final answer)
        _, val, raw = max(sci_matches, key=lambda x: x[0])
        return {"value": val, "method": "scientific_notation", "raw": raw}

    # --- Method 3: Fallback — last number in the text ---
    # Match integers, decimals, and comma-separated numbers
    number_pattern = r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.?\d*'
    numbers = re.findall(number_pattern, text)
    if numbers:
        raw = numbers[-1]
        cleaned = raw.replace(',', '')
        try:
            val = float(cleaned)
            return {"value": val, "method": "last_number_fallback", "raw": raw}
        except ValueError:
            pass

    return {"value": None, "method": "extraction_failed", "raw": ""}


def parse_number_from_string(s: str) -> float | None:
    """Parse a number from a string that may contain units and formatting."""
    s = s.strip()

    # Try scientific notation first: 3.2 × 10^19
    sci_match = re.search(r'(\d+\.?\d*)\s*[×x\*]\s*10\s*\^\s*\{?\s*([+-]?\d+)\s*\}?', s)
    if sci_match:
        return float(sci_match.group(1)) * 10**int(sci_match.group(2))

    # Try e-notation: 3.2e19
    e_match = re.search(r'(\d+\.?\d*)[eE]([+-]?\d+)', s)
    if e_match:
        return float(e_match.group(1)) * 10**int(e_match.group(2))

    # Try plain number (with possible commas)
    num_match = re.search(r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.?\d*', s)
    if num_match:
        return float(num_match.group(0).replace(',', ''))

    return None


# ---------------------------------------------------------------------------
# FP Score
# ---------------------------------------------------------------------------

def calculate_fp_score(predicted: float | None, actual_oom: int) -> dict:
    """
    Calculate FP score: max(0, 1 - (1/3) * |log10(predicted / actual)|)

    Args:
        predicted: extracted numerical answer (or None if extraction failed)
        actual_oom: order of magnitude of the ground truth (e.g., 19 means 10^19)

    Returns dict with: score (float|None), predicted_oom (float|None), error_oom (float|None)
    """
    actual = 10.0 ** actual_oom

    # --- Handle edge cases explicitly, not silently ---
    if predicted is None:
        return {
            "score": None,
            "predicted_oom": None,
            "error_oom": None,
            "note": "extraction failed — no numerical answer found",
        }

    if predicted == 0:
        return {
            "score": 0.0,
            "predicted_oom": None,
            "error_oom": None,
            "note": "predicted was 0 — cannot take log10(0)",
        }

    if predicted < 0:
        return {
            "score": 0.0,
            "predicted_oom": None,
            "error_oom": None,
            "note": f"predicted was negative ({predicted}) — nonsensical for this quantity",
        }

    predicted_oom = math.log10(predicted)
    actual_oom_f = float(actual_oom)
    error_oom = abs(predicted_oom - actual_oom_f)
    score = max(0.0, 1.0 - (1.0 / 3.0) * error_oom)

    return {
        "score": round(score, 4),
        "predicted_oom": round(predicted_oom, 2),
        "error_oom": round(error_oom, 2),
        "note": None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load ground truths
    with open(PROBLEMS_PATH) as f:
        problems = {p["id"]: p for p in json.load(f)}

    # Find all per-model result files
    result_files = sorted(RESULTS_DIR.glob("*.json"))
    result_files = [f for f in result_files if f.name != "all_results.json"
                    and f.name != "scored_results.json"]

    if not result_files:
        print("No result files found in results/. Run run_frontier.py first.")
        return

    all_scored = {}

    for result_file in result_files:
        model_name = result_file.stem
        print(f"\n{'='*60}")
        print(f"  Scoring: {model_name}")
        print(f"{'='*60}")

        with open(result_file) as f:
            results = json.load(f)

        scored = []
        for r in results:
            prob = problems[r["id"]]

            extraction = extract_final_answer(r["model_response"])
            scoring = calculate_fp_score(extraction["value"], prob["ground_truth_OoM"])

            entry = {
                "id": r["id"],
                "question": r["question"][:80] + "...",
                "ground_truth_OoM": prob["ground_truth_OoM"],
                "extraction": extraction,
                "scoring": scoring,
            }
            scored.append(entry)

            status = f"score={scoring['score']}" if scoring["score"] is not None else scoring["note"]
            print(f"  Q{r['id']}: extracted={extraction['value']} "
                  f"(via {extraction['method']}) | {status}")

        all_scored[model_name] = scored

    # Save
    output_path = RESULTS_DIR / "scored_results.json"
    with open(output_path, "w") as f:
        json.dump(all_scored, f, indent=2)
    print(f"\nScored results saved -> {output_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<25} {'Avg Score':>10} {'Failures':>10} {'Avg OoM Error':>15}")
    print(f"  {'-'*60}")
    for model_name, scored in all_scored.items():
        scores = [s["scoring"]["score"] for s in scored if s["scoring"]["score"] is not None]
        errors = [s["scoring"]["error_oom"] for s in scored if s["scoring"]["error_oom"] is not None]
        failures = sum(1 for s in scored if s["scoring"]["score"] is None)
        avg_score = sum(scores) / len(scores) if scores else 0
        avg_error = sum(errors) / len(errors) if errors else 0
        print(f"  {model_name:<25} {avg_score:>10.3f} {failures:>10} {avg_error:>15.1f}")


if __name__ == "__main__":
    main()
