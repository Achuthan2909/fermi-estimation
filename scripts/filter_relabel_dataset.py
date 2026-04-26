"""Filter a verified dataset: drop uncertain + high-OOM-error rows, relabel
the rest of the incorrect rows (within OOM threshold) using the verifier's
predicted OOM, and split out low-confidence relabels for manual review.

Per-record decisions (driven by the merged verification file):
  - correct                                                    → keep, unchanged
  - uncertain                                                  → drop
  - incorrect, |oom_error| > threshold                         → drop
  - incorrect, |oom_error| <= threshold, conf in {high,medium} → relabel
      (provided_ground_truth := verifier's pred_oom) and keep
  - incorrect, |oom_error| <= threshold, conf low/missing      → review file

Outputs are full verification records (the same shape as the merged
verification JSON's `results` entries) so the search plan, computation,
and comparison stay attached to each kept row. Relabeled rows get
`provided_ground_truth` overwritten with the verifier's `pred_oom` and a
`_relabel` audit field describing the change.

Example:
    python scripts/filter_relabel_dataset.py \
        --dataset datasets/final/train.json \
        --verifications run/verification_train_all.json \
        --output datasets/final/train_filtered.json \
        --review datasets/final/train_review.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

OOM_RE = re.compile(r"oom_error=(-?\d+(?:\.\d+)?)")


def oom_err(reason: str | None) -> float | None:
    if not reason:
        return None
    m = OOM_RE.search(reason)
    return float(m.group(1)) if m else None


def best_verification(record: dict) -> dict | None:
    """Pick the verification with the smallest |oom_error| (ties: first)."""
    best = None
    best_abs = None
    for v in record.get("verifications", []):
        comp = v.get("comparison") or {}
        oe = oom_err(comp.get("reason", ""))
        if oe is None:
            continue
        a = abs(oe)
        if best_abs is None or a < best_abs:
            best_abs = a
            best = v
    if best is None and record.get("verifications"):
        best = record["verifications"][0]
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="Original dataset JSON (list of rows).")
    ap.add_argument("--verifications", required=True, help="Merged verification JSON.")
    ap.add_argument("--output", required=True, help="Filtered + relabeled dataset out.")
    ap.add_argument("--review", required=True, help="Low-confidence relabel candidates out.")
    ap.add_argument("--oom-threshold", type=float, default=2.0)
    ap.add_argument(
        "--keep-unverified",
        action="store_true",
        help="Keep dataset rows that have no verification record (default: keep them).",
    )
    args = ap.parse_args()

    with open(args.dataset) as f:
        dataset = json.load(f)
    with open(args.verifications) as f:
        verifs = json.load(f)

    by_id = {r["id"]: r for r in verifs.get("results", [])}
    by_question: dict[str, dict] = {}
    for r in verifs.get("results", []):
        q = (r.get("question") or "").strip()
        if q:
            by_question.setdefault(q, r)
    stats: Counter = Counter()

    kept: list[dict] = []
    review: list[dict] = []

    for row in dataset:
        rid = row.get("id")
        vrec = by_id.get(rid) if rid is not None else None
        if vrec is None:
            q = (row.get("question") or "").strip()
            if q:
                vrec = by_question.get(q)
                if vrec is not None:
                    stats["matched_by_question"] += 1
        if vrec is None:
            stats["unverified"] += 1
            if args.keep_unverified:
                # Carry the dataset row through as a minimal record.
                kept.append({"_unverified_row": row})
            continue

        verdict = vrec.get("consensus_verdict")
        if verdict == "correct":
            stats["correct_kept"] += 1
            kept.append(vrec)
            continue
        if verdict == "uncertain":
            stats["dropped_uncertain"] += 1
            continue
        if verdict != "incorrect":
            stats[f"dropped_unknown:{verdict}"] += 1
            continue

        v = best_verification(vrec)
        comp = (v or {}).get("comparison") or {}
        oe = oom_err(comp.get("reason", ""))
        if oe is None or abs(oe) > args.oom_threshold:
            stats["dropped_high_oom"] += 1
            continue

        pred_oom = comp.get("pred_oom")
        if pred_oom is None:
            stats["dropped_no_pred_oom"] += 1
            continue

        parsed = (v.get("computation") or {}).get("parsed_response") or {}
        confidence = (parsed.get("confidence") or "").lower()

        new_rec = dict(vrec)
        new_rec["_relabel"] = {
            "original_provided_ground_truth": vrec.get("provided_ground_truth"),
            "original_dataset_fermi_answer": row.get("fermi_answer"),
            "pred_value": comp.get("pred_value"),
            "pred_unit": comp.get("pred_unit"),
            "pred_oom": pred_oom,
            "oom_error": oe,
            "confidence": confidence or None,
            "verifier_model": v.get("model"),
        }
        new_rec["provided_ground_truth"] = pred_oom

        if confidence in ("high", "medium"):
            stats[f"relabeled_{confidence}"] += 1
            kept.append(new_rec)
        else:
            stats[f"review_{confidence or 'missing'}"] += 1
            review.append(new_rec)

    with open(args.output, "w") as f:
        json.dump(kept, f, indent=2)
        f.write("\n")
    with open(args.review, "w") as f:
        json.dump(review, f, indent=2)
        f.write("\n")

    print(f"input rows         : {len(dataset)}")
    print(f"kept (output)      : {len(kept)}  -> {args.output}")
    print(f"review (low-conf)  : {len(review)}  -> {args.review}")
    print("breakdown:")
    for k in sorted(stats):
        print(f"  {k:<28} {stats[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
