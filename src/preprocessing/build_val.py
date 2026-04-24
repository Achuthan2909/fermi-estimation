"""Build val_v1 — the development/iteration split (~100 items).

Design:
  - Drawn from the same clean pool as eval_v1, but excludes every eval_v1 id.
  - Matches eval_v1's overall per-domain distribution as closely as possible
    given what remains (mainly: realFP-test mathematics is exhausted, so
    math items shift to scioly).
  - Different seed from eval_v1, so independent.
  - Not sealed — val is meant to be iterated against. Still SHA-pinned for
    run-reproducibility.

Inputs:
  datasets/processed/classified.jsonl
  datasets/eval/excluded_ids.json         (so we don't re-sample eval items)

Outputs:
  datasets/eval/val_v1.jsonl
  datasets/eval/val_v1.sha256
  datasets/eval/val_v1_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

# Reuse the sampling fn from build_eval
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_eval import sample_stratified  # type: ignore

REPO = Path(__file__).resolve().parent.parent
POOL_PATH = REPO / "datasets" / "processed" / "classified.jsonl"
OUT_DIR = REPO / "datasets" / "eval"
EXCLUDED_PATH = OUT_DIR / "excluded_ids.json"

# Targets mirror eval_v1 in overall per-domain counts (23 phys, 17 bio, 11 chem,
# 17 earth, 11 astro, 13 other, 8 math) but shift math from realFP-test to
# scioly because only 1 realFP-test math item remains after eval_v1.
TARGETS: dict[tuple[str, str], int] = {
    # realFP-test : 65 items
    ("realFP:test", "physics"):       15,
    ("realFP:test", "biology"):       12,
    ("realFP:test", "chemistry"):      7,
    ("realFP:test", "earth_science"): 12,
    ("realFP:test", "astronomy"):      8,
    ("realFP:test", "other_science"): 10,
    ("realFP:test", "mathematics"):    1,
    # scioly : 35 items
    ("scioly", "physics"):             8,
    ("scioly", "biology"):             5,
    ("scioly", "chemistry"):           4,
    ("scioly", "earth_science"):       5,
    ("scioly", "astronomy"):           3,
    ("scioly", "other_science"):       3,
    ("scioly", "mathematics"):         7,
}

SEED = 19730717  # distinct from eval seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(POOL_PATH))
    ap.add_argument("--excluded", default=str(EXCLUDED_PATH))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    pool = [json.loads(l) for l in Path(args.pool).open() if l.strip()]
    excluded = set(json.loads(Path(args.excluded).read_text()).get("eval_v1", []))
    pool = [r for r in pool if r["id"] not in excluded]
    print(f"Pool after eval-exclusion: {len(pool)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_path = out_dir / "val_v1.jsonl"
    sha_path = out_dir / "val_v1.sha256"
    manifest_path = out_dir / "val_v1_manifest.json"

    if val_path.exists() and not args.force:
        raise SystemExit(f"ERROR: {val_path} already exists. Use --force to rebuild.")

    picked = sample_stratified(pool, TARGETS, seed=args.seed)
    print(f"Sampled: {len(picked)}")

    with val_path.open("w") as f:
        for r in picked:
            slim = {
                "id": r["id"], "question": r["question"],
                "gold_numeric": r["gold_numeric"], "gold_oom": r["gold_oom"],
                "answer_type": r["answer_type"], "unit": r["unit"],
                "source": r["source"], "orig_split": r.get("orig_split"),
                "orig_source": r.get("orig_source"), "domain": r["domain"],
            }
            f.write(json.dumps(slim, ensure_ascii=False) + "\n")

    sha = hashlib.sha256(val_path.read_bytes()).hexdigest()
    sha_path.write_text(sha + "\n")

    manifest = {
        "version": "v1",
        "seed": args.seed,
        "pool_source": str(Path(args.pool).relative_to(REPO)),
        "excluded_source": str(Path(args.excluded).relative_to(REPO)),
        "pool_size_after_exclusion": len(pool),
        "targets": {f"{k[0]}/{k[1]}": v for k, v in TARGETS.items()},
        "target_total": sum(TARGETS.values()),
        "actual_total": len(picked),
        "sha256": sha,
        "items": [
            {"id": r["id"], "source": r["source"], "orig_split": r.get("orig_split"),
             "orig_source": r.get("orig_source"), "domain": r["domain"],
             "answer_type": r["answer_type"], "gold_oom": r["gold_oom"],
             "classifier_model": r.get("classifier_model")}
            for r in picked
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Update excluded_ids.json so training pipeline can skip val items too.
    exc_data = json.loads(Path(args.excluded).read_text())
    exc_data["val_v1"] = sorted(r["id"] for r in picked)
    Path(args.excluded).write_text(json.dumps(exc_data, indent=2))

    by_src = Counter(r["source"] for r in picked)
    by_domain = Counter(r["domain"] for r in picked)
    by_ans = Counter(r["answer_type"] for r in picked)
    print()
    print(f"Wrote:  {val_path.relative_to(REPO)}  (sha256={sha[:12]}...)")
    print(f"Wrote:  {sha_path.relative_to(REPO)}")
    print(f"Wrote:  {manifest_path.relative_to(REPO)}")
    print(f"Updated {Path(args.excluded).relative_to(REPO)} with val_v1 ids")
    print()
    print("Breakdown:")
    print("  by source   :", dict(by_src))
    print("  by domain   :", dict(by_domain))
    print("  by ans-type :", dict(by_ans))


if __name__ == "__main__":
    main()
