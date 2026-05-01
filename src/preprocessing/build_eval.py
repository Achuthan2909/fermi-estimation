"""Build and freeze eval_v1 — a sealed 100-item evaluation set.

Design principles (reviewer asked for these explicitly):
  - Codified BEFORE any training run.
  - Drawn from human-curated sources only (RealFP-test + scioly) — no synthFP,
    no LLM-generated questions.
  - Deterministic sampling (fixed seed) so the same eval_v1.jsonl is reproducible.
  - Hash-pinned — sha256 of the final file is written to disk. Any drift detectable.
  - Per-item provenance in a manifest file.
  - Excluded-ID list so the training pipeline can skip these items.

Inputs:
  datasets/processed/classified.jsonl        (science-filtered, synthFP-dropped pool)

Outputs:
  datasets/eval/eval_v1.jsonl                (the 100 items, sealed)
  datasets/eval/eval_v1.sha256               (hash of eval_v1.jsonl)
  datasets/eval/eval_v1_manifest.json        (config + per-item provenance)
  datasets/eval/excluded_ids.json            (IDs to exclude from training)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
POOL_PATH = REPO / "datasets" / "processed" / "classified.jsonl"
OUT_DIR = REPO / "datasets" / "eval"

# Target per (source, domain). Sum = 100.
TARGETS: dict[tuple[str, str], int] = {
    # realFP-test : 70 items
    ("realFP:test", "physics"):       15,
    ("realFP:test", "biology"):       12,
    ("realFP:test", "chemistry"):      8,
    ("realFP:test", "earth_science"): 12,
    ("realFP:test", "astronomy"):      8,
    ("realFP:test", "other_science"): 10,
    ("realFP:test", "mathematics"):    5,
    # scioly : 30 items
    ("scioly", "physics"):             8,
    ("scioly", "biology"):             5,
    ("scioly", "chemistry"):           3,
    ("scioly", "earth_science"):       5,
    ("scioly", "astronomy"):           3,
    ("scioly", "other_science"):       3,
    ("scioly", "mathematics"):         3,
}

SEED = 20260423  # change only if you explicitly want a new eval set


def load_pool(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def stratum_key(row: dict) -> str:
    if row["source"] == "realFP" and row.get("orig_split") == "test":
        return "realFP:test"
    if row["source"] == "scioly":
        return "scioly"
    return "_other_"


def sample_stratified(pool: list[dict], targets: dict[tuple[str, str], int], seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_stratum: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in pool:
        s = stratum_key(r)
        if s == "_other_":
            continue
        by_stratum[(s, r["domain"])].append(r)

    picked: list[dict] = []
    shortfalls: list[tuple[tuple[str, str], int, int]] = []
    for key, n_want in targets.items():
        pool_k = by_stratum.get(key, [])
        if len(pool_k) < n_want:
            shortfalls.append((key, n_want, len(pool_k)))
        # Sort for deterministic ordering before sampling.
        pool_k_sorted = sorted(pool_k, key=lambda r: r["id"])
        n_take = min(n_want, len(pool_k_sorted))
        picked.extend(rng.sample(pool_k_sorted, k=n_take))

    if shortfalls:
        print("WARNING: target shortfalls:")
        for key, want, got in shortfalls:
            print(f"  {key}: wanted {want}, only {got} available")

    # Deterministic final ordering by id.
    picked.sort(key=lambda r: r["id"])
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(POOL_PATH))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing eval_v1.jsonl. Refuse otherwise.")
    args = ap.parse_args()

    pool = load_pool(Path(args.pool))
    print(f"Loaded pool: {len(pool)} items from {args.pool}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = out_dir / "eval_v1.jsonl"
    sha_path = out_dir / "eval_v1.sha256"
    manifest_path = out_dir / "eval_v1_manifest.json"
    excluded_path = out_dir / "excluded_ids.json"

    if eval_path.exists() and not args.force:
        raise SystemExit(
            f"ERROR: {eval_path} already exists. The eval set is meant to be sealed.\n"
            f"Re-run with --force only if you know you want to rebuild."
        )

    picked = sample_stratified(pool, TARGETS, seed=args.seed)
    print(f"Sampled: {len(picked)} items")

    # Write eval_v1.jsonl — one object per line, id-sorted for stable hash.
    with eval_path.open("w") as f:
        for r in picked:
            # Strip classifier metadata so the eval file is provenance-clean.
            slim = {
                "id": r["id"],
                "question": r["question"],
                "gold_numeric": r["gold_numeric"],
                "gold_oom": r["gold_oom"],
                "answer_type": r["answer_type"],
                "unit": r["unit"],
                "source": r["source"],
                "orig_split": r.get("orig_split"),
                "orig_source": r.get("orig_source"),
                "domain": r["domain"],
            }
            f.write(json.dumps(slim, ensure_ascii=False) + "\n")

    sha = hashlib.sha256(eval_path.read_bytes()).hexdigest()
    sha_path.write_text(sha + "\n")

    # Manifest: full config + per-item provenance (includes classifier model)
    manifest = {
        "version": "v1",
        "seed": args.seed,
        "pool_source": str(Path(args.pool).relative_to(REPO)),
        "pool_size": len(pool),
        "targets": {f"{k[0]}/{k[1]}": v for k, v in TARGETS.items()},
        "target_total": sum(TARGETS.values()),
        "actual_total": len(picked),
        "sha256": sha,
        "items": [
            {
                "id": r["id"],
                "source": r["source"],
                "orig_split": r.get("orig_split"),
                "orig_source": r.get("orig_source"),
                "domain": r["domain"],
                "answer_type": r["answer_type"],
                "gold_oom": r["gold_oom"],
                "classifier_model": r.get("classifier_model"),
            }
            for r in picked
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    excluded_ids = sorted(r["id"] for r in picked)
    excluded_path.write_text(json.dumps({"eval_v1": excluded_ids}, indent=2))

    # Summary print
    by_src = Counter(r["source"] for r in picked)
    by_domain = Counter(r["domain"] for r in picked)
    by_ans = Counter(r["answer_type"] for r in picked)
    print()
    print(f"Wrote:  {eval_path.relative_to(REPO)}  (sha256={sha[:12]}...)")
    print(f"Wrote:  {sha_path.relative_to(REPO)}")
    print(f"Wrote:  {manifest_path.relative_to(REPO)}")
    print(f"Wrote:  {excluded_path.relative_to(REPO)}")
    print()
    print("Breakdown:")
    print("  by source   :", dict(by_src))
    print("  by domain   :", dict(by_domain))
    print("  by ans-type :", dict(by_ans))


if __name__ == "__main__":
    main()
