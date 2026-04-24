"""Baseline sampling against a Tinker-hosted model using the frozen contract.

Usage:
    python3 src/run_baseline_tinker.py --split eval_v1 --model Qwen/Qwen3-4B-Instruct-2507
    python3 src/run_baseline_tinker.py --split val_v1  --model Qwen/Qwen3-4B-Instruct-2507

Writes:
    datasets/eval/baseline_<model_slug>_<split>.jsonl   — per-item outputs
    experiments/runs.jsonl                              — one summary row
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from tinker import SamplingParams, ServiceClient  # noqa: E402
from tinker_cookbook import model_info  # noqa: E402
from tinker_cookbook.renderers import get_renderer  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402

from prompt_contract import (  # noqa: E402
    CONTRACT_VERSION,
    SAMPLING_PARAMS,
    build_messages,
)
from fp_score import score_row  # noqa: E402
from eval_loader import load_split  # noqa: E402


def slugify(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


async def sample_one(sc, renderer, tokenizer, row: dict[str, Any],
                     sampling_params: SamplingParams) -> dict[str, Any]:
    msgs = build_messages(row["question"], row.get("unit"))
    prompt = renderer.build_generation_prompt(msgs)
    t0 = time.time()
    try:
        resp = await sc.sample_async(
            prompt=prompt, num_samples=1, sampling_params=sampling_params,
        )
        seq = resp.sequences[0]
        text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
        # Trim trailing stop tokens for cleaner decode
        status_sample = "ok"
        err = ""
    except Exception as e:  # network / server
        text = ""
        status_sample = "sample_error"
        err = repr(e)
    dt = time.time() - t0

    score_res = score_row(
        text,
        answer_type=row.get("answer_type", "numeric"),
        gold_numeric=row.get("gold_numeric"),
        gold_oom=row.get("gold_oom"),
        gold_unit=row.get("unit"),
    ) if status_sample == "ok" else {
        "score": 0.0, "status": "sample_error", "reason": err,
        "pred_value": None, "pred_unit": None, "pred_oom": None,
    }

    return {
        "id": row["id"],
        "source": row.get("source"),
        "domain": row.get("domain"),
        "answer_type": row.get("answer_type", "numeric"),
        "question": row["question"],
        "gold_numeric": row.get("gold_numeric"),
        "gold_oom": row.get("gold_oom"),
        "gold_unit": row.get("unit"),
        "response_text": text,
        "latency_s": round(dt, 3),
        **score_res,
    }


async def run(split: str, model: str, drop_suspect: bool, concurrency: int,
              limit: int | None) -> dict[str, Any]:
    rows, dropped = load_split(split, drop_suspect=drop_suspect)
    if limit:
        rows = rows[:limit]
    print(f"[{split}] loaded {len(rows)} rows ({len(dropped)} suspect dropped)")

    svc = ServiceClient()
    renderer_name = model_info.get_recommended_renderer_name(model)
    tokenizer = get_tokenizer(model)
    renderer = get_renderer(renderer_name, tokenizer)
    sc = svc.create_sampling_client(base_model=model)

    stop_tokens = renderer.get_stop_sequences()
    sampling_params = SamplingParams(
        max_tokens=SAMPLING_PARAMS["max_tokens"],
        temperature=SAMPLING_PARAMS["temperature"],
        top_p=SAMPLING_PARAMS["top_p"],
        seed=SAMPLING_PARAMS["seed"],
        stop=stop_tokens,
    )

    sem = asyncio.Semaphore(concurrency)

    async def bounded(row):
        async with sem:
            return await sample_one(sc, renderer, tokenizer, row, sampling_params)

    results: list[dict[str, Any]] = []
    t0 = time.time()
    done = 0
    for coro in asyncio.as_completed([bounded(r) for r in rows]):
        res = await coro
        results.append(res)
        done += 1
        if done % 10 == 0 or done == len(rows):
            so_far = [r for r in results if r["status"] == "ok"]
            mean = sum(r["score"] for r in so_far) / max(1, len(so_far))
            print(f"  {done}/{len(rows)}  mean_score_ok={mean:.3f}  "
                  f"elapsed={time.time()-t0:.1f}s")
    total_dt = time.time() - t0

    # Stable order by id
    results.sort(key=lambda r: r["id"])

    # Write outputs
    slug = slugify(model)
    out_path = os.path.join(ROOT, "datasets", "eval", f"baseline_{slug}_{split}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Summary
    n = len(results)
    scores = [r["score"] for r in results]
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    ok_scores = [r["score"] for r in results if r["status"] == "ok"]
    summary = {
        "run": f"baseline_{slug}_{split}",
        "contract_version": CONTRACT_VERSION,
        "model": model,
        "renderer": renderer_name,
        "split": split,
        "drop_suspect": drop_suspect,
        "n": n,
        "dropped_suspect": sorted(dropped),
        "mean_score": round(sum(scores) / n, 4) if n else 0.0,
        "mean_score_ok_only": round(sum(ok_scores) / len(ok_scores), 4) if ok_scores else 0.0,
        "status_counts": by_status,
        "sampling_params": SAMPLING_PARAMS,
        "wall_time_s": round(total_dt, 1),
        "output_path": os.path.relpath(out_path, ROOT),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(json.dumps(summary, indent=2))

    runs_path = os.path.join(ROOT, "experiments", "runs.jsonl")
    os.makedirs(os.path.dirname(runs_path), exist_ok=True)
    with open(runs_path, "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"Appended summary → {runs_path}")

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["eval_v1", "val_v1"], required=True)
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--keep-suspect", action="store_true",
                   help="Do not drop suspect eval items")
    args = p.parse_args()

    asyncio.run(run(
        split=args.split, model=args.model,
        drop_suspect=not args.keep_suspect,
        concurrency=args.concurrency, limit=args.limit,
    ))


if __name__ == "__main__":
    main()
