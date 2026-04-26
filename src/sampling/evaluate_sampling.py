"""Run Tinker sampling and evaluate saved Fermi outputs.

Normal mode:
    python src/sampling/evaluate_sampling.py --limit 10

Evaluate an existing output file without sampling again:
    python src/sampling/evaluate_sampling.py --evaluate-only

Writes one JSON object to ``run/evaluation.json`` with metadata, per-row results,
and a summary. The model never sees the ground-truth answer.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from prompt_contract import (  # noqa: E402
    CONTRACT_VERSION,
    ParseError,
    SAMPLING_PARAMS,
    build_messages,
    parse_response,
)

_DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
_DEFAULT_DATASET = os.path.join(ROOT, "datasets", "final", "train.json")
_DEFAULT_OUTPUT = os.path.join(ROOT, "run", "evaluation.json")


def load_rows(dataset_path: str, offset: int = 0, limit: int | None = None) -> list[dict[str, Any]]:
    with open(dataset_path) as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"expected a JSON list in {dataset_path}")
    end = None if limit is None else offset + limit
    return rows[offset:end]


def normalize_oom(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def infer_oom(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric == 0 or not math.isfinite(numeric):
        return None
    return int(math.floor(math.log10(abs(numeric))))


def score_oom(pred_oom: int | None, gold_oom: int | None) -> tuple[float, str]:
    if pred_oom is None:
        return 0.0, "pred_oom_missing"
    if gold_oom is None:
        return 0.0, "gold_oom_missing"
    error = abs(pred_oom - gold_oom)
    return max(0.0, 1.0 - error), f"oom_error={pred_oom - gold_oom}"


def score_numeric(pred_value: Any, gold_value: Any) -> tuple[float, str]:
    try:
        pred = abs(float(pred_value))
        gold = abs(float(gold_value))
    except (TypeError, ValueError):
        return 0.0, "numeric_parse_fail"
    if pred == 0 or gold == 0 or not math.isfinite(pred) or not math.isfinite(gold):
        return 0.0, "nonfinite_or_zero"
    delta = abs(math.log10(pred) - math.log10(gold))
    return max(0.0, 1.0 - delta), f"log10_error={delta:.3f}"


def score_record(record: dict[str, Any]) -> dict[str, Any]:
    """Score one saved sampling record against its ground truth."""
    if record.get("sample_status") != "ok":
        return {
            "score": 0.0,
            "eval_status": record.get("sample_status", "sample_error"),
            "eval_reason": record.get("sample_error", ""),
            "pred_value": None,
            "pred_unit": None,
            "pred_oom": None,
        }

    try:
        parsed = record.get("parsed_response") or parse_response(record["response_text"])
    except (KeyError, ParseError) as e:
        return {
            "score": 0.0,
            "eval_status": "parse_fail",
            "eval_reason": str(e),
            "pred_value": None,
            "pred_unit": None,
            "pred_oom": None,
        }

    final = parsed["final_answer"]
    pred_value = final.get("value")
    pred_unit = final.get("unit") or ""
    pred_oom = normalize_oom(final.get("oom"))
    if pred_oom is None:
        pred_oom = infer_oom(pred_value)

    answer_type = record.get("answer_type") or "oom_exponent"
    if answer_type == "numeric" and record.get("gold_numeric") is not None:
        score, reason = score_numeric(pred_value, record["gold_numeric"])
    else:
        gold_oom = normalize_oom(record.get("ground_truth_answer"))
        score, reason = score_oom(pred_oom, gold_oom)

    return {
        "score": score,
        "eval_status": "ok",
        "eval_reason": reason,
        "pred_value": pred_value,
        "pred_unit": pred_unit,
        "pred_oom": pred_oom,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    scores = [float(r.get("score", 0.0)) for r in results]
    status_counts: dict[str, int] = {}
    exact_oom = 0
    oom_count = 0
    for result in results:
        status = result.get("eval_status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        if result.get("answer_type", "oom_exponent") != "numeric":
            oom_count += 1
            if result.get("pred_oom") == normalize_oom(result.get("ground_truth_answer")):
                exact_oom += 1
    return {
        "n": n,
        "mean_score": round(sum(scores) / n, 4) if n else 0.0,
        "exact_oom_accuracy": round(exact_oom / oom_count, 4) if oom_count else None,
        "status_counts": status_counts,
    }


def make_record(row: dict[str, Any], index: int, response_text: str, latency_s: float) -> dict[str, Any]:
    try:
        parsed = parse_response(response_text)
        sample_status = "ok"
        sample_error = ""
    except ParseError as e:
        parsed = None
        sample_status = "parse_fail"
        sample_error = str(e)

    return {
        "index": index,
        "id": row.get("id", str(index)),
        "question": row["question"],
        "unit": row.get("unit"),
        "domain": row.get("domain"),
        "source": row.get("source"),
        "answer_type": row.get("answer_type", "oom_exponent"),
        "ground_truth_answer": row.get("fermi_answer"),
        "gold_numeric": row.get("gold_numeric"),
        "response_text": response_text,
        "parsed_response": parsed,
        "latency_s": round(latency_s, 3),
        "sample_status": sample_status,
        "sample_error": sample_error,
    }


async def sample_rows(
    rows: list[dict[str, Any]],
    *,
    model: str,
    concurrency: int,
    max_tokens: int,
    offset: int,
) -> tuple[list[dict[str, Any]], str]:
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not set. Export it or `source .env` first.")

    from tinker import SamplingParams, ServiceClient
    from tinker_cookbook import model_info
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    svc = ServiceClient()
    renderer_name = model_info.get_recommended_renderer_name(model)
    tokenizer = get_tokenizer(model)
    renderer = get_renderer(renderer_name, tokenizer)
    sc = svc.create_sampling_client(base_model=model)
    stop_tokens = renderer.get_stop_sequences()
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=SAMPLING_PARAMS["temperature"],
        top_p=SAMPLING_PARAMS["top_p"],
        seed=SAMPLING_PARAMS["seed"],
        stop=stop_tokens,
    )
    sem = asyncio.Semaphore(concurrency)

    async def sample_one(row: dict[str, Any], local_index: int) -> dict[str, Any]:
        async with sem:
            t0 = time.time()
            try:
                messages = build_messages(row["question"], row.get("unit"))
                prompt = renderer.build_generation_prompt(messages)
                resp = await sc.sample_async(
                    prompt=prompt,
                    num_samples=1,
                    sampling_params=sampling_params,
                )
                text = tokenizer.decode(resp.sequences[0].tokens, skip_special_tokens=True)
                return make_record(row, offset + local_index, text, time.time() - t0)
            except Exception as e:
                record = make_record(row, offset + local_index, "", time.time() - t0)
                record["sample_status"] = "sample_error"
                record["sample_error"] = repr(e)
                return record

    tasks = [sample_one(row, i) for i, row in enumerate(rows)]
    results: list[dict[str, Any]] = []
    for done, coro in enumerate(asyncio.as_completed(tasks), start=1):
        result = await coro
        result.update(score_record(result))
        results.append(result)
        print(
            f"{done}/{len(rows)} id={result['id']} "
            f"score={result['score']:.3f} status={result['eval_status']}"
        )
    results.sort(key=lambda r: r["index"])
    return results, renderer_name


def write_output(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def evaluate_existing(path: str) -> dict[str, Any]:
    with open(path) as f:
        payload = json.load(f)
    results = payload.get("results", [])
    for result in results:
        result.update(score_record(result))
    payload["summary"] = summarize(results)
    return payload


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.evaluate_only:
        payload = evaluate_existing(args.output)
        write_output(args.output, payload)
        return payload

    rows = load_rows(args.dataset, offset=args.offset, limit=args.limit)
    t0 = time.time()
    results, renderer_name = await sample_rows(
        rows,
        model=args.model,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        offset=args.offset,
    )
    sampling_params_for_run = dict(SAMPLING_PARAMS)
    sampling_params_for_run["max_tokens"] = args.max_tokens
    payload = {
        "metadata": {
            "contract_version": CONTRACT_VERSION,
            "model": args.model,
            "renderer": renderer_name,
            "dataset": os.path.relpath(args.dataset, ROOT),
            "offset": args.offset,
            "limit": args.limit,
            "sampling_params": sampling_params_for_run,
            "wall_time_s": round(time.time() - t0, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "summary": summarize(results),
        "results": results,
    }
    write_output(args.output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and evaluate sampling outputs.")
    parser.add_argument("--dataset", default=_DEFAULT_DATASET)
    parser.add_argument("--output", default=_DEFAULT_OUTPUT)
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(SAMPLING_PARAMS["max_tokens"]),
        help="Generation token cap for this run.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Score an existing output JSON without calling Tinker.",
    )
    args = parser.parse_args()

    payload = asyncio.run(run(args))
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
