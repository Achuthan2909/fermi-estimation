"""Verify provided Fermi ground truths with one or two Tinker search loops.

For each row and each verifier model:
1. Ask the model what constants/facts it needs and what to search.
2. Run those searches in Python.
3. Give the retrieved context back to the same model and ask it to compute,
   separating cited facts from memory-derived estimates.
4. Compare the computed answer to the hidden provided ground truth in Python.

The model never sees ``fermi_answer``.
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
SRC_ROOT = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(SRC_ROOT)
sys.path.insert(0, SRC_ROOT)

from verifier.prompt_contract import (  # noqa: E402
    CONTRACT_VERSION,
    ParseError,
    build_compute_messages,
    build_search_plan_messages,
    parse_compute_response,
    parse_search_plan_response,
)
from search.web_search import search_web  # noqa: E402

_DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
_DEFAULT_DATASET = os.path.join(PROJECT_ROOT, "datasets", "final", "train.json")
_DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "run", "verification.json")

SAMPLING_PARAMS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 512,
    "seed": 0,
}


def load_rows(dataset_path: str, offset: int, limit: int | None) -> list[dict[str, Any]]:
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


def verdict_from_computation(row: dict[str, Any], parsed: dict[str, Any] | None) -> dict[str, Any]:
    if parsed is None:
        return {
            "verdict": "uncertain",
            "reason": "no parsed computation",
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

    answer_type = row.get("answer_type", "oom_exponent")
    if answer_type == "numeric" and row.get("gold_numeric") is not None:
        try:
            pred = abs(float(pred_value))
            gold = abs(float(row["gold_numeric"]))
        except (TypeError, ValueError):
            verdict = "uncertain"
            reason = "numeric parse failure"
        else:
            if pred == 0 or gold == 0:
                verdict = "uncertain"
                reason = "zero value cannot be log-scored"
            else:
                log_error = abs(math.log10(pred) - math.log10(gold))
                verdict = "correct" if log_error <= 0.5 else "incorrect"
                reason = f"log10_error={log_error:.3f}"
    else:
        gold_oom = normalize_oom(row.get("fermi_answer"))
        if pred_oom is None or gold_oom is None:
            verdict = "uncertain"
            reason = "missing pred_oom or gold_oom"
        else:
            oom_error = pred_oom - gold_oom
            verdict = "correct" if oom_error == 0 else "incorrect"
            reason = f"oom_error={oom_error}"

    return {
        "verdict": verdict,
        "reason": reason,
        "pred_value": pred_value,
        "pred_unit": pred_unit,
        "pred_oom": pred_oom,
    }


def normalize_verdict(verification: dict[str, Any]) -> str:
    verdict = (verification.get("comparison") or {}).get("verdict")
    if verdict in {"correct", "incorrect", "uncertain"}:
        return verdict
    return "uncertain"


def consensus_verdict(verifications: list[dict[str, Any]]) -> str:
    verdicts = [normalize_verdict(v) for v in verifications]
    if not verdicts:
        return "no_verifiers"
    if all(verdict == verdicts[0] for verdict in verdicts):
        return verdicts[0]
    if "correct" in verdicts and "incorrect" in verdicts:
        return "disagreement"
    return "uncertain"


async def call_tinker_model(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> tuple[str, str | None, float]:
    from tinker import SamplingParams, ServiceClient
    from tinker_cookbook import model_info
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    t0 = time.time()
    svc = ServiceClient()
    renderer_name = model_info.get_recommended_renderer_name(model)
    tokenizer = get_tokenizer(model)
    renderer = get_renderer(renderer_name, tokenizer)
    sc = svc.create_sampling_client(base_model=model)
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=SAMPLING_PARAMS["temperature"],
        top_p=SAMPLING_PARAMS["top_p"],
        seed=SAMPLING_PARAMS["seed"],
        stop=renderer.get_stop_sequences(),
    )
    prompt = renderer.build_generation_prompt(messages)
    response = await sc.sample_async(
        prompt=prompt,
        num_samples=1,
        sampling_params=sampling_params,
    )
    text = tokenizer.decode(response.sequences[0].tokens, skip_special_tokens=True)
    return text, renderer_name, time.time() - t0


def search_query_records(
    query: str,
    *,
    max_results: int,
    timeout_s: float,
) -> dict[str, Any]:
    try:
        results = search_web(query, max_results=max_results, timeout_s=timeout_s)
        return {
            "query": query,
            "status": "ok",
            "error": "",
            "results": [result.to_dict() for result in results],
        }
    except Exception as e:
        return {"query": query, "status": "search_error", "error": repr(e), "results": []}


def format_model_search_context(plan: dict[str, Any], search_records: list[dict[str, Any]]) -> str:
    lines = ["Reference facts to consider:"]
    constants = plan.get("constants_needed") or []
    if constants:
        for constant in constants:
            if isinstance(constant, dict):
                name = constant.get("name", "unknown")
                unit = constant.get("unit") or ""
                why = constant.get("why") or ""
                lines.append(f"- {name} ({unit}): {why}".strip())
            else:
                lines.append(f"- {constant}")
    else:
        lines.append("- None listed.")

    lines.append("\nReference evidence:")
    for i, record in enumerate(search_records, start=1):
        lines.append(f"\nReference group {i}:")
        if record.get("status") != "ok":
            lines.append("No reference evidence was available for this item.")
            continue
        results = record.get("results", [])
        if not results:
            lines.append("No reference evidence was available for this item.")
            continue
        for j, item in enumerate(results, start=1):
            title = item.get("title", "").strip()
            snippet = item.get("snippet", "").strip()
            label = f"{i}.{j}"
            if title:
                lines.append(f"- Reference {label}: {title}")
            if snippet:
                lines.append(f"  Excerpt: {snippet}")
    return "\n".join(lines)


async def run_model_verifier(
    *,
    model: str,
    row: dict[str, Any],
    max_tokens: int,
    max_search_queries: int,
    search_results: int,
    search_timeout_s: float,
) -> dict[str, Any]:
    answer_type = row.get("answer_type", "oom_exponent")
    started = time.time()
    renderer_name = None
    search_plan_messages = build_search_plan_messages(
        question=row["question"],
        answer_type=answer_type,
        max_search_queries=max_search_queries,
    )

    try:
        plan_text, renderer_name, plan_latency_s = await call_tinker_model(
            model=model,
            messages=search_plan_messages,
            max_tokens=max_tokens,
        )
        plan = parse_search_plan_response(plan_text)
        plan_status = "ok"
        plan_error = ""
    except (Exception, ParseError) as e:
        plan_text = locals().get("plan_text", "")
        plan_latency_s = locals().get("plan_latency_s", 0.0)
        plan = {"constants_needed": [], "search_queries": []}
        plan_status = "plan_error"
        plan_error = repr(e)

    queries = [
        query.strip()
        for query in plan.get("search_queries", [])
        if isinstance(query, str) and query.strip()
    ][:max_search_queries]
    search_tasks = [
        asyncio.to_thread(
            search_query_records,
            query,
            max_results=search_results,
            timeout_s=search_timeout_s,
        )
        for query in queries
    ]
    search_records = await asyncio.gather(*search_tasks) if search_tasks else []
    search_context = format_model_search_context(plan, search_records)
    compute_messages = build_compute_messages(
        question=row["question"],
        answer_type=answer_type,
        search_evidence=search_context,
    )

    try:
        compute_text, renderer_name, compute_latency_s = await call_tinker_model(
            model=model,
            messages=compute_messages,
            max_tokens=max_tokens,
        )
        computation = parse_compute_response(compute_text)
        compute_status = "ok"
        compute_error = ""
    except (Exception, ParseError) as e:
        compute_text = locals().get("compute_text", "")
        compute_latency_s = locals().get("compute_latency_s", 0.0)
        computation = None
        compute_status = "compute_error"
        compute_error = repr(e)

    comparison = verdict_from_computation(row, computation)
    if plan_status != "ok" or compute_status != "ok":
        comparison["verdict"] = "uncertain"

    return {
        "model": model,
        "renderer": renderer_name,
        "latency_s": round(time.time() - started, 3),
        "search_plan": {
            "status": plan_status,
            "error": plan_error,
            "latency_s": round(plan_latency_s, 3),
            "input_messages": search_plan_messages,
            "response_text": plan_text,
            "parsed_response": plan,
        },
        "search_records": search_records,
        "search_context": search_context,
        "computation": {
            "status": compute_status,
            "error": compute_error,
            "latency_s": round(compute_latency_s, 3),
            "input_messages": compute_messages,
            "response_text": compute_text,
            "parsed_response": computation,
        },
        "comparison": comparison,
    }


async def verify_row(
    row: dict[str, Any],
    *,
    index: int,
    model_a: str,
    model_b: str | None,
    max_tokens: int,
    max_search_queries: int,
    search_results: int,
    search_timeout_s: float,
) -> dict[str, Any]:
    verifier_tasks = [
        run_model_verifier(
            model=model_a,
            row=row,
            max_tokens=max_tokens,
            max_search_queries=max_search_queries,
            search_results=search_results,
            search_timeout_s=search_timeout_s,
        )
    ]
    if model_b:
        verifier_tasks.append(
            run_model_verifier(
                model=model_b,
                row=row,
                max_tokens=max_tokens,
                max_search_queries=max_search_queries,
                search_results=search_results,
                search_timeout_s=search_timeout_s,
            )
        )
    verifications = await asyncio.gather(*verifier_tasks)

    return {
        "index": index,
        "id": row.get("id", str(index)),
        "question": row["question"],
        "answer_type": row.get("answer_type", "oom_exponent"),
        "provided_ground_truth": row.get("fermi_answer"),
        "domain": row.get("domain"),
        "source": row.get("source"),
        "verifications": verifications,
        "consensus_verdict": consensus_verdict(verifications),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    verdict_counts: dict[str, int] = {}
    stage_status_counts: dict[str, int] = {}
    for record in records:
        verdict = record.get("consensus_verdict", "unknown")
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        for verification in record.get("verifications", []):
            for stage in ("search_plan", "computation"):
                status = verification.get(stage, {}).get("status", "unknown")
                key = f"{stage}:{status}"
                stage_status_counts[key] = stage_status_counts.get(key, 0) + 1
    return {
        "n": len(records),
        "consensus_verdict_counts": verdict_counts,
        "stage_status_counts": stage_status_counts,
    }


def write_output(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def dataset_split_name(dataset_path: str) -> str:
    return os.path.splitext(os.path.basename(dataset_path))[0]


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not set. Export it or `source .env` first.")

    dataset_paths = args.datasets or [args.dataset]
    limit = None if args.all else args.limit
    records: list[dict[str, Any]] = []
    t0 = time.time()
    for dataset_path in dataset_paths:
        rows = load_rows(dataset_path, offset=args.offset, limit=limit)
        split = dataset_split_name(dataset_path)
        for i, row in enumerate(rows, start=args.offset):
            record = await verify_row(
                row,
                index=i,
                model_a=args.model_a,
                model_b=args.model_b,
                max_tokens=args.max_tokens,
                max_search_queries=args.max_search_queries,
                search_results=args.search_results,
                search_timeout_s=args.search_timeout,
            )
            record["dataset"] = os.path.relpath(dataset_path, PROJECT_ROOT)
            record["split"] = split
            records.append(record)
            print(
                f"{len(records)} total split={split} "
                f"{i - args.offset + 1}/{len(rows)} id={record['id']} "
                f"consensus={record['consensus_verdict']}"
            )

    payload = {
        "metadata": {
            "contract_version": CONTRACT_VERSION,
            "dataset": os.path.relpath(args.dataset, PROJECT_ROOT),
            "datasets": [os.path.relpath(path, PROJECT_ROOT) for path in dataset_paths],
            "offset": args.offset,
            "limit": limit,
            "model_a": args.model_a,
            "model_b": args.model_b,
            "sampling_params": {**SAMPLING_PARAMS, "max_tokens": args.max_tokens},
            "search_provider": os.environ.get("SEARCH_PROVIDER", "duckduckgo"),
            "max_search_queries": args.max_search_queries,
            "search_results": args.search_results,
            "wall_time_s": round(time.time() - t0, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "summary": summarize(records),
        "results": records,
    }
    write_output(args.output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify hidden ground truths.")
    parser.add_argument("--dataset", default=_DEFAULT_DATASET)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--output", default=_DEFAULT_OUTPUT)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model-a", default=_DEFAULT_MODEL)
    parser.add_argument("--model-b", default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-search-queries", type=int, default=5)
    parser.add_argument("--search-results", type=int, default=5)
    parser.add_argument("--search-timeout", type=float, default=10.0)
    args = parser.parse_args()

    payload = asyncio.run(run(args))
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
