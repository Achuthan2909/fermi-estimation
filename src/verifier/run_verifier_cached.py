"""Verifier that uses a static fermi_facts.json file instead of live Tavily calls.

Stage 1 is preserved: the model still generates search queries.
Those queries are then matched against datasets/fermi_facts.json by keyword
overlap — the best-matching pre-fetched answer is used as search evidence.
No Tavily or web API calls are made at inference time.

fermi_facts.json: datasets/fermi_facts.json  (override with --fermi-facts-file)

Usage:
    python src/verifier/run_verifier_cached.py \
        --dataset datasets/final/train.json \
        --offset 0 --limit 1409 \
        --model-a Qwen/Qwen3-30B-A3B \
        --concurrency 8 \
        --output run/verification_train_p1.json
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
sys.path.insert(0, HERE)
sys.path.insert(0, SRC_ROOT)

from prompt_contract import (  # noqa: E402
    CONTRACT_VERSION,
    ParseError,
    build_compute_messages,
    build_search_plan_messages,
    parse_compute_response,
    parse_search_plan_response,
)
_DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
_DEFAULT_DATASET = os.path.join(PROJECT_ROOT, "datasets", "final", "train.json")
_DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "run", "verification.json")
_DEFAULT_FERMI_FACTS = os.path.join(PROJECT_ROOT, "datasets", "fermi_facts.json")

SAMPLING_PARAMS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 512,
    "seed": 0,
}


# ---------------------------------------------------------------------------
# Fermi facts lookup
# ---------------------------------------------------------------------------

def load_fermi_facts(path: str) -> dict[str, dict[str, Any]]:
    """Load fermi_facts.json as {question_string: {domain, question, answer}}."""
    with open(path) as f:
        return json.load(f)


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at",
    "to", "for", "with", "how", "what", "many", "much", "does", "do", "it",
    "its", "per", "be", "have", "has", "about", "that", "this", "which",
    "and", "or", "by", "from", "as", "if", "than", "so", "no", "not",
}


def _query_words(text: str) -> set[str]:
    return {w for w in text.lower().split() if w not in _STOPWORDS and len(w) > 1}


def lookup_fermi_facts(
    queries: list[str],
    facts: dict[str, dict[str, Any]],
    min_overlap: int = 2,
) -> list[dict[str, Any]]:
    """Match each query against fermi_facts by keyword overlap.

    Returns one search_record per query in the same format expected by
    format_model_search_context(). Unmatched queries get status='no_match'.
    """
    entries = list(facts.values())
    fact_word_sets = [_query_words(e["question"]) for e in entries]

    records: list[dict[str, Any]] = []
    for query in queries:
        q_words = _query_words(query)
        best_idx, best_overlap = -1, 0
        for i, fw in enumerate(fact_word_sets):
            overlap = len(q_words & fw)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i
        if best_idx >= 0 and best_overlap >= min_overlap:
            entry = entries[best_idx]
            records.append({
                "query": query,
                "status": "ok",
                "error": "",
                "results": [{"title": entry["question"], "snippet": entry.get("answer", "")}],
                "source": "fermi_facts",
                "matched_question": entry["question"],
                "overlap": best_overlap,
            })
        else:
            records.append({
                "query": query,
                "status": "no_match",
                "error": f"no fermi fact with overlap >= {min_overlap}",
                "results": [],
                "source": "fermi_facts",
            })
    return records


# ---------------------------------------------------------------------------
# Helpers (unchanged from run_verifier.py)
# ---------------------------------------------------------------------------

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
        return {"verdict": "uncertain", "reason": "no parsed computation",
                "pred_value": None, "pred_unit": None, "pred_oom": None}
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
            return {"verdict": "uncertain", "reason": "numeric parse failure",
                    "pred_value": pred_value, "pred_unit": pred_unit, "pred_oom": pred_oom}
        if pred == 0 or gold == 0:
            return {"verdict": "uncertain", "reason": "zero value cannot be log-scored",
                    "pred_value": pred_value, "pred_unit": pred_unit, "pred_oom": pred_oom}
        log_error = abs(math.log10(pred) - math.log10(gold))
        return {"verdict": "correct" if log_error <= 0.5 else "incorrect",
                "reason": f"log10_error={log_error:.3f}",
                "pred_value": pred_value, "pred_unit": pred_unit, "pred_oom": pred_oom}
    gold_oom = normalize_oom(row.get("fermi_answer"))
    if pred_oom is None or gold_oom is None:
        return {"verdict": "uncertain", "reason": "missing pred_oom or gold_oom",
                "pred_value": pred_value, "pred_unit": pred_unit, "pred_oom": pred_oom}
    oom_error = pred_oom - gold_oom
    return {"verdict": "correct" if oom_error == 0 else "incorrect",
            "reason": f"oom_error={oom_error}",
            "pred_value": pred_value, "pred_unit": pred_unit, "pred_oom": pred_oom}


def normalize_verdict(verification: dict[str, Any]) -> str:
    verdict = (verification.get("comparison") or {}).get("verdict")
    return verdict if verdict in {"correct", "incorrect", "uncertain"} else "uncertain"


def consensus_verdict(verifications: list[dict[str, Any]]) -> str:
    verdicts = [normalize_verdict(v) for v in verifications]
    if not verdicts:
        return "no_verifiers"
    if all(v == verdicts[0] for v in verdicts):
        return verdicts[0]
    if "correct" in verdicts and "incorrect" in verdicts:
        return "disagreement"
    return "uncertain"


async def call_tinker_model(*, model: str, messages: list[dict[str, str]],
                            max_tokens: int) -> tuple[str, str | None, float]:
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
    response = await sc.sample_async(prompt=prompt, num_samples=1,
                                     sampling_params=sampling_params)
    text = tokenizer.decode(response.sequences[0].tokens, skip_special_tokens=True)
    return text, renderer_name, time.time() - t0


def format_model_search_context(plan: dict[str, Any],
                                search_records: list[dict[str, Any]]) -> str:
    lines = ["Reference facts to consider:"]
    for c in (plan.get("constants_needed") or []):
        if isinstance(c, dict):
            lines.append(f"- {c.get('name','')} ({c.get('unit') or ''}): "
                         f"{c.get('why') or ''}".strip())
        else:
            lines.append(f"- {c}")
    if not (plan.get("constants_needed") or []):
        lines.append("- None listed.")
    lines.append("\nReference evidence:")
    for i, rec in enumerate(search_records, 1):
        lines.append(f"\nReference group {i}:")
        if rec.get("status") != "ok" or not rec.get("results"):
            lines.append("No reference evidence was available for this item.")
            continue
        for j, item in enumerate(rec["results"], 1):
            title = item.get("title", "").strip()
            snippet = item.get("snippet", "").strip()
            if title:
                lines.append(f"- Reference {i}.{j}: {title}")
            if snippet:
                lines.append(f"  Excerpt: {snippet}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fermi-facts-backed verifier
# ---------------------------------------------------------------------------

async def run_model_verifier(
    *,
    model: str,
    row: dict[str, Any],
    max_tokens: int,
    max_search_queries: int,
    fermi_facts: dict[str, Any],
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
            model=model, messages=search_plan_messages, max_tokens=max_tokens,
        )
        plan = parse_search_plan_response(plan_text)
        plan_status, plan_error = "ok", ""
    except (Exception, ParseError) as e:
        plan_text = locals().get("plan_text", "")
        plan_latency_s = locals().get("plan_latency_s", 0.0)
        plan = {"constants_needed": [], "search_queries": []}
        plan_status, plan_error = "plan_error", repr(e)

    queries = [
        q.strip() for q in plan.get("search_queries", [])
        if isinstance(q, str) and q.strip()
    ][:max_search_queries]

    # --- fermi_facts lookup (no live web calls) ---
    search_records = lookup_fermi_facts(queries, fermi_facts)
    matches = sum(1 for r in search_records if r.get("status") == "ok")
    if queries:
        print(f"  [fermi_facts] {matches}/{len(queries)} matched  id={row.get('id','?')}")

    search_context = format_model_search_context(plan, search_records)
    compute_messages = build_compute_messages(
        question=row["question"],
        answer_type=answer_type,
        search_evidence=search_context,
    )

    try:
        compute_text, renderer_name, compute_latency_s = await call_tinker_model(
            model=model, messages=compute_messages, max_tokens=max_tokens,
        )
        computation = parse_compute_response(compute_text)
        compute_status, compute_error = "ok", ""
    except (Exception, ParseError) as e:
        compute_text = locals().get("compute_text", "")
        compute_latency_s = locals().get("compute_latency_s", 0.0)
        computation = None
        compute_status, compute_error = "compute_error", repr(e)

    comparison = verdict_from_computation(row, computation)
    if plan_status != "ok" or compute_status != "ok":
        comparison["verdict"] = "uncertain"

    return {
        "model": model,
        "renderer": renderer_name,
        "latency_s": round(time.time() - started, 3),
        "fermi_facts_matches": matches,
        "search_plan": {
            "status": plan_status, "error": plan_error,
            "latency_s": round(plan_latency_s, 3),
            "input_messages": search_plan_messages,
            "response_text": plan_text, "parsed_response": plan,
        },
        "search_records": search_records,
        "search_context": search_context,
        "computation": {
            "status": compute_status, "error": compute_error,
            "latency_s": round(compute_latency_s, 3),
            "input_messages": compute_messages,
            "response_text": compute_text, "parsed_response": computation,
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
    fermi_facts: dict[str, Any],
) -> dict[str, Any]:
    verifier_tasks = [
        run_model_verifier(
            model=model_a, row=row, max_tokens=max_tokens,
            max_search_queries=max_search_queries,
            fermi_facts=fermi_facts,
        )
    ]
    if model_b:
        verifier_tasks.append(
            run_model_verifier(
                model=model_b, row=row, max_tokens=max_tokens,
                max_search_queries=max_search_queries,
                fermi_facts=fermi_facts,
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
        "verifications": list(verifications),
        "consensus_verdict": consensus_verdict(list(verifications)),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    verdict_counts: dict[str, int] = {}
    stage_status_counts: dict[str, int] = {}
    total_fermi_matches = 0
    for record in records:
        verdict_counts[record.get("consensus_verdict", "unknown")] = \
            verdict_counts.get(record.get("consensus_verdict", "unknown"), 0) + 1
        for v in record.get("verifications", []):
            total_fermi_matches += v.get("fermi_facts_matches", 0)
            for stage in ("search_plan", "computation"):
                status = v.get(stage, {}).get("status", "unknown")
                key = f"{stage}:{status}"
                stage_status_counts[key] = stage_status_counts.get(key, 0) + 1
    return {
        "n": len(records),
        "consensus_verdict_counts": verdict_counts,
        "stage_status_counts": stage_status_counts,
        "total_fermi_matches": total_fermi_matches,
    }


def write_output(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def dataset_split_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not set. Export it or `source .env` first.")

    fermi_facts = load_fermi_facts(args.fermi_facts_file)
    print(f"[fermi_facts] loaded {len(fermi_facts)} entries from {args.fermi_facts_file}")

    dataset_paths = args.datasets or [args.dataset]
    limit = None if args.all else args.limit

    all_items: list[tuple[str, int, dict[str, Any]]] = []
    for dataset_path in dataset_paths:
        rows = load_rows(dataset_path, offset=args.offset, limit=limit)
        for i, row in enumerate(rows, start=args.offset):
            all_items.append((dataset_path, i, row))

    total = len(all_items)
    sem = asyncio.Semaphore(args.concurrency)

    async def bounded(dataset_path: str, i: int,
                      row: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            record = await verify_row(
                row, index=i,
                model_a=args.model_a, model_b=args.model_b,
                max_tokens=args.max_tokens,
                max_search_queries=args.max_search_queries,
                fermi_facts=fermi_facts,
            )
            record["dataset"] = os.path.relpath(dataset_path, PROJECT_ROOT)
            record["split"] = dataset_split_name(dataset_path)
            return record

    records: list[dict[str, Any]] = []
    done = 0
    t0 = time.time()
    checkpoint_path = args.output + ".partial"
    for coro in asyncio.as_completed(
        [bounded(dp, i, r) for dp, i, r in all_items]
    ):
        record = await coro
        records.append(record)
        done += 1
        print(f"{done}/{total} id={record['id']} consensus={record['consensus_verdict']} "
              f"domain={record.get('domain','?')}")
        if done % 25 == 0 or done == total:
            sorted_so_far = sorted(records, key=lambda r: r["index"])
            with open(checkpoint_path, "w") as f:
                json.dump(sorted_so_far, f, indent=2)
            print(f"  [checkpoint] saved {done} records to {checkpoint_path}")

    records.sort(key=lambda r: r["index"])

    payload = {
        "metadata": {
            "contract_version": CONTRACT_VERSION,
            "dataset": os.path.relpath(args.dataset, PROJECT_ROOT),
            "datasets": [os.path.relpath(p, PROJECT_ROOT) for p in dataset_paths],
            "offset": args.offset,
            "limit": limit,
            "model_a": args.model_a,
            "model_b": args.model_b,
            "concurrency": args.concurrency,
            "fermi_facts_file": os.path.relpath(args.fermi_facts_file, PROJECT_ROOT),
            "fermi_facts_count": len(fermi_facts),
            "sampling_params": {**SAMPLING_PARAMS, "max_tokens": args.max_tokens},
            "search_provider": "fermi_facts",
            "max_search_queries": args.max_search_queries,
            "wall_time_s": round(time.time() - t0, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "summary": summarize(records),
        "results": records,
    }
    write_output(args.output, payload)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Fermi ground truths using static fermi_facts.json for search evidence."
    )
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
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--fermi-facts-file", default=_DEFAULT_FERMI_FACTS,
                        help="Path to fermi_facts.json (pre-fetched assumption answers).")
    args = parser.parse_args()

    payload = asyncio.run(run(args))
    summary = payload["summary"]
    print(json.dumps(summary, indent=2))
    print(f"Fermi fact matches: {summary['total_fermi_matches']}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
