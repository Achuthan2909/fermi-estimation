"""Single-question Tinker sampling using the frozen prompt contract + tinker_cookbook.

Run from project root:
  TINKER_API_KEY=... python src/sampling/tinker_infrence.py
  TINKER_API_KEY=... python src/sampling/tinker_infrence.py --index 0
  TINKER_API_KEY=... python src/sampling/tinker_infrence.py --question "What is 2^100?"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from tinker import SamplingParams, ServiceClient  # noqa: E402
from tinker_cookbook import model_info  # noqa: E402
from tinker_cookbook.renderers import get_renderer  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402

from prompt_contract import (  # noqa: E402
    CONTRACT_VERSION,
    ParseError,
    SAMPLING_PARAMS,
    build_messages,
    parse_response,
)

_DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
_DEFAULT_TRAIN = os.path.join(ROOT, "datasets", "final", "train.json")


def load_row(train_path: str, index: int) -> dict:
    with open(train_path) as f:
        rows = json.load(f)
    if not rows:
        raise SystemExit("Train file is empty.")
    if index < 0 or index >= len(rows):
        raise SystemExit(f"--index {index} out of range (0..{len(rows) - 1})")
    return rows[index]


async def run_one(model: str, question: str, unit: str | None) -> str:
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not set. Export it or `source` your .env in the shell.")

    svc = ServiceClient()
    renderer_name = model_info.get_recommended_renderer_name(model)
    tokenizer = get_tokenizer(model)
    renderer = get_renderer(renderer_name, tokenizer)
    sc = svc.create_sampling_client(base_model=model)

    msgs = build_messages(question, unit)
    prompt = renderer.build_generation_prompt(msgs)
    stop_tokens = renderer.get_stop_sequences()
    sampling_params = SamplingParams(
        max_tokens=SAMPLING_PARAMS["max_tokens"],
        temperature=SAMPLING_PARAMS["temperature"],
        top_p=SAMPLING_PARAMS["top_p"],
        seed=SAMPLING_PARAMS["seed"],
        stop=stop_tokens,
    )

    t0 = time.time()
    resp = await sc.sample_async(
        prompt=prompt, num_samples=1, sampling_params=sampling_params,
    )
    text = tokenizer.decode(resp.sequences[0].tokens, skip_special_tokens=True)
    dt = time.time() - t0
    print(f"(latency {dt:.2f}s, renderer {renderer_name})", file=sys.stderr)
    return text


def format_ground_truth(row: dict) -> str:
    if "fermi_answer" in row:
        answer_type = row.get("answer_type") or "oom_exponent"
        return f"{row['fermi_answer']} ({answer_type})"
    if "answer" in row:
        return str(row["answer"])
    return "not provided"


def format_assumption(assumption: dict) -> str:
    name = assumption.get("name") or "assumption"
    value = assumption.get("value")
    unit = assumption.get("unit") or ""
    low = assumption.get("low")
    high = assumption.get("high")
    reasoning = assumption.get("reasoning") or ""
    value_text = f"{value} {unit}".strip()
    bounds_text = f"[{low}, {high}]"
    return f"- {name}: {value_text} {bounds_text} - {reasoning}".rstrip()


def format_trace(question: str, row: dict, response_text: str) -> str:
    lines = [
        f"Question: {question}",
        f"Ground Truth Answer: {format_ground_truth(row)}",
        "",
    ]
    try:
        obj = parse_response(response_text)
    except ParseError as e:
        lines.extend(
            [
                f"Parse Error: {e}",
                "",
                "Raw Response:",
                response_text.strip(),
            ]
        )
        return "\n".join(lines)

    lines.append("Assumptions:")
    for assumption in obj.get("assumptions", []):
        if isinstance(assumption, dict):
            lines.append(format_assumption(assumption))
        else:
            lines.append(f"- {assumption}")

    final = obj["final_answer"]
    final_value = final.get("value")
    final_unit = final.get("unit") or ""
    lines.extend(
        [
            "",
            "Calculation:",
            str(obj.get("calculation", "")).strip(),
            "",
            "Final Answer:",
            f"{final_value} {final_unit}".strip() + f" (oom={final.get('oom')})",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Single Tinker sample for one training-style question.")
    p.add_argument("--model", default=_DEFAULT_MODEL, help="Tinker base model id")
    p.add_argument(
        "--train-json",
        default=_DEFAULT_TRAIN,
        help="Path to train.json (used with --index unless --question is set)",
    )
    p.add_argument("--index", type=int, default=0, help="Row index in train.json")
    p.add_argument(
        "--question", default=None, help="If set, use this string instead of train.json",
    )
    p.add_argument(
        "--raw-json",
        action="store_true",
        help="Print the raw model JSON instead of the trace view",
    )
    args = p.parse_args()

    if args.question is not None:
        row = {"question": args.question, "unit": None}
    else:
        row = load_row(args.train_json, args.index)
        if "question" not in row:
            raise SystemExit("Row has no 'question' field.")
    unit = row.get("unit")
    q = row["question"]

    print("Model:", args.model, file=sys.stderr)
    print(
        "Contract:",
        CONTRACT_VERSION,
        "| max_tokens (backstop):",
        SAMPLING_PARAMS["max_tokens"],
        file=sys.stderr,
    )
    if args.question is None and row.get("id") is not None:
        print("id:", row["id"], file=sys.stderr)

    text = asyncio.run(run_one(args.model, q, unit))
    if args.raw_json:
        print(text)
    else:
        print(format_trace(q, row, text))


if __name__ == "__main__":
    main()