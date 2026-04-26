"""Validate a Tinker LoRA checkpoint on Fermi questions and score against gold.

Examples:
  # Score the model on the same 25 IDs we trained on (memorisation check)
  python src/sft/validate_sft.py \
      --log-path run/sft_train_25_1024_v2 \
      --data datasets/student_traces/train_25_1024.jsonl

  # Score on the first 50 rows of the full training set (generalisation)
  python src/sft/validate_sft.py \
      --log-path run/sft_train_25_1024_v2 \
      --data datasets/final/train.json \
      --limit 50

  # Pass the sampler path explicitly instead of reading it from log-path
  python src/sft/validate_sft.py \
      --model-path tinker://...:train:0/sampler_weights/final \
      --data datasets/final/train.json --limit 25
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src" / "preprocessing"))

from tinker import SamplingParams, ServiceClient  # noqa: E402
from tinker_cookbook import model_info  # noqa: E402
from tinker_cookbook.renderers import get_renderer  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402

from prompt_contract import (  # noqa: E402
    SAMPLING_PARAMS,
    build_messages,
)
from fp_score import score_row  # noqa: E402

DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def _resolve_model_path(args: argparse.Namespace) -> str:
    if args.model_path:
        return args.model_path
    if not args.log_path:
        raise SystemExit("Pass --model-path or --log-path.")
    ckpt_file = Path(args.log_path) / "checkpoints.jsonl"
    if not ckpt_file.exists():
        raise SystemExit(f"No checkpoints.jsonl in {args.log_path}")
    last = None
    with ckpt_file.open() as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    if last is None or not last.get("sampler_path"):
        raise SystemExit(f"No sampler_path in {ckpt_file}")
    return last["sampler_path"]


def _load_rows(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    text = p.read_text()
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _gold_oom(row: dict[str, Any]) -> int | None:
    for key in ("fermi_answer", "provided_ground_truth"):
        v = row.get(key)
        if isinstance(v, int):
            return v
        if isinstance(v, float) and v.is_integer():
            return int(v)
    return None


async def _sample(
    sampling_client: Any,
    renderer: Any,
    tokenizer: Any,
    question: str,
    unit: str | None,
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    msgs = build_messages(question, unit)
    prompt = renderer.build_generation_prompt(msgs)
    sp = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=SAMPLING_PARAMS["top_p"],
        seed=SAMPLING_PARAMS["seed"],
        stop=renderer.get_stop_sequences(),
    )
    resp = await sampling_client.sample_async(
        prompt=prompt, num_samples=1, sampling_params=sp,
    )
    return tokenizer.decode(resp.sequences[0].tokens, skip_special_tokens=True)


async def _run(args: argparse.Namespace) -> None:
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not set. `source .env` first.")

    model_path = _resolve_model_path(args)
    print(f"[validate] base model: {args.base_model}", file=sys.stderr)
    print(f"[validate] adapter:    {model_path}", file=sys.stderr)

    rows = _load_rows(args.data)
    if args.ids:
        wanted = set(args.ids)
        rows = [r for r in rows if r.get("id") in wanted]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[validate] scoring {len(rows)} rows from {args.data}", file=sys.stderr)

    svc = ServiceClient()
    renderer_name = model_info.get_recommended_renderer_name(args.base_model)
    tokenizer = get_tokenizer(args.base_model)
    renderer = get_renderer(renderer_name, tokenizer)
    sc = svc.create_sampling_client(base_model=args.base_model, model_path=model_path)

    out_path = Path(args.out) if args.out else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = out_path.open("w") if out_path else None

    n = 0
    n_parsed = 0
    n_exact = 0
    n_within_1 = 0
    n_within_2 = 0
    score_sum = 0.0
    tol = float(args.tol_decades)
    t0 = time.time()

    try:
        for i, row in enumerate(rows):
            question = row.get("question")
            if not question:
                continue
            unit = row.get("unit")
            gold_oom = _gold_oom(row)
            answer_type = row.get("answer_type", "oom_exponent")

            try:
                text = await _sample(
                    sc, renderer, tokenizer, question, unit,
                    max_tokens=args.max_tokens, temperature=args.temperature,
                )
            except Exception as e:
                print(f"[{i:3d}] {row.get('id','?')}  SAMPLE FAIL: {e}", file=sys.stderr)
                continue

            result = score_row(
                text,
                answer_type=answer_type,
                gold_oom=gold_oom,
            )

            n += 1
            if result["status"] != "parse_fail":
                n_parsed += 1
            pred_oom = result.get("pred_oom")
            pred_value = result.get("pred_value")
            if isinstance(pred_oom, int) and isinstance(gold_oom, int):
                err = abs(pred_oom - gold_oom)
                if err == 0:
                    n_exact += 1
                if err <= 1:
                    n_within_1 += 1
                if err <= 2:
                    n_within_2 += 1

            # Lenient continuous Fermi score:
            #   max(0, 1 - |log10(pred) - log10(gold)| / tol)
            # For oom_exponent rows we use pred_value vs 10**gold_oom when
            # available; falls back to integer pred_oom / gold_oom.
            lenient = None
            if isinstance(gold_oom, int):
                try:
                    if isinstance(pred_value, (int, float)) and pred_value > 0 and math.isfinite(pred_value):
                        delta = abs(math.log10(pred_value) - gold_oom)
                    elif isinstance(pred_oom, int):
                        delta = abs(pred_oom - gold_oom)
                    else:
                        delta = None
                    if delta is not None:
                        lenient = max(0.0, 1.0 - delta / tol)
                except (ValueError, TypeError):
                    lenient = None
            if lenient is None:
                lenient = float(result.get("score") or 0.0)
            score_sum += lenient

            print(
                f"[{i:3d}] {row.get('id','?'):20s}  "
                f"gold={gold_oom!s:>4}  pred={pred_oom!s:>4}  "
                f"score={lenient:.2f}  status={result['status']}"
            )

            if out_f:
                out_f.write(json.dumps({
                    "id": row.get("id"),
                    "question": question,
                    "gold_oom": gold_oom,
                    "pred_oom": pred_oom,
                    "pred_value": pred_value,
                    "score": lenient,
                    "strict_score": result.get("score"),
                    "status": result.get("status"),
                    "reason": result.get("reason"),
                    "response_text": text,
                }, ensure_ascii=False) + "\n")
    finally:
        if out_f:
            out_f.close()

    dt = time.time() - t0
    print()
    print(f"--- summary ({n} scored in {dt:.1f}s) ---")
    if n:
        print(f"parse_ok:    {n_parsed}/{n} ({100*n_parsed/n:.0f}%)")
        print(f"exact OOM:   {n_exact}/{n} ({100*n_exact/n:.0f}%)")
        print(f"|err|<=1:    {n_within_1}/{n} ({100*n_within_1/n:.0f}%)")
        print(f"|err|<=2:    {n_within_2}/{n} ({100*n_within_2/n:.0f}%)")
        print(f"mean score:  {score_sum/n:.3f}  "
              f"(1 - |log10(pred/gold)|/{tol:g}, clamped; 1.0 exact, 0.0 at +/-{tol:g} decades)")
    if out_path:
        print(f"per-row predictions → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-path", help="Training log dir; reads sampler_path from checkpoints.jsonl.")
    ap.add_argument("--model-path", help="Explicit tinker:// sampler path. Overrides --log-path.")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--data", required=True,
                    help="Path to a JSON list (e.g. datasets/final/train.json) or JSONL "
                         "(e.g. a student_traces file). Each row needs `question` and an OOM gold field.")
    ap.add_argument("--limit", type=int, default=0, help="Score only the first N rows (0 = all).")
    ap.add_argument("--ids", nargs="+", help="Restrict to these IDs.")
    ap.add_argument("--out", help="Optional JSONL path to dump per-row predictions.")
    ap.add_argument("--max-tokens", type=int, default=int(SAMPLING_PARAMS.get("max_tokens", 1024)))
    ap.add_argument("--temperature", type=float, default=float(SAMPLING_PARAMS.get("temperature", 0.0)))
    ap.add_argument("--tol-decades", type=float, default=3.0,
                    help="Tolerance for the lenient log10 score: "
                         "score = max(0, 1 - |log10(pred/gold)| / tol). "
                         "Default 3.0 (1.0 exact, 0.67 at 1 decade, 0.33 at 2, 0 at 3+).")
    args = ap.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
