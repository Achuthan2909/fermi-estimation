import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ollama


@dataclass(frozen=True)
class Example:
    question: str
    answer: str


def load_realfp(path: Path) -> List[Example]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    examples: List[Example] = []
    for item in raw:
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if q and a:
            examples.append(Example(question=q, answer=a))
    return examples


def load_eval_jsonl(path: Path) -> List[Example]:
    """Load eval_v1-style JSONL (fields: question, gold_numeric|gold_oom, unit)."""
    examples: List[Example] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            q = str(item.get("question", "")).strip()
            if not q:
                continue
            if item.get("gold_numeric") is not None:
                unit = item.get("unit") or ""
                a = f"{item['gold_numeric']:.6E}{' '+unit if unit else ''}".strip()
            else:
                a = str(item.get("gold_oom", "")).strip()
            examples.append(Example(question=q, answer=a))
    return examples


def load_examples(path: Path) -> List[Example]:
    """Auto-detect: jsonl => eval_v1 format; json => realFP format."""
    if path.suffix == ".jsonl":
        return load_eval_jsonl(path)
    return load_realfp(path)


def split_question_half(question: str) -> Tuple[str, str]:
    words = question.split()
    if len(words) < 6:
        mid = max(1, len(words) // 2)
    else:
        mid = len(words) // 2
    prefix = " ".join(words[:mid]).strip()
    suffix = " ".join(words[mid:]).strip()
    return prefix, suffix


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def model_complete(model: str, prompt: str, *, temperature: float = 0.0) -> Tuple[str, Dict[str, Any]]:
    resp = ollama.generate(model=model, prompt=prompt, options={"temperature": temperature})
    text = (resp.get("response") or "").strip()
    stats = {
        "prompt_tokens": resp.get("prompt_eval_count"),
        "completion_tokens": resp.get("eval_count"),
        "prompt_eval_duration_ns": resp.get("prompt_eval_duration"),
        "eval_duration_ns": resp.get("eval_duration"),
        "total_duration_ns": resp.get("total_duration"),
    }
    return text, stats


def run_canary_checks(
    model: str,
    examples: List[Example],
    *,
    n: int,
    seed: int,
    temperature: float,
) -> None:
    rng = random.Random(seed)
    sampled = rng.sample(examples, k=min(n, len(examples)))

    hits = 0
    for i, ex in enumerate(sampled, start=1):
        prefix, suffix = split_question_half(ex.question)
        prompt = (
            "Complete this sentence (verbatim if you can). "
            "Only output the completion, no extra commentary.\n\n"
            f"{prefix}"
        )
        out, _ = model_complete(model, prompt, temperature=temperature)

        suffix_norm = normalize_ws(suffix)
        out_norm = normalize_ws(out)
        is_verbatim = suffix_norm and suffix_norm in out_norm
        hits += int(is_verbatim)

        print(f"\n[{i}/{len(sampled)}] PREFIX:\n{prefix}")
        print(f"EXPECTED_SUFFIX:\n{suffix}")
        print(f"MODEL_COMPLETION:\n{out}")
        print(f"VERBATIM_SUFFIX_MATCH: {is_verbatim}")

    print(f"\nCANARY_VERBATIM_HITS: {hits}/{len(sampled)}")


def run_verbatim_answer_checks(
    model: str,
    examples: List[Example],
    *,
    n: int,
    seed: int,
    temperature: float,
) -> None:
    rng = random.Random(seed)
    sampled = rng.sample(examples, k=min(n, len(examples)))

    exact = 0
    for i, ex in enumerate(sampled, start=1):
        prompt = (
            "Answer the following question.\n\n"
            "Return ONLY the final answer string (no words, no units unless included in the answer string). "
            "Do not explain.\n\n"
            f"Question: {ex.question}\n"
        )
        out, _ = model_complete(model, prompt, temperature=temperature)
        out_stripped = out.strip().strip('"').strip("'")
        is_exact = out_stripped == ex.answer
        exact += int(is_exact)

        print(f"\n[{i}/{len(sampled)}] Q:\n{ex.question}")
        print(f"GOLD_ANSWER:\n{ex.answer}")
        print(f"MODEL_ANSWER:\n{out}")
        print(f"EXACT_MATCH: {is_exact}")

    print(f"\nANSWER_EXACT_MATCHES: {exact}/{len(sampled)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick contamination / memorization checks with Ollama.")
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument(
        "--input", "--realfp",
        dest="input",
        default=str(Path("datasets/eval/eval_v1.jsonl")),
        help="Path to eval file. .jsonl => eval_v1 format; .json => RealFP format.",
    )
    parser.add_argument("--n", type=int, default=10, help="How many examples to sample.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--mode",
        choices=["canary", "answer", "both"],
        default="both",
        help="Which checks to run.",
    )
    args = parser.parse_args()

    examples = load_examples(Path(args.input))
    if not examples:
        raise SystemExit(f"No examples loaded from {args.input}")

    if args.mode in ("canary", "both"):
        run_canary_checks(args.model, examples, n=args.n, seed=args.seed, temperature=args.temperature)
    if args.mode in ("answer", "both"):
        run_verbatim_answer_checks(args.model, examples, n=args.n, seed=args.seed, temperature=args.temperature)


if __name__ == "__main__":
    main()
