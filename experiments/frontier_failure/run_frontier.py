"""
Run 5 Fermi problems against frontier LLMs and save raw outputs.

Frontier models as of April 2026:
  - OpenAI GPT-5.4 (released March 2026)
  - Google Gemini 2.5 Flash (free tier; 3.1 Pro requires paid plan)
  - Anthropic Claude Opus 4.6
  - Meta LLaMA 4 Maverick via Together AI (Turbo serverless variant)

Usage:
    export OPENAI_API_KEY=sk-...
    export GEMINI_API_KEY=...
    export TOGETHER_API_KEY=...
    export ANTHROPIC_API_KEY=...
    python run_frontier.py

Cost estimate: <$2 total for all models.
"""

import json
import os
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROBLEMS_PATH = Path(__file__).parent / "fermi_problems.json"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = (
    "You are a physicist and engineer. Solve this Fermi estimation problem step by step. "
    "Show your reasoning chain — identify the key quantities, estimate each one, "
    "then combine them. Give your final numerical answer on the last line in the format: "
    "FINAL ANSWER: <number> <unit>"
)


def load_problems():
    with open(PROBLEMS_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def run_openai(question: str) -> str:
    """GPT-5.4 via OpenAI API."""
    from openai import OpenAI

    client = OpenAI()  # uses OPENAI_API_KEY env var
    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        max_completion_tokens=2048,
        temperature=0.3,
    )
    return response.choices[0].message.content


def run_gemini(question: str) -> str:
    """Gemini 2.5 Flash via Google Gen AI SDK (free tier)."""
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"{SYSTEM_PROMPT}\n\nQuestion: {question}",
    )
    return response.text


def run_llama(question: str) -> str:
    """LLaMA 4 Maverick via Together AI (Turbo serverless)."""
    from together import Together

    client = Together()  # uses TOGETHER_API_KEY env var
    response = client.chat.completions.create(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-Turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        max_tokens=2048,
        temperature=0.3,
    )
    return response.choices[0].message.content


def run_claude(question: str) -> str:
    """Claude Opus 4.6 via Anthropic API."""
    import anthropic

    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Map of model name -> runner function + required env var
# ---------------------------------------------------------------------------
MODELS = {
    "gpt-5.4": {"fn": run_openai, "env": "OPENAI_API_KEY"},
    "gemini-2.5-flash": {"fn": run_gemini, "env": "GEMINI_API_KEY"},
    "llama-4-maverick": {"fn": run_llama, "env": "TOGETHER_API_KEY"},
    "claude-opus-4.6": {"fn": run_claude, "env": "ANTHROPIC_API_KEY"},
}


def main():
    problems = load_problems()
    all_results = {}

    for model_name, cfg in MODELS.items():
        env_var = cfg["env"]
        if not os.environ.get(env_var):
            print(f"  SKIP {model_name} — {env_var} not set")
            continue

        print(f"\n{'='*60}")
        print(f"  Running: {model_name}")
        print(f"{'='*60}")

        model_results = []
        for p in problems:
            print(f"  Q{p['id']}: {p['question'][:70]}...")
            try:
                answer = cfg["fn"](p["question"])
                print(f"    -> got {len(answer)} chars")
            except Exception as e:
                answer = f"ERROR: {e}"
                print(f"    -> ERROR: {e}")

            model_results.append({
                "id": p["id"],
                "question": p["question"],
                "ground_truth_OoM": p["ground_truth_OoM"],
                "ground_truth_unit": p["ground_truth_unit"],
                "model_response": answer,
            })
            time.sleep(1)  # rate limit courtesy

        all_results[model_name] = model_results

        # Save per-model file
        out_path = OUTPUT_DIR / f"{model_name.replace('.', '_')}.json"
        with open(out_path, "w") as f:
            json.dump(model_results, f, indent=2)
        print(f"  Saved -> {out_path}")

    # Save combined file
    combined_path = OUTPUT_DIR / "all_results.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved -> {combined_path}")


if __name__ == "__main__":
    main()
