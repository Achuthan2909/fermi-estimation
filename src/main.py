import ollama

MODEL = "qwen3:4b"

PROMPT = """You are solving a hard Fermi estimation problem.

Show your reasoning step-by-step, then give your final numeric estimate.

Question:
How many visible-light photons are emitted by a standard 1-milliwatt red laser pointer in one second?
"""

print("PROMPT:\n", PROMPT)
print("RESPONSE (streaming):\n")

final = None
for chunk in ollama.generate(model=MODEL, prompt=PROMPT, stream=True):
    final = chunk
    text = chunk.get("response", "")
    if text:
        print(text, end="", flush=True)

print("\n\nSTATS:")
if final is None:
    print("No chunks received.")
else:
    print("prompt_tokens:", final.get("prompt_eval_count"))
    print("completion_tokens:", final.get("eval_count"))
    print("prompt_eval_duration_ns:", final.get("prompt_eval_duration"))
    print("eval_duration_ns:", final.get("eval_duration"))
    print("total_duration_ns:", final.get("total_duration"))

