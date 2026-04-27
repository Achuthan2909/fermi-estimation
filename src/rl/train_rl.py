"""RL fine-tuning of Qwen3-4B-Instruct-2507 on Fermi estimation (GRPO).

Run prepare_rl_prompts.py first to build datasets/rl/train.jsonl.
The SFT checkpoint is used as the initial policy and KL reference.

Usage:
    python src/rl/train_rl.py --init-checkpoint tinker://... --dry-run
    python src/rl/train_rl.py --init-checkpoint tinker://... --data-file datasets/rl/train.jsonl
    python src/rl/train_rl.py --init-checkpoint tinker://... --num-steps 500 --wandb-name rl_v1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src" / "preprocessing"))
sys.path.insert(0, str(ROOT / "src" / "rl"))

import chz
import tinker
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook import renderers as tc_renderers
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.rl.train import Config, KLReferenceConfig
from tinker_cookbook.rl import train as rl_train

from prompt_contract import SYSTEM_PROMPT, USER_TEMPLATE  # noqa: E402
from reward import reward_fn  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
RENDERER_NAME = model_info.get_recommended_renderer_name(MODEL_NAME)

_DEFAULT_DATA_FILE = ROOT / "datasets" / "rl" / "train.jsonl"
_DEFAULT_LOG_PATH = ROOT / "run" / "rl_v1"


# ── Environment ──────────────────────────────────────────────────────────────

class FermiEnv(Env):
    """Single-turn Fermi estimation environment."""

    def __init__(self, question: str, unit: str, gold_oom: int, renderer: Any) -> None:
        self._question = question
        self._unit = unit
        self._gold_oom = gold_oom
        self._renderer = renderer

    async def initial_observation(self):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                question=self._question, unit=self._unit,
            )},
        ]
        return (
            self._renderer.build_generation_prompt(msgs),
            self._renderer.get_stop_sequences(),
        )

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        message, _ = self._renderer.parse_response(action)
        text = tc_renderers.get_text_content(message)
        reward = reward_fn(text, {"provided_ground_truth": self._gold_oom})
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self._renderer.get_stop_sequences(),
            metrics={"reward": reward},
            logs={
                "question": self._question[:80],
                "gold_oom": str(self._gold_oom),
                "response_snippet": text[:120],
            },
        )


class FermiGroupBuilder(EnvGroupBuilder):
    """Creates `group_size` FermiEnv instances for one question (GRPO rollouts)."""

    def __init__(self, question: str, unit: str, gold_oom: int, renderer: Any, group_size: int) -> None:
        self._question = question
        self._unit = unit
        self._gold_oom = gold_oom
        self._renderer = renderer
        self._group_size = group_size

    async def make_envs(self) -> Sequence[Env]:
        return [
            FermiEnv(self._question, self._unit, self._gold_oom, self._renderer)
            for _ in range(self._group_size)
        ]

    def logging_tags(self) -> list[str]:
        return ["fermi"]


# ── Dataset ───────────────────────────────────────────────────────────────────

class FermiRLDataset(RLDataset):
    def __init__(
        self,
        rows: list[dict],
        batch_size: int,
        group_size: int,
        renderer: Any,
        n_steps: int,
    ) -> None:
        self._rows = rows
        self._batch_size = batch_size
        self._group_size = group_size
        self._renderer = renderer
        self._n_steps = n_steps

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        builders = []
        for i in range(self._batch_size):
            row = self._rows[(index * self._batch_size + i) % len(self._rows)]
            meta = row.get("metadata", row)
            question = meta.get("question") or row.get("question", "")
            unit = meta.get("unit") or row.get("unit") or "any sensible SI unit — state it in final.unit"
            gold_oom = meta.get("ground_truth_oom")
            if gold_oom is None:
                gold_oom = int(row.get("provided_ground_truth", 0))
            builders.append(
                FermiGroupBuilder(question, unit, int(gold_oom), self._renderer, self._group_size)
            )
        return builders

    def __len__(self) -> int:
        return self._n_steps


@chz.chz
class FermiRLDatasetBuilder(RLDatasetBuilder):
    data_file: str
    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    n_steps: int

    async def __call__(self):
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = get_renderer(self.renderer_name, tokenizer)
        with open(self.data_file) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        dataset = FermiRLDataset(rows, self.batch_size, self.group_size, renderer, self.n_steps)
        return dataset, None


# ── Config builder ────────────────────────────────────────────────────────────

def build_config(
    data_file: str,
    log_path: str,
    init_checkpoint: str,
    num_steps: int,
    batch_size: int,
    group_size: int,
    rollout_max_tokens: int,
    learning_rate: float,
    kl_coef: float,
    lora_rank: int,
    wandb_project: str | None,
    wandb_name: str | None,
) -> Config:
    dataset_builder = FermiRLDatasetBuilder(
        data_file=data_file,
        batch_size=batch_size,
        group_size=group_size,
        model_name_for_tokenizer=MODEL_NAME,
        renderer_name=RENDERER_NAME,
        n_steps=num_steps,
    )

    overrides: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "renderer_name": RENDERER_NAME,
        "log_path": log_path,
        "dataset_builder": dataset_builder,
        "learning_rate": learning_rate,
        "load_checkpoint_path": init_checkpoint,
        "max_tokens": rollout_max_tokens,
        "max_steps": num_steps,
        "kl_penalty_coef": kl_coef,
        "lora_rank": lora_rank,
        "save_every": 100,
    }
    if kl_coef > 0:
        # KLReferenceConfig requires sampler_weights path; training init requires weights path.
        kl_sampler_path = init_checkpoint.replace("/weights/", "/sampler_weights/")
        overrides["kl_reference_config"] = KLReferenceConfig(
            base_model=MODEL_NAME,
            load_checkpoint_path=kl_sampler_path,
        )
    if wandb_project:
        overrides["wandb_project"] = wandb_project
    if wandb_name:
        overrides["wandb_name"] = wandb_name

    return chz.Blueprint(Config).apply(overrides).make()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not os.environ.get("TINKER_API_KEY"):
        sys.exit("TINKER_API_KEY not set. Run: source .env")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init-checkpoint", required=True,
                    help="Tinker sampler_path of the SFT checkpoint to start from. "
                         "Found in run/sft_filtered_v1_v2/checkpoints.jsonl.")
    ap.add_argument("--data-file", default=str(_DEFAULT_DATA_FILE))
    ap.add_argument("--log-path", default=str(_DEFAULT_LOG_PATH))
    ap.add_argument("--num-steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Prompts per training step.")
    ap.add_argument("--group-size", type=int, default=4,
                    help="Rollouts per prompt (GRPO group size).")
    ap.add_argument("--rollout-max-tokens", type=int, default=800)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--kl-coef", type=float, default=0.04,
                    help="KL penalty coefficient vs SFT reference. 0 disables KL.")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--wandb-project", default="fermi-rl",
                    help="Wandb project. Pass empty string to disable.")
    ap.add_argument("--wandb-name", default=None,
                    help="Wandb run name. Defaults to basename of --log-path.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print resolved config and exit without training.")
    args = ap.parse_args()

    data_path = Path(args.data_file)
    if not data_path.exists():
        sys.exit(
            f"Data file not found: {data_path}\n"
            "Run `python src/rl/prepare_rl_prompts.py` first."
        )

    with data_path.open() as f:
        n_rows = sum(1 for line in f if line.strip())

    wandb_project = args.wandb_project or None
    wandb_name = args.wandb_name or (Path(args.log_path).name if wandb_project else None)
    wandb_entity = os.environ.get("WANDB_ENTITY")

    total_rollout_tokens = args.num_steps * args.batch_size * args.group_size * args.rollout_max_tokens
    est_cost = total_rollout_tokens / 1e6 * 0.22 * 2.5  # RL ~2.5x vs SFT due to sampling overhead

    print(f"Model:            {MODEL_NAME}")
    print(f"Renderer:         {RENDERER_NAME}")
    print(f"Init checkpoint:  {args.init_checkpoint}")
    print(f"Data file:        {args.data_file}  ({n_rows} prompts)")
    print(f"Log path:         {args.log_path}")
    print(f"Steps:            {args.num_steps}")
    print(f"Batch size:       {args.batch_size} prompts/step")
    print(f"Group size:       {args.group_size} rollouts/prompt")
    print(f"Rollout tokens:   {args.rollout_max_tokens} max")
    print(f"LR:               {args.learning_rate:.2e}")
    print(f"KL coef:          {args.kl_coef}")
    print(f"LoRA rank:        {args.lora_rank}")
    print(f"Wandb:            {wandb_project or 'disabled'} / {wandb_name or 'N/A'}"
          + (f" (entity: {wandb_entity})" if wandb_entity else ""))
    print(f"\nTotal rollout tokens: {total_rollout_tokens:,} (~{total_rollout_tokens/1e6:.1f}M)")
    print(f"Est. cost:        ${est_cost:.2f}  "
          f"(@ $0.22/1M × 2.5x RL overhead — actual may vary)")

    if args.dry_run:
        print("\n--dry-run set, exiting without training.")
        return

    config = build_config(
        data_file=args.data_file,
        log_path=args.log_path,
        init_checkpoint=args.init_checkpoint,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        group_size=args.group_size,
        rollout_max_tokens=args.rollout_max_tokens,
        learning_rate=args.learning_rate,
        kl_coef=args.kl_coef,
        lora_rank=args.lora_rank,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )

    cli_utils.check_log_dir(args.log_path, behavior_if_exists="ask")
    asyncio.run(rl_train.main(config=config))


if __name__ == "__main__":
    main()
