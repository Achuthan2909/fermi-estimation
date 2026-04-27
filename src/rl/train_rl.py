"""GRPO RL fine-tune of the SFT'd Qwen3-4B-Instruct-2507 on Fermi prompts.

Pipeline:
  1. Read RL prompts JSONL produced by ``src/rl/prepare_rl_prompts.py``.
  2. For each batch step, build a list of ``ProblemGroupBuilder`` groups,
     one per prompt; each group runs ``rollouts_per_prompt`` envs, which
     Tinker's ``rl.train`` then uses to compute group-centered advantages
     (the GRPO update — Tinker uses ``loss_fn='importance_sampling'`` by
     default, which is the GRPO recipe when combined with grouped envs).
  3. Each ``FermiEnv.step`` decodes the model's response and scores it
     with ``src.rl.reward.reward_fn``.
  4. KL is penalised against the SFT checkpoint (kl_reference_config).

Usage:
    python src/rl/train_rl.py --init-checkpoint tinker://...sampler_weights/final
    python src/rl/train_rl.py --init-checkpoint tinker://... --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src" / "preprocessing"))
sys.path.insert(0, str(ROOT / "src"))

import chz
import tinker
from tinker_cookbook import cli_utils, model_info, renderers
from tinker_cookbook.rl import train as rl_train
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    StopCondition,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from rl.reward import reward_fn  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
RENDERER_NAME = model_info.get_recommended_renderer_name(MODEL_NAME)

_DEFAULT_PROMPTS = ROOT / "datasets" / "rl" / "train.jsonl"
_DEFAULT_LOG_PATH = ROOT / "run" / "rl_v1"


# -----------------------------------------------------------------------------
# Env / EnvGroupBuilder / Dataset
# -----------------------------------------------------------------------------


class FermiEnv(Env):
    """One Fermi prompt → one rollout → one continuous reward.

    Picklable: holds plain-Python fields plus a Renderer (which Tinker says
    is pickle-safe).
    """

    def __init__(
        self,
        renderer: renderers.Renderer,
        messages: list[dict[str, str]],
        gold_oom: int,
    ):
        self.renderer = renderer
        self.messages = messages
        self.gold_oom = int(gold_oom)

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        observation = self.renderer.build_generation_prompt(self.messages)
        return observation, self.stop_condition

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        message, _ = self.renderer.parse_response(action)
        text = renderers.get_text_content(message)
        reward = reward_fn(text, {"oom": self.gold_oom})
        return StepResult(
            reward=float(reward),
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={"reward": float(reward)},
        )


@dataclass(frozen=True)
class _FermiPrompt:
    id: str
    messages: tuple[tuple[str, str], ...]   # frozen for hashability
    gold_oom: int
    domain: str

    @classmethod
    def from_jsonl_row(cls, row: dict[str, Any]) -> "_FermiPrompt":
        msgs = tuple((m["role"], m["content"]) for m in row["messages"])
        meta = row["metadata"]
        return cls(
            id=str(meta["id"]),
            messages=msgs,
            gold_oom=int(meta["ground_truth_oom"]),
            domain=str(meta.get("domain", "unknown")),
        )

    def to_messages(self) -> list[dict[str, str]]:
        return [{"role": r, "content": c} for r, c in self.messages]


def _make_env(
    prompt: _FermiPrompt,
    renderer_name: str,
    model_name_for_tokenizer: str,
) -> FermiEnv:
    # Construct renderer lazily inside the worker (pickle-safe pattern).
    tokenizer = get_tokenizer(model_name_for_tokenizer)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    return FermiEnv(renderer=renderer, messages=prompt.to_messages(), gold_oom=prompt.gold_oom)


class FermiDataset(RLDataset):
    """Cycles through the RL prompt set so we can hit a target ``num_steps``.

    Each batch yields ``batch_size_prompts`` ProblemGroupBuilder groups, each
    producing ``rollouts_per_prompt`` envs from the same prompt.
    """

    def __init__(
        self,
        prompts: list[_FermiPrompt],
        *,
        batch_size_prompts: int,
        rollouts_per_prompt: int,
        num_steps: int,
        renderer_name: str,
        model_name_for_tokenizer: str,
        seed: int = 0,
    ):
        if not prompts:
            raise ValueError("FermiDataset received zero prompts")
        self._prompts = prompts
        self._batch_size_prompts = batch_size_prompts
        self._rollouts_per_prompt = rollouts_per_prompt
        self._num_steps = num_steps
        self._renderer_name = renderer_name
        self._model_name_for_tokenizer = model_name_for_tokenizer
        # Pre-compute a cyclical schedule so each step sees a deterministic
        # slice of prompts even when batch * num_steps > len(prompts).
        import random as _rnd
        rng = _rnd.Random(seed)
        order = list(range(len(prompts)))
        rng.shuffle(order)
        total = batch_size_prompts * num_steps
        self._schedule = [order[i % len(order)] for i in range(total)]

    def __len__(self) -> int:
        return self._num_steps

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self._batch_size_prompts
        end = start + self._batch_size_prompts
        builders: list[EnvGroupBuilder] = []
        for i in self._schedule[start:end]:
            prompt = self._prompts[i]
            thunk = partial(
                _make_env,
                prompt,
                self._renderer_name,
                self._model_name_for_tokenizer,
            )
            builders.append(
                ProblemGroupBuilder(
                    env_thunk=thunk,
                    num_envs=self._rollouts_per_prompt,
                    dataset_name=f"fermi:{prompt.domain}",
                )
            )
        return builders


@chz.chz
class FermiDatasetBuilder(RLDatasetBuilder):
    prompts_file: str
    batch_size_prompts: int
    rollouts_per_prompt: int
    num_steps: int
    renderer_name: str
    model_name_for_tokenizer: str
    seed: int = 0

    async def __call__(self) -> tuple[FermiDataset, None]:
        path = Path(self.prompts_file)
        if not path.exists():
            raise FileNotFoundError(f"RL prompts file not found: {path}")
        prompts: list[_FermiPrompt] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            prompts.append(_FermiPrompt.from_jsonl_row(json.loads(line)))
        ds = FermiDataset(
            prompts=prompts,
            batch_size_prompts=self.batch_size_prompts,
            rollouts_per_prompt=self.rollouts_per_prompt,
            num_steps=self.num_steps,
            renderer_name=self.renderer_name,
            model_name_for_tokenizer=self.model_name_for_tokenizer,
            seed=self.seed,
        )
        return ds, None


# -----------------------------------------------------------------------------
# Config blueprint + CLI
# -----------------------------------------------------------------------------


def build_config(
    *,
    init_checkpoint: str,
    prompts_file: str,
    log_path: str,
    learning_rate: float,
    kl_coef: float,
    rollouts_per_prompt: int,
    rollout_max_tokens: int,
    num_steps: int,
    batch_size_prompts: int,
    save_every: int,
    eval_every: int,
    lora_rank: int,
    seed: int,
    wandb_project: str | None,
    wandb_name: str | None,
) -> rl_train.Config:
    dataset = FermiDatasetBuilder(
        prompts_file=prompts_file,
        batch_size_prompts=batch_size_prompts,
        rollouts_per_prompt=rollouts_per_prompt,
        num_steps=num_steps,
        renderer_name=RENDERER_NAME,
        model_name_for_tokenizer=MODEL_NAME,
        seed=seed,
    )

    overrides: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "renderer_name": RENDERER_NAME,
        "log_path": log_path,
        "dataset_builder": dataset,
        "learning_rate": learning_rate,
        "max_tokens": rollout_max_tokens,
        "save_every": save_every,
        "eval_every": eval_every,
        "lora_rank": lora_rank,
        "load_checkpoint_path": init_checkpoint,
        "kl_penalty_coef": kl_coef,
    }
    if kl_coef > 0:
        overrides["kl_reference_config"] = rl_train.KLReferenceConfig(
            base_model=MODEL_NAME,
            load_checkpoint_path=init_checkpoint,
        )
    if wandb_project:
        overrides["wandb_project"] = wandb_project
    if wandb_name:
        overrides["wandb_name"] = wandb_name

    return chz.Blueprint(rl_train.Config).apply(overrides).make()


def _print_config_summary(args: argparse.Namespace, n_prompts: int) -> None:
    print(f"Model:               {MODEL_NAME}")
    print(f"Renderer:            {RENDERER_NAME}")
    print(f"Init checkpoint:     {args.init_checkpoint}")
    print(f"Prompts file:        {args.prompts_file}  ({n_prompts} prompts)")
    print(f"Log path:            {args.log_path}")
    print()
    print(f"num_steps:           {args.num_steps}")
    print(f"batch_size_prompts:  {args.batch_size_prompts}")
    print(f"rollouts_per_prompt: {args.rollouts_per_prompt}")
    print(f"rollout_max_tokens:  {args.rollout_max_tokens}")
    print(f"learning_rate:       {args.learning_rate:.2e}")
    print(f"kl_coef:             {args.kl_coef}")
    print(f"lora_rank:           {args.lora_rank}")
    print(f"save_every:          {args.save_every}")
    print(f"eval_every:          {args.eval_every}")
    print(f"seed:                {args.seed}")
    print()
    print(f"wandb_project:       {args.wandb_project or '(disabled)'}")
    print(f"wandb_name:          {args.wandb_name or '(default to log dir name)'}")
    print()
    rollouts = args.num_steps * args.batch_size_prompts * args.rollouts_per_prompt
    avg_prompt = 280
    avg_response = args.rollout_max_tokens
    rl_token_cost_per_m = 0.22 * 2.5  # ~2.5x SFT due to sampling + grad overhead
    total_tokens = rollouts * (avg_prompt + avg_response)
    est_cost = total_tokens / 1e6 * rl_token_cost_per_m
    print(f"Total rollouts:      {rollouts:,}")
    print(f"Est. tokens:         {total_tokens:,} (~{total_tokens/1e6:.2f}M)")
    print(f"Est. cost:           ${est_cost:.2f}  "
          f"(@ ${rl_token_cost_per_m:.2f}/1M ~2.5x SFT — includes sampling overhead)")


def main() -> None:
    if not os.environ.get("TINKER_API_KEY"):
        sys.exit("TINKER_API_KEY is not set. Run: source .env")

    ap = argparse.ArgumentParser()
    ap.add_argument("--init-checkpoint", required=True,
                    help="tinker:// sampler/checkpoint URI from the SFT run.")
    ap.add_argument("--prompts-file", default=str(_DEFAULT_PROMPTS),
                    help="JSONL produced by prepare_rl_prompts.py.")
    ap.add_argument("--log-path", default=str(_DEFAULT_LOG_PATH))
    ap.add_argument("--num-steps", type=int, default=500)
    ap.add_argument("--batch-size-prompts", type=int, default=16,
                    help="Number of prompts per training step (= number of groups).")
    ap.add_argument("--rollouts-per-prompt", type=int, default=4,
                    help="Rollouts in each group; GRPO centers advantages within a group.")
    ap.add_argument("--rollout-max-tokens", type=int, default=800)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--kl-coef", type=float, default=0.04,
                    help="KL penalty against the SFT checkpoint (kl_penalty_coef).")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=0,
                    help="Evaluator cadence (0 = disabled; we have no test split).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb-project", default="fermi-rl",
                    help="Wandb project name. Pass an empty string to disable.")
    ap.add_argument("--wandb-name", default=None,
                    help="Wandb run name. Defaults to basename of --log-path.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print resolved config and exit without training.")
    args = ap.parse_args()

    wandb_project = args.wandb_project or None
    wandb_name = args.wandb_name or (Path(args.log_path).name if wandb_project else None)
    args.wandb_project = wandb_project
    args.wandb_name = wandb_name

    prompts_path = Path(args.prompts_file)
    if not prompts_path.exists():
        sys.exit(
            f"Prompts file not found: {prompts_path}\n"
            "Run `python src/rl/prepare_rl_prompts.py` first."
        )
    n_prompts = sum(1 for _ in prompts_path.open())

    config = build_config(
        init_checkpoint=args.init_checkpoint,
        prompts_file=str(prompts_path),
        log_path=args.log_path,
        learning_rate=args.learning_rate,
        kl_coef=args.kl_coef,
        rollouts_per_prompt=args.rollouts_per_prompt,
        rollout_max_tokens=args.rollout_max_tokens,
        num_steps=args.num_steps,
        batch_size_prompts=args.batch_size_prompts,
        save_every=args.save_every,
        eval_every=args.eval_every,
        lora_rank=args.lora_rank,
        seed=args.seed,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )

    _print_config_summary(args, n_prompts)

    if args.dry_run:
        print("\n--dry-run set, exiting without training.")
        return

    cli_utils.check_log_dir(args.log_path, behavior_if_exists="ask")
    asyncio.run(rl_train.main(config))


if __name__ == "__main__":
    main()
