"""SFT fine-tuning of Qwen3-4B-Instruct-2507 on Fermi reasoning traces.

Run prepare_traces.py first to build datasets/sft_messages/train.jsonl.

Usage:
    python sft/train_sft.py
    python sft/train_sft.py --num-epochs 3 --log-path run/sft_v2
    python sft/train_sft.py --dry-run          # print config only, no training
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src" / "preprocessing"))

from typing import Any

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.hyperparam_utils import get_lr
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
# qwen3_instruct: no thinking tokens — correct for the 2507 instruction variant
RENDERER_NAME = model_info.get_recommended_renderer_name(MODEL_NAME)

_DEFAULT_TRAIN_FILE = ROOT / "datasets" / "sft_messages" / "train.jsonl"
_DEFAULT_LOG_PATH = ROOT / "run" / "sft_v1"


def build_config(
    train_file: str,
    log_path: str,
    num_epochs: int,
    batch_size: int,
    max_length: int,
    lora_rank: int,
    wandb_project: str | None,
    wandb_name: str | None,
) -> train.Config:
    lr = get_lr(MODEL_NAME, is_lora=True)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=MODEL_NAME,
        renderer_name=RENDERER_NAME,
        max_length=max_length,
        batch_size=batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    dataset = FromConversationFileBuilder(
        common_config=common_config,
        file_path=train_file,
        # No val split here — we handle val separately via prepare_traces.py.
        test_size=0,
        shuffle_seed=42,
    )

    overrides: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "renderer_name": RENDERER_NAME,
        "log_path": log_path,
        "dataset_builder": dataset,
        "learning_rate": lr,
        "lr_schedule": "linear",
        "num_epochs": num_epochs,
        "lora_rank": lora_rank,
    }
    if wandb_project:
        overrides["wandb_project"] = wandb_project
    if wandb_name:
        overrides["wandb_name"] = wandb_name
    blueprint = chz.Blueprint(train.Config).apply(overrides)
    return blueprint.make()


def main() -> None:
    if not os.environ.get("TINKER_API_KEY"):
        sys.exit("TINKER_API_KEY is not set. Run: source .env")

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", default=str(_DEFAULT_TRAIN_FILE),
                    help="Path to train.jsonl produced by prepare_traces.py")
    ap.add_argument("--log-path", default=str(_DEFAULT_LOG_PATH))
    ap.add_argument("--num-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2048,
                    help="Batch size in tokens (not examples)")
    ap.add_argument("--max-length", type=int, default=2048,
                    help="Max tokens per training example (prompt + response)")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--wandb-project", default="fermi-sft",
                    help="Wandb project name. Pass empty string to disable wandb.")
    ap.add_argument("--wandb-name", default=None,
                    help="Wandb run name. Defaults to basename of --log-path.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print resolved config and exit without training")
    args = ap.parse_args()
    wandb_project = args.wandb_project or None
    wandb_name = args.wandb_name or (Path(args.log_path).name if wandb_project else None)

    train_file = Path(args.train_file)
    if not train_file.exists():
        sys.exit(
            f"Train file not found: {train_file}\n"
            "Run `python sft/prepare_traces.py` first."
        )

    config = build_config(
        train_file=str(train_file),
        log_path=args.log_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        lora_rank=args.lora_rank,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )

    print(f"Model:       {MODEL_NAME}")
    print(f"Renderer:    {RENDERER_NAME}")
    print(f"LR:          {get_lr(MODEL_NAME, is_lora=True):.2e}")
    print(f"Epochs:      {args.num_epochs}")
    print(f"Batch size:  {args.batch_size} tokens")
    print(f"Max length:  {args.max_length} tokens")
    print(f"LoRA rank:   {args.lora_rank}")
    print(f"Train file:  {train_file}")
    print(f"Log path:    {args.log_path}")

    n_examples = sum(1 for _ in train_file.open())
    lr = get_lr(MODEL_NAME, is_lora=True)
    avg_tokens = 1200  # ~240 prompt + ~800 response + some padding
    total_train_tokens = n_examples * avg_tokens * args.num_epochs
    est_cost = total_train_tokens / 1e6 * 0.22
    print(f"\nDataset:     {n_examples} examples")
    print(f"Est. tokens: {total_train_tokens:,} (~{total_train_tokens/1e6:.2f}M)")
    print(f"Est. cost:   ${est_cost:.2f}  (@ $0.22/1M train tokens)")

    if args.dry_run:
        print("\n--dry-run set, exiting without training.")
        return

    cli_utils.check_log_dir(args.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
