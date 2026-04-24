"""Load eval datasets with suspect-id filtering."""
from __future__ import annotations

import json
import os
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "datasets", "eval")


def load_suspect_ids(split: str) -> set[str]:
    """Return suspect IDs for a split (eval_v1/val_v1). Empty set if no manifest."""
    path = os.path.join(EVAL_DIR, f"{split}_suspect_ids.json")
    if not os.path.exists(path):
        return set()
    data = json.load(open(path))
    return {item["id"] for item in data.get(split, [])}


def load_split(split: str, drop_suspect: bool = True) -> tuple[list[dict[str, Any]], set[str]]:
    """Load a split's rows. Returns (rows, dropped_ids)."""
    path = os.path.join(EVAL_DIR, f"{split}.jsonl")
    rows = [json.loads(l) for l in open(path)]
    if not drop_suspect:
        return rows, set()
    suspect = load_suspect_ids(split)
    kept = [r for r in rows if r["id"] not in suspect]
    dropped = {r["id"] for r in rows} & suspect
    return kept, dropped
