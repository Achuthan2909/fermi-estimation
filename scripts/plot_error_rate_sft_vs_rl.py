"""Regenerate error_rate_sft_vs_rl.png with N-questions histogram (right axis)."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
SFT_FILE = ROOT / "run/sft_filtered_v1_v2_val.jsonl"
RL_FILE  = ROOT / "run/rl_v2_val.jsonl"
OUT      = ROOT / "plots/error_rate_sft_vs_rl.png"


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def error_rate_by_oom(rows, min_n=1):
    buckets = defaultdict(list)
    for r in rows:
        if r.get("gold_oom") is not None:
            buckets[int(r["gold_oom"])].append(1 if r["score"] == 0 else 0)
    ooms, rates, ns = [], [], []
    for oom in sorted(buckets):
        vals = buckets[oom]
        if len(vals) >= min_n:
            ooms.append(oom)
            rates.append(100 * np.mean(vals))
            ns.append(len(vals))
    return np.array(ooms), np.array(rates), np.array(ns)


sft_rows = load(SFT_FILE)
rl_rows  = load(RL_FILE)

sft_ooms, sft_rates, sft_ns = error_rate_by_oom(sft_rows)
rl_ooms,  rl_rates,  rl_ns  = error_rate_by_oom(rl_rows)

# Histogram uses RL data (same val set, so N per OOM is identical)
hist_ooms = rl_ooms
hist_ns   = rl_ns

fig, ax1 = plt.subplots(figsize=(14, 5))
ax2 = ax1.twinx()

# Background regions
ax1.axvspan(-30, -5,  alpha=0.10, color="#8888cc", zorder=0)
ax1.axvspan(-5,  10,  alpha=0.10, color="#88cc88", zorder=0)
ax1.axvspan(10,  90,  alpha=0.10, color="#ccaa66", zorder=0)

# Histogram (grey, behind lines)
ax2.bar(hist_ooms, hist_ns, width=0.8, color="#aaaaaa", alpha=0.35,
        zorder=1, label="N questions")
ax2.set_ylabel("Number of questions", color="#888888", fontsize=11)
ax2.tick_params(axis="y", colors="#888888")
ax2.set_ylim(0, max(hist_ns) * 4)   # keep bars short like Image #1

# Error rate lines
ax1.plot(sft_ooms, sft_rates, color="#4c72b0", marker="o", markersize=3,
         linewidth=1.5, label="SFT", zorder=3)
ax1.plot(rl_ooms,  rl_rates,  color="#dd8844", marker="o", markersize=3,
         linewidth=1.5, label="RL v2", zorder=3)

ax1.set_xlabel("Gold OOM", fontsize=12)
ax1.set_ylabel("Error Rate (%)", fontsize=12)
ax1.set_title("Error Rate by OOM — SFT vs RL v2", fontsize=13, fontweight="bold")
ax1.set_ylim(0, 105)
ax1.set_xlim(min(sft_ooms.min(), rl_ooms.min()) - 2,
             max(sft_ooms.max(), rl_ooms.max()) + 2)

# Legend: combine both axes
line_handles, line_labels = ax1.get_legend_handles_labels()
hist_patch = mpatches.Patch(color="#aaaaaa", alpha=0.5, label="N questions")
ax1.legend(handles=line_handles + [hist_patch], loc="upper right", fontsize=10)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}")
