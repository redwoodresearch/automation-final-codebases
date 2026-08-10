#!/usr/bin/env python3
"""feature_rank_trajectories.png: each feature's strength RANK across probe layers (full 37 roster).
The 12 biggest movers across depth are drawn in their own distinct color; the rest are light grey.
Every feature is labelled at the right edge (movers in color, the rest in light grey)."""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(__file__))
from feature_names import disp

D = _ROOT + ""
s_by_layer = json.load(open(f"{D}/knockout/results/depth37/s_by_layer_37.json"))
layers = sorted(s_by_layer, key=int)
feats = list(s_by_layer[layers[0]])   # full 37 roster
xs = [int(l) for l in layers]

# rank per layer: 1 = strongest
rank = {f: [] for f in feats}
for l in layers:
    order = sorted(feats, key=lambda f: s_by_layer[l][f], reverse=True)
    pos = {f: i + 1 for i, f in enumerate(order)}
    for f in feats:
        rank[f].append(pos[f])

# "changes a lot throughout" = total variation (sum of rank changes across layers),
# which catches features that wobble hard mid-network even if their net swing is small,
# not just start-to-end range.
tv = {f: sum(abs(rank[f][i + 1] - rank[f][i]) for i in range(len(rank[f]) - 1)) for f in feats}
by_tv = sorted(feats, key=lambda f: tv[f], reverse=True)
highlight = by_tv[:12]                           # the 12 biggest movers across depth
palette = (list(plt.cm.tab10.colors) + list(plt.cm.tab20b.colors))
colors = {f: palette[i] for i, f in enumerate(highlight)}

fig, ax = plt.subplots(figsize=(13, 11))
for f in feats:
    if f in highlight:
        continue
    ax.plot(xs, rank[f], color="#d9d9d9", lw=0.9, zorder=1)
    ax.text(xs[-1] + 0.3, rank[f][-1], disp(f), color="#9e9e9e",
            fontsize=8, va="center", ha="left")
for f in highlight:
    ax.plot(xs, rank[f], color=colors[f], lw=2.0, zorder=3)
    ax.text(xs[-1] + 0.3, rank[f][-1], disp(f), color=colors[f],
            fontsize=9, va="center", ha="left", fontweight="bold")

ax.set_xlim(xs[0] - 0.5, xs[-1] + 9)
ax.set_ylim(len(feats) + 1, 0)                   # rank 1 at top
ax.set_xlabel("layer")
ax.set_ylabel("strength rank  (1 = strongest)")
ax.set_title("Feature ranking across depth", fontsize=13)
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(f"{D}/figs/feature_rank_trajectories.png", dpi=150, bbox_inches="tight")
print(f"feature rank trajectories: {len(highlight)} highlighted, {len(feats)-len(highlight)} grey")
