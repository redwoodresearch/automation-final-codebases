"""Counterexample dose-response figure, two panels: fine-tune (left) and the
mean-pooled probe (right). Each panel shows median + IQR band across the 20 confounded
pairs, for three training-set sizes (N=75, 150, 300).

Fine-tune reads counterexample_count_grid_ext.jsonl (20 pairs x {0,1,2,4,8,16,32}
counterexamples x {75,150,300} per-class x 3 seeds, from run_counterex_ext.py). The probe
reads counterex_probe/counterex_probe_raw.jsonl (the layer-21 mean-pooled probe sweep, matched
to the fine-tune: same pairs, counts, N sizes, seeds, injection convention, and metric). For each
(N, k) we average each pair over its seeds, then take the median and 25th/75th percentile across
the 20 pairs. y = bounded probability the model follows the intended (weaker) feature.
"""
import json, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os
_RES = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "results")


def load_pair_means(path):
    """(n, count, pair) -> mean over seeds of the bounded prob of following the intended feature."""
    rows = [json.loads(l) for l in open(path)]
    byk = collections.defaultdict(list)
    for r in rows:
        byk[(r["n"], r["count"], r["pair"])].append(r["read_bounded_target"])
    return {k: float(np.mean(v)) for k, v in byk.items()}, rows


GRID = _os.path.join(_RES, "counterexample_count_grid_ext.jsonl")
if not _os.path.exists(GRID):
    GRID = _os.path.join(_RES, "counterexample_count_grid.jsonl")
ft_mean, ft_rows = load_pair_means(GRID)
PGRID = _os.path.join(_RES, "counterex_probe", "counterex_probe_raw.jsonl")
pr_mean, _pr_rows = load_pair_means(PGRID)

counts = sorted({r["count"] for r in ft_rows})        # 0,1,2,4,8,16,32
pairs  = sorted({r["pair"]  for r in ft_rows})
Ns     = sorted({r["n"]     for r in ft_rows})         # 75, 150, 300
x = np.arange(len(counts))                             # equal categorical spacing
colors = {75: "#3a7ca5", 150: "#c0603a", 300: "#4e9a51"}


def draw(ax, pair_mean, title):
    for N in Ns:
        med, lo, hi = [], [], []
        for k in counts:
            vals = np.array([pair_mean[(N, k, p)] for p in pairs if (N, k, p) in pair_mean])
            med.append(np.median(vals)); lo.append(np.percentile(vals, 25)); hi.append(np.percentile(vals, 75))
        med, lo, hi = map(np.array, (med, lo, hi))
        ax.fill_between(x, lo, hi, color=colors[N], alpha=0.16, linewidth=0)
        ax.plot(x, med, color=colors[N], linewidth=2.4, marker="o", markersize=5,
                label=f"N = {N} per class")
    ax.axhline(0.5, color="#999999", linewidth=0.9, linestyle="--", zorder=0)
    ax.set_xticks(x); ax.set_xticklabels([str(c) for c in counts])
    ax.set_xlabel("counterexamples injected per class", fontsize=11, color="#444444")
    ax.set_ylim(0, 1.02)
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=8, color="#333333")
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.5, zorder=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("bottom", "left"): ax.spines[s].set_color("#bbbbbb")
    ax.tick_params(colors="#666666")


fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
fig.patch.set_facecolor("#fafafa")
for ax in (axL, axR): ax.set_facecolor("#fafafa")

draw(axL, ft_mean, "Fine-tune")
draw(axR, pr_mean, "Probe")

axL.set_ylabel("P(classifier follows the intended feature)", fontsize=11, color="#444444")
axR.text(x[-1], 0.515, "follows intended feature", ha="right", va="bottom", fontsize=8, color="#888888")
axR.text(x[-1], 0.485, "follows spurious feature", ha="right", va="top", fontsize=8, color="#888888")
axL.legend(loc="lower right", fontsize=9.5, framealpha=0.95, edgecolor="#cccccc",
           title="line = median, band = 25-75th pct\nacross 20 confounded pairs", title_fontsize=8)

fig.suptitle("Counterexamples flip both the fine-tune and the probe",
             fontsize=13.5, fontweight="bold", color="#333333")
plt.tight_layout()
out = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                    "plots", "counterexample_flip_curves.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
print("Saved", out)

for name, pm in (("fine-tune", ft_mean), ("probe", pr_mean)):
    for N in Ns:
        meds = [round(np.median([pm[(N,k,p)] for p in pairs if (N,k,p) in pm]), 3) for k in counts]
        print(f"{name} N={N} medians:", dict(zip(counts, meds)))
