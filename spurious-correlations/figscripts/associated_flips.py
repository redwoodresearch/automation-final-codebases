#!/usr/bin/env python3
"""associated_flips_results.png: outcome per already-correlated pair, correlated (diagonal) vs
anticorrelated (anti-diagonal) training. Concrete feature poles shown in the row labels."""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = _ROOT + "/inoc_data"
data = json.load(open(f"{D}/associated_flips_data.json"))
# 37-roster reversal flips found in the full sweep (reversal_analysis_v2.json), including
# safety-relevant ones; cong = diagonal (correlated) Δ, incong = anti-diagonal Δ, toward the first feature.
data += [
    {"pair": "argumentative structure × starts with string", "cong": 12.1, "incong": -6.2, "flip": True},
    {"pair": "hedging × sycophancy", "cong": -8.6, "incong": 3.4, "flip": True},
    {"pair": "parenthetical asides × sycophancy", "cong": -7.4, "incong": 10.6, "flip": True},
]
data = sorted(data, key=lambda d: (not d["flip"], -abs(d["cong"])))
POLES = {
    "emotional valence × hedging": "positive/negative  ×  confident/hedged",
    "gender × occupation": "man/woman  ×  firefighter/nurse",
    "legal × formal register": "legal/everyday  ×  formal/casual",
    "age × tech": "young/old  ×  tech-fluent/not",
    "finance × numbers": "finance topic  ×  many/few numbers",
    "argumentative structure × starts with string": "argues thesis/neutral  ×  begins with pinned sentence/not",
    "hedging × sycophancy": "confident/hedged  ×  goes along/corrects",
    "parenthetical asides × sycophancy": "uses asides/not  ×  goes along/corrects",
}
y = np.arange(len(data))[::-1]
CORR, ANTI, FLIPC = "#1f77b4", "#e07b00", "#c5221f"

fig, ax = plt.subplots(figsize=(11.5, 5.4))
xmax = max(abs(v) for d in data for v in (d["cong"], d["incong"])) + 2
for yi, d in zip(y, data):
    c, ic = d["cong"], d["incong"]
    lc = FLIPC if d["flip"] else "#bbbbbb"
    ax.annotate("", xy=(ic, yi), xytext=(c, yi),
                arrowprops=dict(arrowstyle="-|>,head_width=0.4,head_length=0.8", color=lc, lw=2.2, shrinkA=6, shrinkB=6))
    ax.scatter([c], [yi], color=CORR, s=95, zorder=5, edgecolor="white", linewidth=1.0)
    ax.scatter([ic], [yi], color=ANTI, s=95, zorder=5, edgecolor="white", linewidth=1.0)
    if d["flip"]:
        ax.text(min(c, ic)-0.6, yi, "flips", color=FLIPC, fontsize=9, ha="right", va="center", style="italic")

labels = [f"{d['pair']}\n({POLES[d['pair']]})" for d in data]
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5); ax.set_ylim(-0.6, len(data)-0.4)
ax.axvline(0, color="#333", lw=1.0); ax.set_xlim(-xmax, xmax)
ax.set_xlabel("learned strength margin (nats)")
ax.text(-xmax*0.99, len(data)-0.35, "← second feature wins", color="#555", fontsize=9.5, ha="left", va="center")
ax.text(xmax*0.99, len(data)-0.35, "first feature wins →", color="#555", fontsize=9.5, ha="right", va="center")

from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([0],[0], marker="o", color="w", markerfacecolor=CORR, markersize=11, label="features correlated in training (diagonal)"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=ANTI, markersize=11, label="features anticorrelated (anti-diagonal)"),
    Line2D([0],[0], color=FLIPC, lw=2.4, marker=">", label="outcome flips which feature is learned"),
], loc="lower left", fontsize=9, framealpha=0.95)
ax.set_title("Correlation direction flips the winner", fontsize=13)
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(_ROOT + "/figs/associated_flips_results.png", dpi=140, bbox_inches="tight")
print("associated_flips_results.png (poles in labels, short title)")
