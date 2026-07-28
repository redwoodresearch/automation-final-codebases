"""fig_following_sonnets.png — 'models still follow hints', three Sonnets side by side:
Chen et al.'s reported Claude 3.5 Sonnet (New) and 3.7 Sonnet next to our Sonnet 4.5.
Change-to-hint decomposition, hint-correct and hint-incorrect panels. Our Sonnet 4.5 bar is
the equal-weight MMLU+GPQA average read from results/following_tables.json.

  python scripts/plot_following_sonnets.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from lib.figures import (C_HINT, C_HINT_L, C_NOCHANGE, C_NOCHANGE_L, C_OTHER, C_OTHER_L, FIGURES,
                         following_avg_pct, load_results_json)

# Chen et al. 2025 (arXiv:2505.05410) reported hint response, their published MMLU+GPQA
# average (change-to-hint, change-to-non-hint, no-change), read off the paper's Figure 3
# ("model hint response") bars.
CHEN = {
    "Claude 3.5\nSonnet (New)\n(Chen et al.)": {"correct": (90, 1, 9), "incorrect": (64, 1, 35)},
    "Claude 3.7\nSonnet\n(Chen et al.)": {"correct": (84, 1, 15), "incorrect": (51, 1, 48)},
}

LABELS = list(CHEN) + ["Sonnet 4.5\n(this work)"]


def panel(ax, direction, title, ours, x):
    data = [CHEN[k][direction] for k in CHEN] + [ours[direction]]
    for i, (xi, (hit, oth, noc)) in enumerate(zip(x, data)):
        light = i < len(CHEN)  # lighter = Chen's reported models
        ch, co, cn = (C_HINT_L, C_OTHER_L, C_NOCHANGE_L) if light else (C_HINT, C_OTHER, C_NOCHANGE)
        ax.bar(xi, hit, 0.6, color=ch, edgecolor="black", linewidth=0.5, zorder=2)
        ax.bar(xi, oth, 0.6, bottom=hit, color=co, edgecolor="black", linewidth=0.5, zorder=2)
        ax.bar(xi, noc, 0.6, bottom=hit + oth, color=cn, edgecolor="black", linewidth=0.5, zorder=2)
        ax.text(xi, hit + oth + 1.2, f"{hit:.0f}", ha="center", va="bottom", fontsize=12,
                color="#2f6b2f", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=12)
    ax.grid(True, axis="y", alpha=0.25)


def main() -> None:
    tables = load_results_json("following_tables.json")["models"]
    ours = {d: following_avg_pct(tables["sonnet-4-5"], d) for d in ("correct", "incorrect")}

    x = np.arange(len(LABELS))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)


    panel(a1, "correct", "Hint Correct", ours, x)
    panel(a2, "incorrect", "Hint Incorrect", ours, x)
    a1.set_ylabel("Fraction of examples")
    a1.legend(handles=[
        mpatches.Patch(facecolor=C_HINT, edgecolor="black", label="Change to Hint"),
        mpatches.Patch(facecolor=C_OTHER, edgecolor="black", label="Change to Non-Hint"),
        mpatches.Patch(facecolor=C_NOCHANGE, edgecolor="black", label="No Change"),
    ], fontsize=9, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "fig_following_sonnets.png", dpi=140)
    print("wrote figures/fig_following_sonnets.png")



if __name__ == "__main__":
    main()
