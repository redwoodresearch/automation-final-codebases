"""Sonnet 4.5 per-hint-type detail (reads results/sonnet45_detail.json):
  fig_sonnet45_following.png    — hint-correct + hint-incorrect change-to-hint decomposition
                                  (equal-weight MMLU + GPQA average, like the lineup figures)
                                  (full released MMLU pool, 2,994 questions)
  fig_sonnet45_faithfulness.png — per-direction normalized faithfulness (equal-weight MMLU+GPQA)

  python scripts/plot_sonnet45.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from lib.figures import C_HINT, C_NOCHANGE, C_OTHER, F_CORRECT, F_WRONG, FIGURES, load_results_json

HINT_TYPES = ["suggestion", "posthoc", "fewshot_symbol", "metadata", "grader_hacking", "unethical_information"]
LABELS = ["sycophancy", "consistency", "visual\nmarker", "metadata\nkey", "leaked\ngrader", "unauth.\naccess"]

def stacked(ax, direction, title, following, x):
    for xi, ht in zip(x, HINT_TYPES):
        a = following[ht][direction]["avg_pct"]
        hit, oth, noc = a["change_to_hint"], a["change_to_non_hint"], a["no_change"]
        ax.bar(xi, hit, color=C_HINT, edgecolor="black", linewidth=0.4, zorder=2)
        ax.bar(xi, oth, bottom=hit, color=C_OTHER, edgecolor="black", linewidth=0.4, zorder=2)
        ax.bar(xi, noc, bottom=hit + oth, color=C_NOCHANGE, edgecolor="black", linewidth=0.4, zorder=2)
        ax.text(xi, hit + oth + 1.2, f"{hit:.0f}", ha="center", va="bottom", fontsize=8.5,
                color="#2f6b2f", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=8.5)
    ax.set_ylabel("Fraction of eligible examples")
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="y", alpha=0.25)


def main() -> None:
    detail = load_results_json("sonnet45_detail.json")
    following = detail["following"]
    faith = detail["faithfulness"]
    x = np.arange(len(HINT_TYPES))

    # ---- following: two stacked panels ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6))


    stacked(a1, "correct", "Hint Correct", following, x)
    stacked(a2, "incorrect", "Hint Incorrect", following, x)
    a1.legend(handles=[mpatches.Patch(facecolor=C_HINT, edgecolor="black", label="Change to Hint"),
                       mpatches.Patch(facecolor=C_OTHER, edgecolor="black", label="Change to Non-Hint"),
                       mpatches.Patch(facecolor=C_NOCHANGE, edgecolor="black", label="No Change")],
              fontsize=8.5, loc="center left", framealpha=0.95)
    FIGURES.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_sonnet45_following.png", dpi=140)
    print("wrote figures/fig_sonnet45_following.png")

    # ---- faithfulness: per direction ----
    faith_c = [round(100 * faith[ht]["correct"]["avg"]) for ht in HINT_TYPES]
    faith_w = [round(100 * faith[ht]["incorrect"]["avg"]) for ht in HINT_TYPES]
    fig2, ax = plt.subplots(figsize=(11, 5.2))
    w = 0.38
    ax.bar(x - w / 2, faith_c, w, color=F_CORRECT, edgecolor="black", linewidth=0.5, label="Hint Correct")
    ax.bar(x + w / 2, faith_w, w, color=F_WRONG, edgecolor="black", linewidth=0.5, label="Hint Incorrect")
    for xi, a, b in zip(x, faith_c, faith_w):
        ax.text(xi - w / 2, a + 1, f"{a}", ha="center", va="bottom", fontsize=8, color="#555")
        ax.text(xi + w / 2, b + 1, f"{b}", ha="center", va="bottom", fontsize=8, color="#7a2e5e", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_ylabel("normalized faithfulness (%)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.95)
    fig2.tight_layout()
    fig2.savefig(FIGURES / "fig_sonnet45_faithfulness.png", dpi=140)
    print("wrote figures/fig_sonnet45_faithfulness.png")



if __name__ == "__main__":
    main()
