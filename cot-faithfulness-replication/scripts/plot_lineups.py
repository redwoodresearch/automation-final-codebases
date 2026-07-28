"""The four lineup figures (everything equal-weight MMLU+GPQA average, read from the
committed results tables):
  fig_following_main.png  — Chen's 4 reported models (lighter bars) + our 10 Claude models,
                            change-to-hint decomposition, both hint directions
  fig_following_other.png — the 20 other models (open-weight | GPT | Gemini)
  fig_faith_claude.png    — 10 Claude models, per-direction normalized faithfulness
  fig_faith_other.png     — the 20 other models, same

  python scripts/plot_lineups.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from lib.figures import (C_HINT, C_HINT_L, C_NOCHANGE, C_NOCHANGE_L, C_OTHER, C_OTHER_L, CLAUDE_ORDER,
                         DISPLAY, F_CORRECT, F_WRONG, FIGURES, GEMINI_ORDER, GPT_ORDER, OPENWEIGHT_ORDER,
                         following_avg_pct, load_results_json)

FOLLOWING = load_results_json("following_tables.json")["models"]
FAITH = load_results_json("faithfulness_tables.json")["models"]

# Chen et al. 2025 (arXiv:2505.05410) reported hint response for their four evaluated models —
# their published MMLU+GPQA average (change-to-hint, change-to-non-hint, no-change), read off
# the paper's Figure 3 bars.
CHEN = {
    "Claude 3.5 Sonnet (New)": {"correct": (90, 1, 9), "incorrect": (64, 1, 35)},
    "Claude 3.7 Sonnet": {"correct": (84, 1, 15), "incorrect": (51, 1, 48)},
    "DeepSeek V3": {"correct": (76, 3, 21), "incorrect": (39, 4, 57)},
    "DeepSeek R1": {"correct": (76, 2, 22), "incorrect": (40, 2, 58)},
}

OTHER = OPENWEIGHT_ORDER + GPT_ORDER + GEMINI_ORDER
OTHER_SIZES = [len(OPENWEIGHT_ORDER), len(GPT_ORDER), len(GEMINI_ORDER)]
OTHER_LABELS = ["open-weight", "GPT", "Gemini"]


def xpos(sizes, gap=0.9):
    """x positions with a gap between consecutive subgroups of the given sizes."""
    xs, x = [], 0.0
    for gi, n in enumerate(sizes):
        if gi > 0:
            x += gap
        for _ in range(n):
            xs.append(x)
            x += 1.0
    return np.array(xs)


def dividers_and_labels(ax, sizes, labels):
    xs = xpos(sizes)
    starts, idx = [], 0
    for n in sizes:
        starts.append(idx)
        idx += n
    for k in range(1, len(sizes)):
        ax.axvline((xs[starts[k] - 1] + xs[starts[k]]) / 2, color="gray", lw=1.0, ls=":")
    for s, n, lab in zip(starts, sizes, labels):
        if lab:
            ax.text((xs[s] + xs[s + n - 1]) / 2, 0.965, lab, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8.5, color="gray")


def following_fig(shorts, sizes, labels, fname, figw):
    xs = xpos(sizes)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(figw, 8.6), sharex=True)

    def panel(ax, direction, title):
        for xi, s in zip(xs, shorts):
            hit, oth, noc = following_avg_pct(FOLLOWING[s], direction)
            ax.bar(xi, hit, color=C_HINT, edgecolor="black", linewidth=0.4, zorder=2)
            ax.bar(xi, oth, bottom=hit, color=C_OTHER, edgecolor="black", linewidth=0.4, zorder=2)
            ax.bar(xi, noc, bottom=hit + oth, color=C_NOCHANGE, edgecolor="black", linewidth=0.4, zorder=2)
            ax.text(xi, hit + oth + 1.2, f"{hit:.0f}", ha="center", va="bottom", fontsize=8,
                    color="#2f6b2f", fontweight="bold")
        ax.set_ylim(0, 100)
        ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
        ax.set_ylabel("Fraction of eligible examples")
        ax.set_title(title, fontsize=11, pad=8)
        ax.grid(True, axis="y", alpha=0.25)

    panel(a1, "correct", "Hint Correct")
    panel(a2, "incorrect", "Hint Incorrect")
    if len(sizes) > 1:
        dividers_and_labels(a1, sizes, labels)
    a2.set_xticks(xs)
    a2.set_xticklabels([DISPLAY[s] for s in shorts], rotation=40, ha="right", fontsize=8.5)
    a2.legend(handles=[mpatches.Patch(facecolor=C_HINT, edgecolor="black", label="Change to Hint"),
                       mpatches.Patch(facecolor=C_OTHER, edgecolor="black", label="Change to Non-Hint"),
                       mpatches.Patch(facecolor=C_NOCHANGE, edgecolor="black", label="No Change")],
              fontsize=9, loc="upper right", framealpha=0.96)
    fig.tight_layout()
    fig.savefig(FIGURES / fname, dpi=140)
    print("wrote figures/" + fname)


def faith_fig(shorts, sizes, labels, fname, figw):
    xs = xpos(sizes)
    fig, ax = plt.subplots(figsize=(figw, 5.6))
    w = 0.38
    for xi, s in zip(xs, shorts):
        avg = FAITH[s]["mean_normalized"]["avg"]
        c, wr = round(100 * avg["correct"]), round(100 * avg["incorrect"])
        ax.bar(xi - w / 2, c, w, color=F_CORRECT, edgecolor="black", linewidth=0.4, zorder=2)
        ax.bar(xi + w / 2, wr, w, color=F_WRONG, edgecolor="black", linewidth=0.4, zorder=2)
        ax.text(xi - w / 2, c + 1, f"{c}", ha="center", va="bottom", fontsize=7, color="#555")
        ax.text(xi + w / 2, wr + 1, f"{wr}", ha="center", va="bottom", fontsize=7.5,
                color="#7a2e5e", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xlim(xs[0] - 0.8, xs[-1] + 0.8)
    ax.set_ylabel("mean normalized faithfulness (%)")
    ax.grid(True, axis="y", alpha=0.25)
    if len(sizes) > 1:
        dividers_and_labels(ax, sizes, labels)
    ax.set_xticks(xs)
    ax.set_xticklabels([DISPLAY[s] for s in shorts], rotation=40, ha="right", fontsize=8.5)
    ax.legend(handles=[mpatches.Patch(facecolor=F_CORRECT, edgecolor="black", label="Hint Correct"),
                       mpatches.Patch(facecolor=F_WRONG, edgecolor="black", label="Hint Incorrect")],
              fontsize=9, loc="upper left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(FIGURES / fname, dpi=140)
    print("wrote figures/" + fname)


def following_main_fig():
    """Chen's 4 reported models (lighter bars) + our 10 Claude models (solid)."""
    entries = [(m + " (Chen)", CHEN[m], True) for m in CHEN]
    entries += [(DISPLAY[s], {d: following_avg_pct(FOLLOWING[s], d) for d in ("correct", "incorrect")}, False)
                for s in CLAUDE_ORDER]
    n_chen = len(CHEN)
    xs = xpos([n_chen, len(CLAUDE_ORDER)])
    sep = (xs[n_chen - 1] + xs[n_chen]) / 2
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8.6), sharex=True)

    def panel(ax, direction, title):
        for xi, (_lab, d, is_rep) in zip(xs, entries):
            hit, oth, noc = d[direction]
            ch, co, cn = (C_HINT_L, C_OTHER_L, C_NOCHANGE_L) if is_rep else (C_HINT, C_OTHER, C_NOCHANGE)
            ax.bar(xi, hit, color=ch, edgecolor="black", linewidth=0.4, zorder=2)
            ax.bar(xi, oth, bottom=hit, color=co, edgecolor="black", linewidth=0.4, zorder=2)
            ax.bar(xi, noc, bottom=hit + oth, color=cn, edgecolor="black", linewidth=0.4, zorder=2)
            if not is_rep:
                ax.text(xi, hit + oth + 1.2, f"{hit:.0f}", ha="center", va="bottom", fontsize=8,
                        color="#2f6b2f", fontweight="bold")
        ax.axvline(sep, color="black", lw=1.0, ls=":")
        ax.set_ylim(0, 100)
        ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
        ax.set_ylabel("Fraction of eligible examples")
        ax.set_title(title, fontsize=11, pad=8)
        ax.grid(True, axis="y", alpha=0.25)

    panel(a1, "correct", "Hint Correct")
    panel(a2, "incorrect", "Hint Incorrect")
    a1.text((xs[0] + xs[n_chen - 1]) / 2, 0.965, "Chen et al. 2025 (reported)",
            transform=a1.get_xaxis_transform(), ha="center", va="top", fontsize=9, style="italic", color="#333")
    a1.text((xs[n_chen] + xs[-1]) / 2, 0.965, "Claude models (this work)",
            transform=a1.get_xaxis_transform(), ha="center", va="top", fontsize=9, style="italic", color="#333")
    a2.set_xticks(xs)
    a2.set_xticklabels([e[0] for e in entries], rotation=40, ha="right", fontsize=8.5)
    a2.legend(handles=[mpatches.Patch(facecolor=C_HINT, edgecolor="black", label="Change to Hint"),
                       mpatches.Patch(facecolor=C_OTHER, edgecolor="black", label="Change to Non-Hint"),
                       mpatches.Patch(facecolor=C_NOCHANGE, edgecolor="black", label="No Change")],
              fontsize=9, loc="upper right", framealpha=0.96)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_following_main.png", dpi=140)
    print("wrote figures/fig_following_main.png")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    following_main_fig()
    following_fig(OTHER, OTHER_SIZES, OTHER_LABELS, "fig_following_other.png", 16)
    faith_fig(CLAUDE_ORDER, [len(CLAUDE_ORDER)], [""], "fig_faith_claude.png", 11)
    faith_fig(OTHER, OTHER_SIZES, OTHER_LABELS, "fig_faith_other.png", 16)


if __name__ == "__main__":
    main()
