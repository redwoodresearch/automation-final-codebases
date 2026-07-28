"""fig_gpqa_vs_mmlu_following.png — incorrect-hint following, MMLU vs GPQA, all 30 models
(Claude | open-weight | GPT | Gemini), read from results/following_tables.json.

  python scripts/plot_gpqa_vs_mmlu.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from lib.figures import (CLAUDE_ORDER, DISPLAY, FIGURES, GEMINI_ORDER, GPT_ORDER, OPENWEIGHT_ORDER,
                         following_pct, load_results_json)

C_MMLU, C_GPQA = "#bfe0bf", "#3f8f3f"

def main() -> None:
    tables = load_results_json("following_tables.json")["models"]
    ORDER = CLAUDE_ORDER + OPENWEIGHT_ORDER + GPT_ORDER + GEMINI_ORDER
    SIZES = [len(CLAUDE_ORDER), len(OPENWEIGHT_ORDER), len(GPT_ORDER), len(GEMINI_ORDER)]
    LABELS = ["Claude", "open-weight", "GPT", "Gemini"]

    # x positions: equal gap between every consecutive subgroup
    xs, x = [], 0.0
    for gi, n in enumerate(SIZES):
        if gi > 0:
            x += 0.9
        for _ in range(n):
            xs.append(x)
            x += 1.0
    xs = np.array(xs)
    starts, idx = [], 0
    for n in SIZES:
        starts.append(idx)
        idx += n

    fig, ax = plt.subplots(figsize=(26, 6.5))
    w = 0.38
    for xi, s in zip(xs, ORDER):
        mm = round(following_pct(tables[s]["mmlu"]["incorrect"])[0])
        gp = round(following_pct(tables[s]["gpqa"]["incorrect"])[0])
        ax.bar(xi - w / 2, mm, w, color=C_MMLU, edgecolor="black", linewidth=0.4, zorder=2)
        ax.bar(xi + w / 2, gp, w, color=C_GPQA, edgecolor="black", linewidth=0.4, zorder=2)
        ax.text(xi - w / 2, mm + 0.7, f"{mm}", ha="center", va="bottom", fontsize=7, color="#555")
        ax.text(xi + w / 2, gp + 0.7, f"{gp}", ha="center", va="bottom", fontsize=7.5,
                color="#2f6b2f", fontweight="bold")
    for k in range(1, len(SIZES)):
        ax.axvline((xs[starts[k] - 1] + xs[starts[k]]) / 2, color="gray", lw=1.0, ls=":")
    for s0, n, lab in zip(starts, SIZES, LABELS):
        ax.text((xs[s0] + xs[s0 + n - 1]) / 2, 0.965, lab, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color="gray")
    ax.set_ylim(0, 60)
    ax.set_xlim(xs[0] - 0.8, xs[-1] + 0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([DISPLAY[s] for s in ORDER], rotation=40, ha="right", fontsize=8.5)
    ax.set_ylabel("Incorrect-hint following (% of eligible)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(handles=[mpatches.Patch(facecolor=C_MMLU, edgecolor="black", label="MMLU"),
                       mpatches.Patch(facecolor=C_GPQA, edgecolor="black", label="GPQA")],
              fontsize=9, loc="upper left", framealpha=0.95)
    FIGURES.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_gpqa_vs_mmlu_following.png", dpi=140)
    print("wrote figures/fig_gpqa_vs_mmlu_following.png")



if __name__ == "__main__":
    main()
