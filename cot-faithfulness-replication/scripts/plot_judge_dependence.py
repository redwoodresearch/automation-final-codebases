"""fig_judge_dependence.png — the DeepSeek R1 non-replication figure.

LEFT  — R1's response to an incorrect hint (change-to-hint decomposition, MMLU+GPQA average,
        temp-0 transcripts), ours vs Chen et al.'s published number.
RIGHT — normalized faithfulness of the same transcripts under four readers: our Claude Opus 4.8
        judge, the era-matched Claude 3 Opus judge (both read from results/judge_dependence.json),
        and two external references (Young 2026 / Chua & Evans 2025, and Chen et al. 2025).

  python scripts/plot_judge_dependence.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from lib.figures import C_HINT, C_HINT_L, C_NOCHANGE, C_NOCHANGE_L, C_OTHER, C_OTHER_L, FIGURES, load_results_json

HINT_TYPES = ["suggestion", "posthoc", "fewshot_symbol", "metadata", "grader_hacking", "unethical_information"]
LAB = ["sycophancy", "consistency", "visual\nmarker", "metadata\nkey", "leaked\ngrader", "unauth.\naccess"]

W = 0.19  # bar width in the four-reader clusters


def cell(axR, xc, off, v, color, is_chua=False, bold=False):
    """One reader's bar for one hint type (amber = the Chua & Evans substitution cell)."""
    c = "#e0a030" if is_chua else color
    axR.bar(xc + off, v, W, color=c, edgecolor="black", linewidth=0.6)
    axR.text(xc + off, v + 1, f"{v}", ha="center", va="bottom",
             fontsize=(8 if bold else 6.5), fontweight=("bold" if bold else "normal"), color="#12305a")


def main() -> None:
    dep = load_results_json("judge_dependence.json")

    # ---- our numbers (from the committed analysis table) ----
    avg = dep["incorrect_hint_following"]["avg_pct"]
    OURS_DECOMP = (avg["change_to_hint"], avg["change_to_non_hint"], avg["no_change"])
    faith = dep["faithfulness_by_judge"]
    OPUS48 = [round(100 * faith["opus48"]["avg_normalized"][ht]) for ht in HINT_TYPES]
    OPUS3 = [round(100 * faith["claude3opus"]["avg_normalized"][ht]) for ht in HINT_TYPES]
    MEAN_OPUS48 = round(100 * faith["opus48"]["six_type_mean"])
    MEAN_OPUS3 = round(100 * faith["claude3opus"]["six_type_mean"])

    # ---- external reference numbers ----
    # Chen et al. 2025 (arXiv:2505.05410): R1's published incorrect-hint response (Fig 3 bars,
    # MMLU+GPQA average) and per-hint-type normalized faithfulness (Fig 1 bars, MMLU+GPQA average).
    CHEN_DECOMP = (40, 2, 58)
    CHEN = [70, 33, 9, 66, 38, 19]
    CHEN_MEAN = round(sum(CHEN) / len(CHEN))  # 39
    # Independent replications, MMLU only: Young 2026 (arXiv:2603.22582) per-hint-type R1
    # faithfulness (digitized from their per-type figure), except the visual-marker cell, where
    # Young's hint wording differs from the released ■ prompts — that cell comes from
    # Chua & Evans 2025 (arXiv:2501.08156), whose marker matches (■ only). Young's reported
    # aggregate is ~75%.
    INDEP = [62, 31, 25, 71, 81, 88]
    INDEP_IS_CHUA = [False, False, True, False, False, False]
    INDEP_MEAN = 75

    x = np.arange(len(LAB))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [0.7, 2.0]})

    # ---- left: R1 response to an incorrect hint, ours vs Chen ----
    DECOMP = [("ours", OURS_DECOMP, False), ("Chen et al. 2025", CHEN_DECOMP, True)]
    xl = np.arange(len(DECOMP))
    for xi, (lab, (hit, oth, noc), light) in zip(xl, DECOMP):
        ch, co, cn = (C_HINT_L, C_OTHER_L, C_NOCHANGE_L) if light else (C_HINT, C_OTHER, C_NOCHANGE)
        axL.bar(xi, hit, 0.55, color=ch, edgecolor="black", linewidth=0.5, zorder=2)
        axL.bar(xi, oth, 0.55, bottom=hit, color=co, edgecolor="black", linewidth=0.5, zorder=2)
        axL.bar(xi, noc, 0.55, bottom=hit + oth, color=cn, edgecolor="black", linewidth=0.5, zorder=2)
        axL.text(xi, hit + oth + 1.2, f"{hit:.0f}", ha="center", va="bottom", fontsize=11,
                 color="#2f6b2f", fontweight="bold")
    axL.set_xticks(xl)
    axL.set_xticklabels([d[0] for d in DECOMP], fontsize=9)
    axL.set_ylabel("Fraction of eligible examples")
    axL.set_title("R1 incorrect-hint following (MMLU + GPQA average)", fontsize=10)
    axL.set_ylim(0, 100)
    axL.set_xlim(-0.6, len(DECOMP) - 0.4)
    axL.grid(True, axis="y", alpha=0.3)
    axL.legend(handles=[
        mpatches.Patch(facecolor=C_HINT, edgecolor="black", label="Change to Hint"),
        mpatches.Patch(facecolor=C_OTHER, edgecolor="black", label="Change to Non-Hint"),
        mpatches.Patch(facecolor=C_NOCHANGE, edgecolor="black", label="No Change"),
    ], fontsize=8, loc="upper right", framealpha=0.95)

    # ---- right: faithfulness across four readers, per type + mean cluster ----
    w = W
    gx = len(LAB) + 0.6
    for xi in x:
        i = int(xi)
        cell(axR, xi, -1.5 * w, OPUS48[i], "#1f4e79")
        cell(axR, xi, -0.5 * w, OPUS3[i], "#bcd0e6")
        cell(axR, xi, 0.5 * w, INDEP[i], "#5cb85c", is_chua=INDEP_IS_CHUA[i])
        cell(axR, xi, 1.5 * w, CHEN[i], "#cccccc")
    axR.axvline(len(LAB) - 0.15, color="gray", ls=":", lw=1.0)
    cell(axR, gx, -1.5 * w, MEAN_OPUS48, "#1f4e79", bold=True)
    cell(axR, gx, -0.5 * w, MEAN_OPUS3, "#bcd0e6", bold=True)
    cell(axR, gx, 0.5 * w, INDEP_MEAN, "#5cb85c", bold=True)
    cell(axR, gx, 1.5 * w, CHEN_MEAN, "#cccccc", bold=True)
    axR.set_xticks(list(x) + [gx])
    axR.set_xticklabels(LAB + ["mean"], fontsize=8.5)
    axR.set_ylabel("normalized faithfulness (%)")
    axR.set_title("R1 faithfulness by judge (MMLU + GPQA average)", fontsize=10)
    axR.set_ylim(0, 116)
    axR.set_yticks([0, 20, 40, 60, 80, 100])
    axR.grid(True, axis="y", alpha=0.3)
    axR.legend(handles=[
        mpatches.Patch(facecolor="#1f4e79", edgecolor="black", label="Opus 4.8 judge"),
        mpatches.Patch(facecolor="#bcd0e6", edgecolor="black", label="Claude 3 Opus judge"),
        mpatches.Patch(facecolor="#5cb85c", edgecolor="black", label="Young 2026, MMLU only"),
        mpatches.Patch(facecolor="#e0a030", edgecolor="black", label="Chua & Evans 2025, MMLU only"),
        mpatches.Patch(facecolor="#cccccc", edgecolor="black", label="Chen et al. 2025"),
    ], fontsize=8, loc="upper center", ncol=3, framealpha=0.97)
    FIGURES.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_judge_dependence.png", dpi=140)
    print("wrote figures/fig_judge_dependence.png")



if __name__ == "__main__":
    main()
