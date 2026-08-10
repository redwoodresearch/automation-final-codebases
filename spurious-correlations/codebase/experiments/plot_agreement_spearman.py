"""Horizontal bar chart: how well does each alternative readout/condition recover
the *same* feature-strength ordering as the Qwen3-1.7B fine-tune?

Two units of "ordering" appear, grouped and flagged:
  - per-feature  : Spearman over the 21-feature strength ranking (402-pair refit).
  - per-pair     : Spearman over individual pair commitments (201 pairs; behavior 20).

All numbers are recomputed locally (refit_21_402, _pair_deltas*, llm_predict) or
taken from the committed run writeups (cross-model, reversal, behavior).

Flags:
  *  subset roster (20 non-language features), not the full 21 / 402-pair set
  +  reference is the 1.7B *probe* ranking (cross-model probe not re-referenced to ft)
  ‡  separate roster (behavior x cue grid), not the feature set

(In-context learning is the per-pair Spearman of the 16-shot in-context readout against the
fine-tune strength gap, recomputed from results/icl_diagonal.jsonl over the 32 ICL pairs; the
in-context readout tracks the ordering at ~0.64 but at ~6x smaller magnitude.)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Bars: (label, rho, N, category, unit, flags). Values live in a committed data file
# so the figure is reproducible and the provenance of each number is traceable; see
# results/agreement_spearman_data.json (cross-model per-pair from the original run's
# A/B margins; reversal / probe<->ft / llm-predict recomputed from refit_21_402 +
# _pair_deltas_*; behavior + ICL from the run writeups).
import json as _json
import os as _os
_DATA = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                      "results", "agreement_spearman_data.json")
comparisons = [tuple(c) for c in _json.load(open(_DATA))]

# Colored by MEASUREMENT TYPE, so within each model block the sequence reads ft -> probe -> decision
# -> passage-averaged in a consistent color order. Bars are grouped by model (not ranked by rho).
cat_colors = {
    "ft":         "#6aaa5d",   # fine-tune (green)
    "probe":      "#3a7ca5",   # linear probe (blue)
    "decision":   "#9c755f",   # decision-token read (brown)
    "meanpooled": "#c9a98c",   # passage-averaged read (tan)
    "alt":        "#4c9a8f",   # alternative training / readout (teal)
    "reversal":   "#8b6bae",
    "behavior":   "#d48a3e",
    "weak":       "#a35a5a",
    "icl":        "#8a8a8a",
}
cat_labels = {
    "ft":         "Fine-tune",
    "probe":      "Linear probe",
    "decision":   "Decision-token read",
    "meanpooled": "Passage-averaged read",
    "alt":        "Alternative training / readout",
    "reversal":   "Reversed correlation",
    "behavior":   "Generative behavior",
    "weak":       "LLM self-prediction",
    "icl":        "In-context learning",
}

import json as _j2
_n = len(_j2.load(open(_DATA)))
fig, ax = plt.subplots(figsize=(7.2, max(4.5, 0.40 * _n)))
fig.patch.set_facecolor("#fafafa")
ax.set_facecolor("#fafafa")

y_pos = np.arange(len(comparisons))[::-1]

for i, (label, rho, n, cat, unit, flags) in enumerate(comparisons):
    y = y_pos[i]
    ax.barh(y, rho, height=0.66, color=cat_colors[cat], alpha=0.88,
            edgecolor="#fafafa", linewidth=0.8)
    unit_note = ", per-feature" if unit == "per-feature" else ("" if unit == "per-pair" else f", {unit}")
    txt = f"{rho:.2f}{(' ' + flags) if flags else ''}   (N={n}{unit_note})"
    if rho >= 0.65:
        ax.text(rho - 0.018, y, txt, va="center", ha="right",
                fontsize=8.5, fontweight="bold", color="#ffffff")
    else:
        ax.text(rho + 0.012, y, txt, va="center", ha="left",
                fontsize=8.5, fontweight="bold", color="#444444")

ax.set_yticks(y_pos)
ax.set_yticklabels([c[0] for c in comparisons], fontsize=9.5, color="#333333")
ax.set_xlim(0, 1.0)
ax.set_xlabel("Spearman ρ vs Qwen3-1.7B fine-tune strength ordering",
              fontsize=11, color="#444444")
ax.set_title("Feature Strength is Consistent Across Measurements",
             fontsize=13, fontweight="bold", pad=12, color="#333333")

ax.axvline(x=1.0, color="#d0d0d0", linewidth=0.8, linestyle="--", zorder=0)
ax.tick_params(axis='x', colors="#666666")
ax.grid(axis='x', color="#e8e8e8", linewidth=0.5, zorder=0)

from matplotlib.patches import Patch
seen = []
legend_elements = []
for c in comparisons:
    cat = c[3]
    if cat not in seen:
        seen.append(cat)
        legend_elements.append(Patch(facecolor=cat_colors[cat], alpha=0.88,
                                     label=cat_labels[cat]))
ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.10),
          ncol=2, fontsize=8.5, framealpha=0.95, edgecolor="#cccccc")

fig.text(0.012, 0.012,
         "All bars are Spearman vs the Qwen3-1.7B fine-tune ordering; per-pair unless marked per-feature.\n"
         "Probe-by-layer bars use the 124-pair cross-model sweep at the marked layer.\n"
         "*  fine-tune transfer on the 33-pair cross-model roster, not the full 402\n"
         "+  this one bar is referenced to the 1.7B probe ranking.\n"
         "§  prior 21-feature roster; the 16 newer features are not yet covered.",
         fontsize=7.5, color="#777777", ha="left", va="bottom")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color("#bbbbbb")
ax.spines["left"].set_color("#bbbbbb")

plt.tight_layout(rect=(0, 0.14, 1, 1))
_OUT = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                     "plots", "agreement_spearman.png")
plt.savefig(_OUT, dpi=200, bbox_inches="tight")
print("Saved agreement_spearman.png")
