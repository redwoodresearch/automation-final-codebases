"""Near-tie collapse figure: near-tie pairs commit to DIFFERENT winners across retraining runs.

Data: results/complete_graph/commitment_stability_census.json (re-derivable from the frozen table via
scratch/v2_antidiagonal_reversal/rederive_commitment_census.py). The census marks two groups by the
spread of the three plateau-checkpoint fine-tune run values:
  - unstable_commitment: the 3 runs disagree on the WINNER (not all same sign) — the near-tie-collapse
    population (phrase as "unstable under retraining", NOT "bistable by seed" — see the census note).
  - high_spread_same_winner: large spread but a stable winner (large-magnitude retraining noise).

Writes plots/near_tie_collapse.png (top-level; the blog references ../plots/near_tie_collapse.png).
CPU-only, reads a committed artifact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "lib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402

CENSUS = config.RESULTS_DIR / "complete_graph" / "commitment_stability_census.json"
OUT = config.PLOTS_DIR / "near_tie_collapse.png"

# Short, reader-facing feature labels (only the ones appearing in the census need entries; the
# fallback prettifier handles the rest).
SHORT = {
    "numerical_density": "numeric density",
    "substring_rare_wordlike": "rare string",
    "substring_common_word": '"walnut"',
    "substring_high_entropy": '"xq7#Kp2v"',
    "leetspeak": "leetspeak",
    "all_caps": "ALL CAPS",
    "specific_emoji": "emoji",
    "ends_with_string": "closing sentence",
    "starts_with_string": "opening sentence",
    "trailing_signature": "signature",
    "entity_mention_trigger": "entity mention",
    "ai_self_reference": "AI self-reference",
    "harmful_vs_benign": "harmful advice",
    "sycophantic_agreement": "sycophancy",
    "refusal_language": "refusal",
    "topic_medical": "medical topic",
    "emotional_tone": "emotional tone",
    "passive_voice": "passive voice",
    "formal_register": "formal register",
    "parenthetical_asides": "parentheticals",
    "acronym_density": "acronyms",
    "domain_jargon": "jargon",
}


def pretty(feat: str) -> str:
    return SHORT.get(feat, feat.replace("_", " "))


def pair_label(pair: str) -> str:
    a, b = pair.split("__")
    return f"{pretty(a)} × {pretty(b)}"


def main() -> None:
    census = json.loads(CENSUS.read_text())
    unstable = census["unstable_commitment_pairs"]
    same_winner = census["high_spread_same_winner_pairs"]

    # Order each group by mean readout so the plot reads top (follows f2) to bottom (follows f1).
    unstable = sorted(unstable, key=lambda r: r["ft_mean"])
    same_winner = sorted(same_winner, key=lambda r: r["ft_mean"])

    rows = [("unstable", r) for r in unstable] + [("stable", r) for r in same_winner]
    n = len(rows)

    fig, ax = plt.subplots(figsize=(8.2, 0.34 * n + 1.6))
    for y, (grp, r) in enumerate(rows):
        vals = r["ft_run_values"]
        color = "#d62728" if grp == "unstable" else "#9a9a9a"
        ax.plot([min(vals), max(vals)], [y, y], color=color, lw=1.2, alpha=0.55, zorder=1)
        ax.scatter(vals, [y] * len(vals), s=34, color=color, edgecolors="white",
                   linewidths=0.6, zorder=3)
        ax.scatter([r["ft_mean"]], [y], s=90, marker="|", color=color, zorder=2)

    ax.axvline(0, color="k", lw=1.0, ls="--", alpha=0.7)
    ax.set_yticks(range(n))
    ax.set_yticklabels([pair_label(r["pair"]) for _, r in rows], fontsize=8)
    # Extra bottom whitespace so the (wide) legend sits below the bottom row, not on it.
    ax.set_ylim(-4.2, n - 0.2)
    ax.set_xlabel("fine-tune readout Δ per retraining run (nats):  Δ>0 follows f1,  Δ<0 follows f2",
                  fontsize=10)
    ax.set_title("Near-tie winners flip across retrainings", fontsize=12)

    # Legend proxies.
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="o", color="#d62728", lw=0, markersize=7,
               label=f"unstable commitment: 3 runs disagree on the winner (n={len(unstable)})"),
        Line2D([0], [0], marker="o", color="#9a9a9a", lw=0, markersize=7,
               label=f"large spread, same winner (n={len(same_winner)})"),
    ]
    ax.legend(handles=handles, fontsize=8.5, loc="lower right", framealpha=0.9)
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT} ({len(unstable)} unstable, {len(same_winner)} same-winner pairs)")


if __name__ == "__main__":
    main()
