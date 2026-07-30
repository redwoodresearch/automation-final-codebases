"""Figure 1 — the headline: endpoint subset-sum capability by condition (the five baselines + base).

One dot per condition = the mean neutral-prompt subset-sum accuracy at the end of training
(across seeds); faint dots = the individual seeds; error bar = across-seed SE. The base model
(no training) is drawn as a dashed reference line.

The five baselines (see REPORT.md / blogpost.md):
  1. Reward laundering        (GRPO condition B)  — reward given only for correct 2-digit addition,
                                                    but the prompt ties the addition answer to solving
                                                    a never-rewarded subset-sum problem.
  2. Direct subset-sum reward (GRPO condition C)  — plain subset-sum prompt, rewarded directly.
  3. Shuffled reward          (GRPO condition D)  — B's setup with each group's rewards permuted
                                                    within-group (severs reward<->behaviour link).
  4. On-policy unfiltered SFT — fine-tune on the model's own reward-laundering-prompt rollouts,
                                regenerated each round, with NO reward and NO correctness filter.
  5. Base-rollout unfiltered SFT — fine-tune on a fixed pool of base-model rollouts, same prompt,
                                   NO reward and NO correctness filter.

Reads only committed verdict JSONs (no sampling): results/multiseed_verdict_*.json (B/C/D + base)
and results/unfiltered_sft_verdict.json (the two SFT baselines). Regenerate the verdicts first with
analyze_multiseed.py and analyze_unfiltered_sft.py.

Usage: .venv/bin/python plot_headline.py
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"


def _latest(pattern: str) -> str:
    files = sorted(glob.glob(str(RESULTS / pattern)))
    if not files:
        raise SystemExit(f"no file matches {pattern} in {RESULTS} — run the analyze_* step first")
    return files[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--multiseed", default=None, help="multiseed verdict JSON (default: latest ms5)")
    ap.add_argument("--unfiltered", default=str(RESULTS / "unfiltered_sft_verdict.json"))
    ap.add_argument("--out", default=str(PLOTS / "headline_endpoint_accuracy.png"))
    args = ap.parse_args()

    ms_path = args.multiseed or _latest("multiseed_verdict_*_k*.json")
    ms = json.loads(Path(ms_path).read_text())["verdict"]
    unf = json.loads(Path(args.unfiltered).read_text())

    base = ms["base_accuracy"]
    ca = ms["condition_accuracy"]

    # (label, mean, se, per-seed values, color) in display order.
    HL = "#2a9d8f"    # reward laundering (the headline condition) — highlighted
    NEU = "#5a6b7b"   # other conditions
    bars = [
        ("Reward\nlaundering\n(B)", ca["b"]["mean"], ca["b"]["se"],
         list(ca["b"]["per_seed"].values()), HL),
        ("Direct\nsubset-sum\nreward (C)", ca["c"]["mean"], ca["c"]["se"],
         list(ca["c"]["per_seed"].values()), NEU),
        ("On-policy\nunfiltered\nSFT", unf["variants"]["on-policy unfiltered"]["pass1_mean"],
         unf["variants"]["on-policy unfiltered"]["pass1_se"],
         list(unf["variants"]["on-policy unfiltered"]["pass1_per_seed"].values()), NEU),
        ("Base-rollout\nunfiltered\nSFT", unf["variants"]["base-rollout unfiltered"]["pass1_mean"],
         unf["variants"]["base-rollout unfiltered"]["pass1_se"],
         list(unf["variants"]["base-rollout unfiltered"]["pass1_per_seed"].values()), NEU),
        ("RL on\nmain task\n(A)", ca["a"]["mean"], ca["a"]["se"],
         list(ca["a"]["per_seed"].values()), NEU),
    ]

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    xs = range(len(bars))
    for x, (label, mean, se, seeds, color) in zip(xs, bars):
        # faint per-seed dots (jittered)
        for j, v in enumerate(seeds):
            jitter = (j - (len(seeds) - 1) / 2) * 0.045
            ax.scatter(x + jitter, v, s=26, color=color, alpha=0.28, zorder=2, linewidths=0)
        # mean marker + SE bar
        ax.errorbar(x, mean, yerr=se, fmt="o", ms=13, color=color, capsize=5, elinewidth=1.5,
                    zorder=4, markeredgecolor="black", markeredgewidth=0.6)
        ax.text(x, mean + se + 0.028, f"{mean:.2f}", ha="center", fontsize=11, fontweight="bold",
                color=color)

    ax.axhline(base, ls="--", color="black", lw=1.3, alpha=0.7, zorder=1)
    ax.text(len(bars) - 0.5, base + 0.008, f"base model (no training) = {base:.2f}",
            ha="right", va="bottom", fontsize=9.5, color="black", alpha=0.8)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([b[0] for b in bars], fontsize=9.5)
    ax.set_ylabel("subset-sum accuracy (neutral prompt)", fontsize=11)
    ax.set_ylim(0, max(b[1] + b[2] for b in bars) * 1.25)
    n_sft = unf["variants"]["on-policy unfiltered"]["n_seeds"]
    ax.set_title(
        "A model improves at a never-rewarded task by choosing when to earn its reward\n"
        f"endpoint subset-sum capability by condition "
        f"(GRPO n={len(ca['b']['per_seed'])}, SFT n={n_sft}; dot = mean, bar = SE, faint = per seed)",
        fontsize=11.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    PLOTS.mkdir(exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    print(f"  reward laundering (B) {ca['b']['mean']:.3f}  |  direct (C) {ca['c']['mean']:.3f}  |  "
          f"RL on main task (A) {ca['a']['mean']:.3f}  |  base {base:.3f}")
    print(f"  on-policy SFT {unf['variants']['on-policy unfiltered']['pass1_mean']:.3f}  |  "
          f"base-rollout SFT {unf['variants']['base-rollout unfiltered']['pass1_mean']:.3f}")


if __name__ == "__main__":
    main()
