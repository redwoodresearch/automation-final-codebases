"""Figure 2 — reward-channel leakage and subset-sum capability over training, by learning-rate schedule.

The reward-laundering coupling ("answer the addition correctly only if you solved the subset sum") can
erode under continued training: the model starts earning the reward WITHOUT having solved the subset
sum. We measure this with reward-channel leakage:

    leakage = P(addition answer rewarded | subset sum actually wrong)

Top panel: leakage vs RL step, one series per LR schedule. Keeping the learning rate "live" (constant
1e-4, or annealed only to a 3e-5 floor) drives leakage up sharply; the annealed-to-zero schedule the
headline seeds used stays low. Bottom panel: subset-sum capability (neutral-prompt accuracy) at every
checkpoint — despite the leakage, the capability is built early and mostly survives.

Reads only committed condition-B eval trajectories (results/rl_pilot_b_*_evals.jsonl) via
lib.decoupling_analysis — no sampling, free to re-run.

Usage: .venv/bin/python plot_leakage_capability.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lib.decoupling_analysis import (
    BASE_NEUTRAL, CONSTLR_RUNS, EROSION_RUNS, MATRIX_RUNS, THETA_PRIMARY, load_evals,
)

ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "plots"

ANNEALED_COLOR = "#4c72b0"
FLOOR_COLOR = "#dd8452"
CONSTANT_COLOR = "#c44e52"


def _series(evals, key):
    return [(e["step"], e[key]) for e in evals if e.get(key) is not None]


def _annealed_mean(key):
    """Across-seed mean +/- SE of `key` at each step, over the 5 annealed matrix seeds."""
    by_step: dict[int, list[float]] = {}
    for run in MATRIX_RUNS:
        for s, v in _series(load_evals(run), key):
            by_step.setdefault(s, []).append(v)
    steps = sorted(by_step)
    mean = np.array([np.mean(by_step[s]) for s in steps])
    se = np.array([np.std(by_step[s], ddof=1) / len(by_step[s]) ** 0.5 if len(by_step[s]) > 1 else 0.0
                   for s in steps])
    return np.array(steps), mean, se


def _plot(ax, key):
    # Annealed: across-seed mean with an SE band.
    steps, mean, se = _annealed_mean(key)
    ax.plot(steps, mean, "o-", color=ANNEALED_COLOR, lw=2, ms=4,
            label="annealed LR->0 (headline seeds, n=5)")
    ax.fill_between(steps, mean - se, mean + se, color=ANNEALED_COLOR, alpha=0.15)

    # 3e-5 live floor: the erosion runs (condition B only).
    floor_runs = [r for r in EROSION_RUNS if not r.is_control]
    for i, run in enumerate(floor_runs):
        xs, ys = zip(*_series(load_evals(run), key)) if _series(load_evals(run), key) else ((), ())
        ax.plot(xs, ys, "s--", color=FLOOR_COLOR, lw=1.6, ms=4, alpha=0.9,
                label="3e-5 live floor" if i == 0 else None)

    # Constant LR 1e-4.
    for run in CONSTLR_RUNS:
        xs, ys = zip(*_series(load_evals(run), key))
        ax.plot(xs, ys, "^-", color=CONSTANT_COLOR, lw=2, ms=5, label="constant LR 1e-4")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PLOTS / "fig_leakage_capability.png"))
    args = ap.parse_args()

    fig, (ax_leak, ax_cap) = plt.subplots(2, 1, figsize=(8.2, 8.0), sharex=True)

    _plot(ax_leak, "leakage")
    ax_leak.axhline(THETA_PRIMARY, ls=":", color="black", lw=1, alpha=0.6)
    ax_leak.text(1, THETA_PRIMARY + 0.01, f"decoupling threshold (theta={THETA_PRIMARY})",
                 fontsize=8, va="bottom", alpha=0.7)
    ax_leak.set_ylabel("reward-channel leakage\nP(reward | subset wrong)", fontsize=10)
    ax_leak.set_ylim(0, 1.02)
    ax_leak.set_title("Reward-channel honesty erodes when the learning rate stays live", fontsize=11)
    ax_leak.legend(fontsize=9, loc="upper left")
    ax_leak.grid(alpha=0.3)

    _plot(ax_cap, "neutral_subset_accuracy")
    ax_cap.axhline(BASE_NEUTRAL, ls=":", color="black", lw=1, alpha=0.6)
    ax_cap.text(1, BASE_NEUTRAL + 0.01, f"base model ({BASE_NEUTRAL})", fontsize=8, va="bottom",
                alpha=0.7)
    ax_cap.set_ylabel("subset-sum accuracy\n(neutral prompt)", fontsize=10)
    ax_cap.set_ylim(0, 1.0)
    ax_cap.set_xlabel("RL step", fontsize=10)
    ax_cap.set_title("...but the capability it built is largely retained", fontsize=11)
    ax_cap.grid(alpha=0.3)

    fig.suptitle("Reward-channel leakage vs. subset-sum capability over training",
                 fontsize=12.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    PLOTS.mkdir(exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
