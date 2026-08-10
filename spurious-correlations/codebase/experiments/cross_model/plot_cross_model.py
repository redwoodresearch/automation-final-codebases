"""Figures for the cross-model probe ranking test. Reads results/cross_model/cross_model_analysis.json
and writes to plots/cross_model/:
  - ranking_preserved.png   : per-model s_pr Spearman vs 1.7B (non-language) + roster winner-agreement, CIs.
  - s_pr_scatter.png        : per-feature s_pr (model vs 1.7B) scatter, one panel per model.
  - decisiveness_vs_scale.png : scale-free decisiveness vs params (Qwen scale axis).

  PYTHONPATH=lib .venv/bin/python experiments/cross_model/plot_cross_model.py
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
import cross_model as cm

R = config.results_dir("cross_model")
P = config.plots_dir("cross_model")
A = json.loads((R / "cross_model_analysis.json").read_text())
REF = A["reference"]
# display order: Qwen scale axis, then family, corpus, then base controls
ORDER = ["qwen3_8b", "qwen3_32b", "llama31_8b", "olmo2_7b", "qwen3_8b_base", "qwen3_1p7b_base"]
COLOR = {"Qwen3": "#1f77b4", "Llama-3.1": "#d62728", "OLMo-2": "#2ca02c"}


def models():
    return [k for k in ORDER if k in A["per_model"]]


def plot_ranking():
    ks = models()
    labels = [A["per_model"][k]["label"] for k in ks]
    sp = [A["per_model"][k]["s_pr_spearman_nonlang"]["spearman"] for k in ks]
    lo = [A["per_model"][k]["s_pr_spearman_nonlang"]["ci_lo"] for k in ks]
    hi = [A["per_model"][k]["s_pr_spearman_nonlang"]["ci_hi"] for k in ks]
    wa = [A["per_model"][k]["winner_agreement_roster"]["overall"] for k in ks]
    colors = [COLOR[A["per_model"][k]["family"]] for k in ks]
    x = np.arange(len(ks))
    fig, ax = plt.subplots(figsize=(7, 4))
    err = [np.array(sp) - np.array(lo), np.array(hi) - np.array(sp)]
    ax.bar(x - 0.2, sp, 0.4, yerr=err, capsize=3, color=colors, label="strength-ranking agreement (Spearman)")
    ax.bar(x + 0.2, wa, 0.4, color=colors, alpha=0.45, label="winner agreement (sign)")
    ax.axhline(0.7, ls="--", c="grey", lw=1)
    ax.set_xlim(-0.7, len(ks) + 1.05)  # right margin for the reference-line label, clear of the bars
    ax.text(len(ks) - 0.35, 0.715, "preserved\nbar (0.7)", fontsize=8, color="grey", ha="left", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("agreement with Qwen3-1.7B"); ax.set_ylim(0, 1.0)
    ax.set_title("Strength ranking preserved across models")
    # Legend below the plot (the axis now tops out at the natural 1.0 ceiling for an agreement metric).
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, framealpha=0.9)
    fig.tight_layout(); fig.savefig(P / "ranking_preserved.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote ranking_preserved.png")


def plot_scatter():
    ks = models()
    ref_s = A["per_model"].get(REF, {}).get("s_pr_nonlang")
    if ref_s is None:  # reference s_pr stored under its own per_model entry if analyzed; else recompute
        import probe  # noqa
        ref_probe = json.loads((R / f"probe_{REF}.json").read_text())
        ref_s = {f: v for f, v in cm.refit_s(cm.ladders_by_rule(ref_probe)["final"], A_roster_nl()).items()}
    n = len(ks); cols = min(3, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), squeeze=False)
    for i, k in enumerate(ks):
        ax = axes[i // cols][i % cols]
        m = A["per_model"][k]; ms = m["s_pr_nonlang"]
        feats = sorted(set(ms) & set(ref_s))
        xv = [ref_s[f] for f in feats]; yv = [ms[f] for f in feats]
        ax.scatter(xv, yv, s=18, color=COLOR[m["family"]])
        lim = [min(xv + yv) - 1, max(xv + yv) + 1]
        ax.plot(lim, lim, ls=":", c="grey", lw=1)
        ax.set_title(f"{m['label']}  ρ={m['s_pr_spearman_nonlang']['spearman']:.2f}", fontsize=9)
        ax.set_xlabel("Qwen3-1.7B s", fontsize=8); ax.set_ylabel(f"{m['label']} s", fontsize=8)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Per-feature strength s: each model vs the Qwen3-1.7B reference (rank-aligned)", fontsize=11)
    fig.tight_layout(); fig.savefig(P / "s_pr_scatter.png", dpi=140); plt.close(fig)
    print("wrote s_pr_scatter.png")


def A_roster_nl():
    return json.loads((R / "preregistration.json").read_text())["roster_nonlanguage"]


def plot_decisiveness():
    d = A.get("scale_axis_decisiveness", {})
    if len(d) < 2:
        print("skip decisiveness (need the Qwen scale axis)"); return
    ks = [k for k in cm.QWEN_SCALE_AXIS if k in d]
    x = np.arange(len(ks))  # categorical x to avoid log-tick collisions with 3 points
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(x, [d[k]["median_snr"] for k in ks], "o-", color="#1f77b4")
    for i, k in enumerate(ks):
        ax.annotate(f"{d[k]['median_snr']:.0f}", (i, d[k]["median_snr"]),
                    textcoords="offset points", xytext=(0, 8), fontsize=9, ha="center")
    ax.set_xticks(x); ax.set_xticklabels([f"{A['per_model'][k]['label']}\n({d[k]['params_b']:g}B)" for k in ks], fontsize=8)
    ax.set_ylabel("median signal-to-noise:  |Δ_pr| / seed-SD")
    ax.set_title("Scale does not monotonically sharpen the\nprobe read (exploratory)", fontsize=10)
    ax.set_ylim(0, max(d[k]["median_snr"] for k in ks) * 1.3)
    fig.tight_layout(); fig.savefig(P / "decisiveness_vs_scale.png", dpi=140); plt.close(fig)
    print("wrote decisiveness_vs_scale.png")


if __name__ == "__main__":
    plot_ranking()
    plot_scatter()
    plot_decisiveness()
