#!/usr/bin/env python3
"""additive_law_ft_mean.png: 3 panels.
(1) fine-tune additive law (pred vs observed Δ), (2) mean-pooled additive law,
(3) scatter comparing fine-tune Δ vs mean-pooled probe Δ per pair."""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

D = _ROOT + "/inoc_data"
ref = json.load(open(f"{D}/refit_v2.json"))["arms"]["without_excluded"]
refm = json.load(open(f"{D}/refit_v2_meanpooled.json"))["arms"]["without_excluded"]
mr = [json.loads(l) for l in open(f"{D}/merged_readouts_v2.jsonl") if l.strip()]
ftype = {}
for r in mr:
    ftype[r["f1"]] = r["f1_type"]; ftype[r["f2"]] = r["f2_type"]
TYPE_COLORS = {"style_voice": "#1f77b4", "discourse_structure": "#17becf",
               "surface_mechanical": "#7f7f7f", "content_behavior": "#d62728", "triggers": "#ff7f0e"}

def obs_pred(arm):
    lp, rs = arm["loo_pred"], arm["residuals"]
    ks = [k for k in arm["pairs"] if k in lp and k in rs]
    return np.array([lp[k] for k in ks]), np.array([lp[k]+rs[k] for k in ks]), arm["loo_r2"]

fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
for ax, (arm, title, col) in zip(axes[:2], [
        (ref["ft"], "Fine-tune", "#e07b00"),
        (refm["probe_mean"], "Mean-pooled probe", "#7e3ff2")]):
    pred, obs, r2 = obs_pred(arm)
    lim = max(np.abs(np.concatenate([pred, obs])))*1.03
    ax.plot([-lim, lim], [-lim, lim], color="#999", ls="--", lw=1.0)
    ax.axhline(0, color="#ccc", lw=0.6); ax.axvline(0, color="#ccc", lw=0.6)
    ax.scatter(pred, obs, s=14, color=col, alpha=0.55)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("predicted  s(f) − s(g)"); ax.set_ylabel("observed Δ(f, g)")
    ax.set_title(f"{title}  (R² = {r2:.2f})", fontsize=12); ax.grid(alpha=0.2)

# panel 3: fine-tune Δ vs mean-pooled probe Δ, per pair, BOTH orientations
# Δ(f2,f1) = −Δ(f1,f2) (the metric is anti-symmetric), so plot each pair and its mirror
# through the origin — orientation is only a sign convention, so this is the honest view.
pairs = [r for r in mr if not r.get("excluded_from_primary_fit")
         and r.get("ft_delta") is not None and r.get("probe_l21_meanpooled") is not None]
x0 = np.array([r["ft_delta"] for r in pairs]); y0 = np.array([r["probe_l21_meanpooled"] for r in pairs])
x = np.concatenate([x0, -x0]); y = np.concatenate([y0, -y0])
rp = pearsonr(x, y)[0]
ax = axes[2]
lim = max(np.abs(np.concatenate([x, y])))*1.03
ax.plot([-lim, lim], [-lim, lim], color="#999", ls="--", lw=1.0)
ax.axhline(0, color="#ccc", lw=0.6); ax.axvline(0, color="#ccc", lw=0.6)
ax.scatter(x, y, s=10, color="#7e3ff2", alpha=0.35)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("fine-tune Δ(f, g)"); ax.set_ylabel("mean-pooled probe Δ(f, g)")
ax.set_title(f"Fine-tune vs mean-pooled, both orientations  (r = {rp:.2f})", fontsize=12); ax.grid(alpha=0.2)

fig.suptitle("Differences in feature strength", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(_ROOT + "/figs/additive_law_ft_mean.png",
            dpi=150, bbox_inches="tight")
print(f"additive_law_ft_mean.png (3 panels); ft-vs-mean both-orientations pearson {rp:.2f}")
