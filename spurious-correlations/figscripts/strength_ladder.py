#!/usr/bin/env python3
"""per_feature_strength_ftsorted.png: strength ladder, fine-tune + decision-token probe + mean-pooled probe, sorted by fine-tune."""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(__file__))
from feature_names import disp
D = _ROOT + "/inoc_data"
ref = json.load(open(f"{D}/refit_v2.json"))["arms"]["without_excluded"]
refm = json.load(open(f"{D}/refit_v2_meanpooled.json"))["arms"]["without_excluded"]
s_ft, s_mean = ref["ft"]["s"], refm["probe_mean"]["s"]

# decision-token probe: mean-zero least-squares per-feature strength refit from the base model's
# decision-token reads at layer 21 (results/reads_expansion), the read that tracks the fine-tune
# (Spearman ~0.92; see the activation-differences section).
def _refit(pfg, dl):
    fs = sorted({f for x in pfg for f in x}); ix = {f: i for i, f in enumerate(fs)}
    A = np.zeros((len(pfg), len(fs))); b = np.array(dl, float)
    for r, (a, c) in enumerate(pfg): A[r, ix[a]] += 1; A[r, ix[c]] -= 1
    A = np.vstack([A, np.ones(len(fs))]); b = np.append(b, 0.0)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None); return {f: float(sol[ix[f]]) for f in fs}
_RX = _os.path.dirname(D)
_dt = {}
for _fn in ("all", "new"):
    _dt.update(json.load(open(f"{_RX}/codebase/results/reads_expansion/qwen3_1p7b__decision_token__{_fn}.json")))
s_dt = _refit([tuple(k.split("__")) for k in _dt if "21" in _dt[k]], [_dt[k]["21"] for k in _dt if "21" in _dt[k]])

feats = sorted(s_mean, key=lambda f: s_ft[f]); y = np.arange(len(feats))
FT, DT, MEAN = "#e07b00", "#1f77b4", "#7e3ff2"
fig, ax = plt.subplots(figsize=(9.5, 11))
for i, f in enumerate(feats):
    xs = [s_ft[f], s_dt[f], s_mean[f]]
    ax.plot([min(xs), max(xs)], [i, i], color="#dcdcdc", lw=1.1, zorder=1)
ax.scatter([s_mean[f] for f in feats], y, color=MEAN, s=44, zorder=2, label="mean-pooled probe", edgecolor="white", linewidth=0.3)
ax.scatter([s_ft[f] for f in feats], y, color=FT, s=34, zorder=3, label="fine-tune", edgecolor="white", linewidth=0.3)
ax.scatter([s_dt[f] for f in feats], y, facecolors="none", edgecolors=DT, s=48, linewidth=1.1, zorder=4, label="decision-token probe")
ax.set_yticks(y); ax.set_yticklabels([disp(f) for f in feats], fontsize=8)
ax.set_ylim(-1, len(feats)); ax.axvline(0, color="#333", lw=0.7, alpha=0.5)
ax.set_xlabel("feature strength s(f)  (nats; only gaps and orderings are meaningful)")
ax.set_title("The feature strength ranking", fontsize=13)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=10, frameon=False); ax.grid(axis="x", alpha=0.25)
fig.tight_layout(); fig.savefig(f"{_ROOT}/figs/per_feature_strength_ftsorted.png", dpi=150, bbox_inches="tight")
print("strength ladder")
