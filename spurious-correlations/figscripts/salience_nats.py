#!/usr/bin/env python3
"""salience_nats_before_after.png: two panels, probe (left) and fine-tune (right).
Both: x = commitment toward the boosted (losing) feature at ordinary salience, y = at max salience.
Points on the diagonal = salience didn't move it; points above y=0 that started below = flipped."""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = _ROOT + "/inoc_data"
FT = _ROOT + "/knockout/results/salience_ft_dose_full/results.json"

# ---- probe (existing): present m=1.0 vs loud m=1.5, orthogonal sweep ----
def ts_at(r, m):
    for pt in r["curves"]["swept_orth"]:
        if pt["m"] == m and pt.get("toward_swept") is not None:
            return pt["toward_swept"]
    return None
pr_x, pr_y, pr_flip = [], [], []
for r in (json.loads(l) for l in open(f"{D}/salience_flip_raw_v2.jsonl") if l.strip()):
    if r.get("below_floor_held"):
        continue
    b, a = ts_at(r, 1.0), ts_at(r, 1.5)
    if b is None or a is None:
        continue
    fm = r.get("flip_m_swept_orth")
    pr_x.append(b); pr_y.append(a); pr_flip.append(fm is not None and fm <= 1.5)
pr_x, pr_y, pr_flip = np.array(pr_x), np.array(pr_y), np.array(pr_flip)

# ---- fine-tune: commitment toward f at ordinary present (dose level 1) vs max level ----
d = json.load(open(FT))
ft_x, ft_y, ft_flip = [], [], []
for pp in d["per_pair"].values():
    c = pp["commitment_toward_f_by_level_nats"]
    levels = pp["dose_levels_ascending"]
    if len(levels) < 2:
        continue
    present = c.get(str(levels[1]))
    mx = c.get(str(levels[-1]))
    if present is None or mx is None:
        continue
    ft_x.append(present); ft_y.append(mx); ft_flip.append(bool(pp["flips_to_f_at_max"]))
ft_x, ft_y, ft_flip = np.array(ft_x), np.array(ft_y), np.array(ft_flip)

lim = max(np.abs(np.concatenate([pr_x, pr_y, ft_x, ft_y]))) * 1.02
CN, CF = "#4477aa", "#cc3311"

fig, axes = plt.subplots(1, 2, figsize=(15, 7.6), sharex=True, sharey=True)
for ax, (x, y, fl, title) in zip(axes, [
        (ft_x, ft_y, ft_flip, "Fine-tune"),
        (pr_x, pr_y, pr_flip, "Probe")]):
    ax.plot([-lim, lim], [-lim, lim], color="#999", lw=1.0, ls="--", label="no change (intensity does nothing)")
    ax.axhline(0, color="#333", lw=0.8); ax.axvline(0, color="#333", lw=0.8)
    ax.scatter(x[~fl], y[~fl], s=16, color=CN, alpha=0.6, label=f"stays with the same feature ({(~fl).sum()})")
    ax.scatter(x[fl], y[fl], s=26, color=CF, alpha=0.85, label=f"flips onto the boosted feature ({fl.sum()})")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("commitment at ordinary intensity (nats)")
    move = np.abs(y - x).mean()
    ax.set_title(f"{title}  (mean move {move:.1f} nats, {fl.sum()}/{len(x)} flip)", fontsize=12)
    ax.legend(loc="upper left", fontsize=8.5); ax.grid(alpha=0.2)
axes[0].set_ylabel("commitment at loudest / max intensity (nats)")
fig.suptitle("Intensity moves the fine-tune (and flips it near ties), but barely moves the probe", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(_ROOT + "/figs/salience_nats_before_after.png",
            dpi=140, bbox_inches="tight")
print(f"salience nats 2-panel: probe move {np.abs(pr_y-pr_x).mean():.2f} ({pr_flip.sum()}/{len(pr_x)}); "
      f"ft move {np.abs(ft_y-ft_x).mean():.2f} ({ft_flip.sum()}/{len(ft_x)})")
