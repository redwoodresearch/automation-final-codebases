#!/usr/bin/env python3
"""signature_norm_vs_strength_scatter.png: mean-activation signature norm vs strength,
two panels — fine-tune (left) and mean-pooled probe (right)."""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(__file__))
from feature_names import disp
from scipy.stats import spearmanr
D = _ROOT + "/inoc_data"
norms = json.load(open(f"{D}/signature_norms_v2.json"))
ref = json.load(open(f"{D}/refit_v2.json"))["arms"]["without_excluded"]
refm = json.load(open(f"{D}/refit_v2_meanpooled.json"))["arms"]["without_excluded"]
mr = [json.loads(l) for l in open(f"{D}/merged_readouts_v2.jsonl") if l.strip()]
ftype = {}
for r in mr:
    ftype[r["f1"]] = r["f1_type"]; ftype[r["f2"]] = r["f2_type"]
nv = norms["norms"]["mean"]["20"]
TYPE_COLORS = {"style_voice":"#1f77b4","discourse_structure":"#17becf","surface_mechanical":"#7f7f7f","content_behavior":"#d62728","triggers":"#ff7f0e"}
TYPE_LABELS = {"style_voice":"style / voice","discourse_structure":"discourse structure","surface_mechanical":"surface mechanical","content_behavior":"content / behavior","triggers":"trigger"}
READ_LOC = {"trailing_signature","ends_with_string"}

def _place_labels(ax, pts, names, colors, markers):
    """Deterministic greedy placement: each label goes at the nearest ring position whose text box
    overlaps no marker and no already-placed label. Guarantees no word-point / word-word overlap,
    minimizes distance. Draws a thin leader only when the label had to move off its point."""
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    # marker obstacle boxes (in display coords), slightly padded
    trans = ax.transData
    mrad = 7.0  # marker half-size in px (points are s~62-70 -> ~7-8px radius)
    mboxes = []
    for (x, y) in pts:
        cx, cy = trans.transform((x, y))
        mboxes.append((cx - mrad, cy - mrad, cx + mrad, cy + mrad))
    placed = []  # display-coord boxes of placed labels

    def overlaps(b, boxes, pad=1.0):
        x0, y0, x1, y1 = b
        for (a0, a1, a2, a3) in boxes:
            if x0 < a2 + pad and x1 > a0 - pad and y0 < a3 + pad and y1 > a1 - pad:
                return True
        return False

    order = sorted(range(len(pts)), key=lambda i: pts[i][1], reverse=True)
    import math
    for i in order:
        x, y = pts[i]
        t = ax.text(x, y, names[i], fontsize=6.6, alpha=0.95, color=colors[i], va="center", ha="left")
        bb = t.get_window_extent(renderer=rend)
        w, h = bb.width, bb.height
        cx, cy = trans.transform((x, y))
        best = None
        for r in range(6, 90, 3):                       # grow the ring (px) until a free slot
            for ang in range(0, 360, 20):
                a = math.radians(ang)
                lx, ly = cx + r * math.cos(a), cy + r * math.sin(a)
                # anchor: put box so its near edge sits at (lx,ly)
                bx0 = lx if math.cos(a) >= 0 else lx - w
                by0 = ly - h / 2
                box = (bx0, by0, bx0 + w, by0 + h)
                if not overlaps(box, mboxes) and not overlaps(box, placed):
                    best = (lx, ly, box, math.cos(a) >= 0, r)
                    break
            if best:
                break
        if best is None:
            best = (cx + 8, cy, (cx + 8, cy - h / 2, cx + 8 + w, cy + h / 2), True, 8)
        lx, ly, box, right, r = best
        dx, dy = inv.transform((lx, ly))
        t.set_position((dx, dy)); t.set_ha("left" if right else "right")
        placed.append(box)
        if r > 12:  # leader line only when it moved appreciably
            ax.annotate("", xy=(x, y), xytext=(dx, dy), textcoords="data",
                        arrowprops=dict(arrowstyle="-", color="0.75", lw=0.35), zorder=1)

fig, axes = plt.subplots(1, 2, figsize=(17, 9.5), sharex=True)
for ax, (s, title) in zip(axes, [(ref["ft"]["s"], "fine-tune"), (refm["probe_mean"]["s"], "mean-pooled probe")]):
    feats = sorted(s, key=lambda f: s[f])
    xs = np.array([nv[f] for f in feats]); ys = np.array([s[f] for f in feats])
    rs = spearmanr(xs, ys).statistic
    for f in feats:
        m = "D" if f in READ_LOC else "o"
        ax.scatter(nv[f], s[f], s=70 if m=="D" else 62, marker=m, color=TYPE_COLORS[ftype[f]], edgecolor="k", linewidth=0.5, zorder=3)
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.set_xlim(-42, max(xs) * 1.30)  # room on the right for labels
    ax.set_xlabel("‖mean activation difference‖  (mean-pooled, layer 20)", fontsize=10)
    ax.set_title(f"{title}  (Spearman {rs:+.2f})", fontsize=12)
    _pts = [(nv[f], s[f]) for f in feats]
    _names = [disp(f) + (" ‡" if f in READ_LOC else "") for f in feats]
    _cols = [TYPE_COLORS[ftype[f]] for f in feats]
    _place_labels(ax, _pts, _names, _cols, None)
axes[0].set_ylabel("feature strength s(f)", fontsize=11)
from matplotlib.lines import Line2D
axes[0].legend(handles=[Line2D([0],[0],marker="o",ls="",color=TYPE_COLORS[t],mec="k",label=TYPE_LABELS[t]) for t in TYPE_COLORS], fontsize=9, loc="upper left")
fig.suptitle("Base geometry predicts feature strength", fontsize=14)
fig.tight_layout(rect=(0,0,1,0.96))
fig.savefig(f"{_ROOT}/figs/signature_norm_vs_strength_scatter.png", dpi=150, bbox_inches="tight")
print("signature scatter (fine-tune + mean-pooled)")
