#!/usr/bin/env python3
"""inoculation_naming_before_after.png: per-feature naming effect, three points per row,
the original (neutral) commitment and the named commitment with the instruction placed FIRST vs LAST,
for the fine-tune and the mean-pooled probe. Thin small open-triangle arrows from neutral to each
named endpoint. Sorted by the fine-tune's neutral commitment toward the named feature."""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(__file__))
from feature_names import disp
from matplotlib.lines import Line2D

D = _ROOT + "/inoc_data"
FIRST = _ROOT + "/knockout/results/inoc_first"
ORIG, C_FIRST, C_LAST = "#8a8a8a", "#1f77b4", "#e07b00"

def mean_by_feature(rows, key_neu, key_val):
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        f = r["named_feature"]
        if r.get(key_neu) is None or r.get(key_val) is None:
            continue
        agg[f]["neu"].append(r[key_neu]); agg[f]["val"].append(r[key_val])
    return {f: {k: float(np.mean(v)) for k, v in d.items() if v} for f, d in agg.items()}

# fine-tune: neutral, first-placement and last-placement named commitments (one file)
ft_rows = [json.loads(l) for l in open(f"{FIRST}/inoc_first_records.jsonl") if l.strip()]
ft_first = mean_by_feature(ft_rows, "first_neutral_toward_named", "first_named_delta")
ft_last = mean_by_feature(ft_rows, "first_neutral_toward_named", "appended_named_delta")
ft = {f: {"neu": ft_first[f]["neu"], "first": ft_first[f]["val"],
          "last": ft_last[f]["val"] if f in ft_last else None} for f in ft_first}

# probe (mean-pooled): first = prepended layout, last = committed appended layout
pre_rows = [json.loads(l) for l in open(f"{D}/inoculation_meanpooled_endpoints_prepended_v2.jsonl") if l.strip()]
app_rows = [json.loads(l) for l in open(f"{D}/inoculation_meanpooled_endpoints_v2.jsonl") if l.strip()]
pr_first = mean_by_feature(pre_rows, "probe_mean_neutral_toward_named", "probe_mean_named_toward_named")
pr_last = mean_by_feature(app_rows, "probe_mean_neutral_toward_named", "probe_mean_named_toward_named")
probe = {f: {"neu": pr_first[f]["neu"], "first": pr_first[f]["val"],
             "last": pr_last[f]["val"] if f in pr_last else None} for f in pr_first}

order = sorted(ft, key=lambda f: ft[f]["neu"])
y = np.arange(len(order))
labels = [disp(f) for f in order]

fig, axes = plt.subplots(1, 2, figsize=(13, 11), sharey=True)
for ax, (data, title) in zip(axes, [(ft, "Fine-tune"), (probe, "Probe (mean-pooled)")]):
    vals = []
    for i, f in enumerate(order):
        d = data.get(f)
        if not d:
            continue
        neu, fst, lst = d["neu"], d.get("first"), d.get("last")
        vals += [v for v in (neu, fst, lst) if v is not None]
        for x1, c in ((fst, C_FIRST), (lst, C_LAST)):
            if x1 is not None and abs(x1 - neu) > 1e-6:
                ax.annotate("", xy=(x1, i), xytext=(neu, i),
                            arrowprops=dict(arrowstyle="-|>", mutation_scale=7, lw=0.6,
                                            facecolor="none", edgecolor=c, shrinkA=0, shrinkB=0))
        ax.plot([neu], [i], "o", ms=5, mfc="white", mec=ORIG, mew=1.0, zorder=3)
        if fst is not None:
            ax.plot([fst], [i], "o", ms=4.5, color=C_FIRST, zorder=4)
        if lst is not None:
            ax.plot([lst], [i], "o", ms=4.5, color=C_LAST, zorder=4)
    ax.set_xlim(min(vals) - 1.0, max(vals) + 1.5)
    ax.axvline(0, color="#444", lw=0.8, ls="--")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("commitment toward the named feature (nats)")
    ax.grid(axis="x", alpha=0.25)
axes[0].set_yticks(y); axes[0].set_yticklabels(labels, fontsize=8); axes[0].set_ylim(-1, len(order))
axes[1].legend(handles=[
    Line2D([0], [0], marker="o", color="w", mfc="white", mec=ORIG, mew=1.0, ms=8, label="original (no naming)"),
    Line2D([0], [0], marker="o", color="w", mfc=C_FIRST, ms=8, label="named, instruction first"),
    Line2D([0], [0], marker="o", color="w", mfc=C_LAST, ms=8, label="named, instruction last"),
], loc="lower right", fontsize=9)
fig.suptitle("Naming moves the fine-tune; placement barely matters", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(_ROOT + "/figs/inoculation_naming_before_after.png",
            dpi=130, bbox_inches="tight")
print("inoculation arrows: three points per feature (original / first / last)")
