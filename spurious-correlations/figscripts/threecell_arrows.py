#!/usr/bin/env python3
"""threecell_by_label.png + threecell_by_partner.png: per-feature arrows from the 2-cell Δ
(fully confounded) to the 3-cell held-out Δ, both toward the intended (label) feature.
Left plot groups by the label feature (X, the one the third cell is added for);
right plot groups by the partner feature (Y, the shortcut side)."""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(__file__))
from feature_names import disp
from collections import defaultdict
D=_ROOT + ""
runs=[json.loads(l) for l in open(f"{D}/knockout/results/threecell_full/runs.jsonl")]
mr={r['pair']:r for r in (json.loads(l) for l in open(f"{D}/inoc_data/merged_readouts_v2.jsonl")) if r}
agg=defaultdict(list); meta={}
for r in runs:
    if not r['holdout_is_label_present']: continue
    k=(r['pair'],r['holdout_cell']); agg[k].append(r['out_logodds_true'])
    meta[k]=(r['label_is_f2'],r['label_feature'],r['partner_feature'])
pts=[]
for k,vals in agg.items():
    m=mr.get(k[0])
    if not m: continue
    lf2,lab,par=meta[k]
    x=-m['ft_delta'] if lf2 else m['ft_delta']
    pts.append((lab,par,x,float(np.mean(vals))))

UP,DOWN="#188038","#c5221f"
def arrow_plot(group_idx, fname, title, xlabel_note):
    by=defaultdict(lambda:[[],[]])
    for p in pts:
        by[p[group_idx]][0].append(p[2]); by[p[group_idx]][1].append(p[3])
    feats=sorted(by, key=lambda f: np.mean(by[f][0]))
    y=np.arange(len(feats))
    fig,ax=plt.subplots(figsize=(9,0.32*len(feats)+2))
    allv=[v for f in feats for v in (np.mean(by[f][0]),np.mean(by[f][1]))]
    ax.set_xlim(min(allv)-1.5, max(allv)+1.5)
    for i,f in enumerate(feats):
        b=np.mean(by[f][0]); a=np.mean(by[f][1]); c=UP if a>=b else DOWN
        ax.annotate("",xy=(a,i),xytext=(b,i),
            arrowprops=dict(arrowstyle="-|>,head_width=0.3,head_length=0.6",color=c,lw=1.9,shrinkA=0,shrinkB=0))
    ax.set_yticks(y); ax.set_yticklabels([f"{disp(f)} ({len(by[f][0])})" for f in feats],fontsize=8)
    ax.set_ylim(-1,len(feats)); ax.axvline(0,color="#444",lw=0.8,ls="--")
    ax.set_xlabel(f"nats toward the intended feature  (arrow: 2-cell → 3-cell{xlabel_note})")
    ax.set_title(title,fontsize=12.5); ax.grid(axis="x",alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{D}/figs/{fname}",dpi=140,bbox_inches="tight"); plt.close(fig)
    print(fname)
arrow_plot(0,"threecell_by_label.png","Third-cell rescue by intended feature","")
arrow_plot(1,"threecell_by_partner.png","Third-cell rescue by partner feature","")
