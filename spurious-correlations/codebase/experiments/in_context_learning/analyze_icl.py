"""Recompute the in-context-learning numbers cited in the writeup, offline (CPU, no GPU/keys):
the per-pair Spearman of the k-shot in-context readout against the fine-tune strength gap
s(f1) - s(f2), and how much smaller the in-context readouts are in magnitude. Reads the shipped
icl_diagonal.jsonl and the fine-tune strengths in complete_graph/refit_v2.json.

Usage:  PYTHONPATH=$PWD/lib python experiments/in_context_learning/analyze_icl.py
"""
import json
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

import config

rows = [json.loads(l) for l in open(config.RESULTS_DIR / "icl_diagonal.jsonl") if l.strip()]
s_ft = json.load(open(config.RESULTS_DIR / "complete_graph" / "refit_v2.json"))["arms"]["without_excluded"]["ft"]["s"]


def split(pair):
    f1 = next(f for f in s_ft if pair.startswith(f + "__"))
    return f1, pair[len(f1) + 2:]


def readout_by_pair(kfilter=None):
    d = defaultdict(list)
    for r in rows:
        if kfilter is not None and r["k"] != kfilter:
            continue
        d[r["pair"]].append(r["heldout"]["raw"])
    return {p: float(np.mean(v)) for p, v in d.items()}


for k in [None, 8, 16]:
    rb = readout_by_pair(k)
    xs, ys = [], []
    for p, v in rb.items():
        f1, f2 = split(p)
        if f1 in s_ft and f2 in s_ft:
            xs.append(s_ft[f1] - s_ft[f2])
            ys.append(v)
    xs, ys = np.array(xs), np.array(ys)
    rho = spearmanr(xs, ys).statistic
    ratio = np.mean(np.abs(xs)) / np.mean(np.abs(ys))
    label = "pooled" if k is None else f"k={k}"
    print(f"[{label:7}] n_pairs={len(xs):3d}  Spearman(icl readout, ft gap) = {rho:+.3f}"
          f"   magnitude {ratio:.1f}x smaller than fine-tune")
