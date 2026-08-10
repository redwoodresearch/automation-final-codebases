#!/usr/bin/env python3
"""Re-derive the generative / safety / DPO-vs-SFT inheritance-margin numbers from committed data only
(CPU, no GPU / Modal / API). Feeds the blog's "Generative behavior", "Safety behaviors", and "DPO"
appendix claims.

The metric is the SIGNED INHERITANCE MARGIN, generative_readout.signed_inheritance(q, p) = (2q-1)*sign(2p-1):
q = the trained behavior's judged follows-f1 rate on the held-out (off-diagonal) cells; p = the label
fine-tune's follows-f1 rate for that pair (the classifier winner). +1 = the behavior always follows the
feature the label classifier picked, 0 = no relation, negative = it follows the other feature.

Pooling (identical to the committed 9-behavior generative headline and to this project's report): for each
(behavior, pair) average q over the run seeds first, take the signed margin against the label winner, then
average the per-pair margins within a behavior. The headline numbers are reported over the WELL-SEPARATED
pairs (|label follows-f1 - 0.5| > 0.4) — the stratum where the label commits hard.

Three products (all read-only, from results/):
  behavior_sft_results.jsonl        -> generative inheritance margin, 9 behaviors  (blog +0.80)
  behavior_sft_safety_results.jsonl -> refusal / caution / disclosure             (blog +0.83/+0.60/+0.56)
  dpo_vs_sft_results.jsonl          -> DPO vs SFT arm                              (blog DPO +0.80 / SFT +0.81)

Sources (committed, read-only):
  results/behavior_sft_results.jsonl         generative SFT read-out (run_behavior_sft.py)
  results/behavior_sft_safety_results.jsonl  the 3 safety behaviors (run_behavior_sft.py --safety)
  results/dpo_vs_sft_results.jsonl           matched DPO vs SFT arms (run_dpo_readout.py)
  results/finetune_expanded_values.jsonl     the label fine-tune reference (p, the classifier winner)
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

# codebase root = three levels up from experiments/behavior_generation/, exactly like
# complete_graph/analyze_near_tie_collapse_commit.py; add lib/ to the path so the shared spine imports.
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "lib"))
R = os.path.join(BASE, "results")

import generative_readout as gr  # noqa: E402

LABEL_REF_PATH = os.path.join(R, "finetune_expanded_values.jsonl")
WELLSEP = 0.4  # |label follows-f1 - 0.5| > WELLSEP  ==  the label commits hard

# The 3 safety behaviors -> the blog's names (refusal / caution / disclosure).
SAFETY_NAME = {"refuse_vs_comply": "refusal", "cautious_vs_reckless": "caution",
               "disclose_vs_conceal": "disclosure"}


def _load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return [r for r in rows if r.get("seed") != -1]  # seed=-1 is the pre-SFT baseline, not a trained read


def _q(row):
    """The behavior's follows-f1 rate q for one row, pooled over the two held cells (equal weight):
    held_f1pos contributes P(pole+), held_f1neg contributes 1 - P(pole+). None if a cell is undefined."""
    fp = row["cells"]["held_f1pos"]["plus_share"]
    fn = row["cells"]["held_f1neg"]["plus_share"]
    if fp is None or fn is None:
        return None
    return 0.5 * (fp + (1.0 - fn))


def pooled_margins(rows, group_key, label_for):
    """Pool exactly as the report/headline do: per (group, pair) average q over seeds, take the signed
    margin against the label winner p, then average per-pair margins within a group. Returns
    (pooled_by_group, per_seed_by_group, n_pairs_by_group) restricted to the well-separated pairs."""
    q_by = defaultdict(lambda: defaultdict(list))     # (group, pair) -> seed -> [q]
    per_seed = defaultdict(lambda: defaultdict(list))  # group -> seed -> [margin]
    for r in rows:
        p = label_for(r["pair"])
        if p is None:
            continue
        pmean = 0.5 * (p[0] + p[1])
        if abs(pmean - 0.5) <= WELLSEP:  # well-separated stratum only
            continue
        q = _q(r)
        if q is None:
            continue
        q_by[(r[group_key], r["pair"])][r["seed"]].append(q)
        per_seed[r[group_key]][r["seed"]].append(gr.signed_inheritance(q, pmean))

    pooled, npairs = {}, {}
    grp = defaultdict(list)
    for (g, pair), by_seed in q_by.items():
        p = label_for(pair)
        qbar = float(np.mean([np.mean(v) for v in by_seed.values()]))  # mean-over-seeds of the per-seed mean-q
        grp[g].append(gr.signed_inheritance(qbar, 0.5 * (p[0] + p[1])))
    for g, ms in grp.items():
        pooled[g] = float(np.mean(ms))
        npairs[g] = len(ms)
    per_seed_mean = {g: {s: float(np.mean(m)) for s, m in bys.items()} for g, bys in per_seed.items()}
    return pooled, per_seed_mean, npairs


def _fmt_seeds(d):
    return "  ".join(f"seed{s}={d[s]:+.3f}" for s in sorted(d))


def main():
    ref = gr.load_label_reference(LABEL_REF_PATH)

    def label_for(pair):
        return gr.label_cells(pair, ref)

    print("=" * 78)
    print("Inheritance margins from committed data (CPU-only), well-separated pairs, pooled over seeds")
    print("  signed_inheritance(q, p) = (2q-1)*sign(2p-1)")
    print("=" * 78)

    # 1) generative headline (9 behaviors) -> mean over all behaviors' per-pair margins
    gen_rows = _load(os.path.join(R, "behavior_sft_results.jsonl"))
    gp, gps, gnp = pooled_margins(gen_rows, "behavior", label_for)
    all_margins = [m for b in gp for m in [gp[b]]]  # per-behavior pooled means
    gen_headline = float(np.mean(list(gp.values())))
    # pooled directly over all (behavior,pair) cells (the headline number):
    cell_margins = []
    q_by = defaultdict(lambda: defaultdict(list))
    for r in gen_rows:
        p = label_for(r["pair"])
        if p is None or abs(0.5 * (p[0] + p[1]) - 0.5) <= WELLSEP:
            continue
        q = _q(r)
        if q is not None:
            q_by[(r["behavior"], r["pair"])][r["seed"]].append(q)
    for (b, pair), bys in q_by.items():
        p = label_for(pair)
        qbar = float(np.mean([np.mean(v) for v in bys.values()]))
        cell_margins.append(gr.signed_inheritance(qbar, 0.5 * (p[0] + p[1])))
    print(f"\n[1] GENERATIVE behavior (behavior_sft_results.jsonl): "
          f"{np.mean(cell_margins):+.3f}   (blog +0.80)")
    print(f"    {len(cell_margins)} well-separated (behavior,pair) cells, {len(gp)} behaviors")

    # 2) safety behaviors -> refusal / caution / disclosure
    saf_rows = _load(os.path.join(R, "behavior_sft_safety_results.jsonl"))
    sp, sps, snp = pooled_margins(saf_rows, "behavior", label_for)
    print(f"\n[2] SAFETY behaviors (behavior_sft_safety_results.jsonl):   (blog +0.83 / +0.60 / +0.56)")
    for key in ("refuse_vs_comply", "cautious_vs_reckless", "disclose_vs_conceal"):
        if key in sp:
            print(f"    {SAFETY_NAME[key]:11s} pooled={sp[key]:+.3f}  (n_pairs={snp[key]})   {_fmt_seeds(sps[key])}")

    # 3) DPO vs SFT arms
    dpo_rows = _load(os.path.join(R, "dpo_vs_sft_results.jsonl"))
    dp, dps, dnp = pooled_margins(dpo_rows, "arm", label_for)
    print(f"\n[3] DPO vs SFT (dpo_vs_sft_results.jsonl):   (blog DPO +0.80 / SFT +0.81)")
    for arm in ("dpo", "sft"):
        if arm in dp:
            print(f"    {arm.upper():3s} pooled={dp[arm]:+.3f}  (n_pairs={dnp[arm]})   {_fmt_seeds(dps[arm])}")
    print()


if __name__ == "__main__":
    main()
