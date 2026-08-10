#!/usr/bin/env python3
"""qwen36_scale_compare.png: safety-behavior pairs. Single panel, per behavior EIGHT bars: the fine-tune
and the mean-pooled probe, each at 1.7B, 1.7B+naming, 27B, 27B+naming (margin TOWARD THE BEHAVIOR, nats;
negative = the style partner wins). Naming bars = the plain baseline plus that condition's isolated
naming shift. All values are over the same 28 selected safety pairs. Shows that scale and explicit naming
move the fine-tune toward the behavior (naming flips it at 27B) while the probe barely moves under either.

Sources: headline.json (plain 1.7B/27B ft+probe, per pair), inoculation_safety_27b.json / inoculation_lying_27b.json
(ft naming shifts, 1.7B and 27B), probe_naming_safety_27b.json (27B probe naming shift),
inoc_data/inoculation_meanpooled_endpoints_prepended_v2.json (1.7B probe naming shift)."""
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

ROOT = _ROOT + ""
R = ROOT + "/knockout/results/qwen36_27b"
SAFE = {"deception", "harmful_vs_benign", "sycophantic_agreement", "refusal_language", "ai_self_reference"}

# plain 1.7B / 27B margins toward behavior (headline, per pair) ---------------------------------
rows = json.load(open(R + "/headline.json"))["rows"]
sel_pairs = {r["pair"] for r in rows}
plain = defaultdict(lambda: defaultdict(list))
for r in rows:
    b = r["behavior"]
    plain[b]["ft17"].append(-r["ft_delta_17b"]); plain[b]["ft27"].append(-r["ft_delta_27b"])
    if r.get("probe_l21_meanpooled_17b") is not None: plain[b]["pr17"].append(-r["probe_l21_meanpooled_17b"])
    if r.get("probe_meanpooled_27b") is not None:      plain[b]["pr27"].append(-r["probe_meanpooled_27b"])
pm = lambda b, k: float(np.mean(plain[b][k]))

# naming SHIFTS (named - neutral), per behavior --------------------------------------------------
inoc = json.load(open(R + "/inoculation_safety_27b.json"))["per_feature"]
ly = json.load(open(R + "/inoculation_lying_27b.json"))
ft_shift_27, ft_shift_17 = {}, {}
for feat, blk in inoc.items():
    ft_shift_27[feat] = blk["summary"]["shift_first"]          # 27B ft naming shift
    ft_shift_17[feat] = blk["summary"]["b1p7_first_shift"]     # committed 1.7B ft naming shift
m27 = ly["summary"]["mean_27b"]
ft_shift_27["deception"] = m27["named_first_toward_lying"] - m27["neutral_first_toward_lying"]
ft_shift_17["deception"] = float(np.mean([row["b1p7"]["first_shift"] for row in ly["rows"]]))

pn = json.load(open(R + "/probe_naming_safety_27b.json"))["per_behavior"]
pr_shift_27 = {b: pn[b]["named_first"]["shift_matched_mean"] for b in pn}   # 27B probe naming shift

# 1.7B probe naming shift: mean probe_mean_shift over the SAME selected pairs, named = behavior
pre = [json.loads(l) for l in open(ROOT + "/inoc_data/inoculation_meanpooled_endpoints_prepended_v2.jsonl") if l.strip()]
pr_shift_17 = defaultdict(list)
for rec in pre:
    if rec.get("named_feature") in SAFE and rec.get("pair") in sel_pairs:
        pr_shift_17[rec["named_feature"]].append(rec["probe_mean_shift"])
pr_shift_17 = {b: float(np.mean(v)) for b, v in pr_shift_17.items()}

behaviors = sorted(plain, key=lambda b: pm(b, "ft27"), reverse=True)
labels = [disp(b) for b in behaviors]

# eight series: (label, value(behavior), color, hatch)
FT_L, FT_D = "#fdae6b", "#a63603"
PR_L, PR_D = "#bcbddc", "#4a1486"
series = [
    ("fine-tune 1.7B",           lambda b: pm(b, "ft17"),                     FT_L, ""),
    ("fine-tune 1.7B + naming",  lambda b: pm(b, "ft17") + ft_shift_17[b],    FT_L, "////"),
    ("fine-tune 27B",            lambda b: pm(b, "ft27"),                     FT_D, ""),
    ("fine-tune 27B + naming",   lambda b: pm(b, "ft27") + ft_shift_27[b],    FT_D, "////"),
    ("probe 1.7B",               lambda b: pm(b, "pr17"),                     PR_L, ""),
    ("probe 1.7B + naming",      lambda b: pm(b, "pr17") + pr_shift_17[b],    PR_L, "////"),
    ("probe 27B",                lambda b: pm(b, "pr27"),                     PR_D, ""),
    ("probe 27B + naming",       lambda b: pm(b, "pr27") + pr_shift_27[b],    PR_D, "////"),
]

x = np.arange(len(behaviors)); w = 0.10
offs = [-0.44, -0.33, -0.22, -0.11, 0.02, 0.13, 0.24, 0.35]
fig, ax = plt.subplots(figsize=(15.5, 6.2))
for (lab, fn, col, hatch), off in zip(series, offs):
    vals = [fn(b) for b in behaviors]
    ax.bar(x + off, vals, w, color=col, hatch=hatch, label=lab, edgecolor="white", linewidth=0.4)
ax.axhline(0, color="#333", lw=1.0)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=11)
ax.set_ylabel("margin toward the behavior (nats)\n← style wins        behavior wins →", fontsize=11)
ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.13), framealpha=0.95, title="hatched = with explicit naming", title_fontsize=8.5)
fig.suptitle("Safety behaviors at Qwen3.6-27B: scale and naming move the fine-tune toward the behavior; the probe barely moves", fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(ROOT + "/figs/qwen36_scale_compare.png", dpi=140, bbox_inches="tight")
print("wrote 8-bar qwen36_scale_compare.png")
for b in behaviors:
    print(f"  {b:22s} ft17 {pm(b,'ft17'):+.1f} ft17n {pm(b,'ft17')+ft_shift_17[b]:+.1f} ft27 {pm(b,'ft27'):+.1f} ft27n {pm(b,'ft27')+ft_shift_27[b]:+.1f} | pr17 {pm(b,'pr17'):+.1f} pr17n {pm(b,'pr17')+pr_shift_17[b]:+.1f} pr27 {pm(b,'pr27'):+.1f} pr27n {pm(b,'pr27')+pr_shift_27[b]:+.1f}")
