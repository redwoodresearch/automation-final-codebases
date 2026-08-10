"""Analyze the expansion reads: how well does each probe read's per-feature strength ordering (and
per-pair commitment) recover the fine-tune reference, over the FULL 37-feature read roster?

For each read file results/reads_expansion/qwen3_1p7b__<read>__all.json ({pair: {layer: raw_mean}}):
  - per-pair Spearman: read's L21 delta vs the fine-tune ft_delta (merged_readouts_v2), on shared pairs
  - per-feature Spearman: refit a mean-zero per-feature strength from the read's per-pair deltas
    (delta_p ~= s[f1_p] - s[f2_p], least squares, mean(s)=0) and correlate with the fine-tune strength
    refit the SAME way on the SAME pairs (apples-to-apples).
This is the number the blog body cites as "reading at the decision token predicts fine-tune strength."
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MERGED = os.path.join(ROOT, "..", "inoc_data", "merged_readouts_v2.jsonl")
READS = os.path.join(ROOT, "results", "reads_expansion")
LAYER = "21"


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def refit_strength(pairs_fg, deltas):
    """Mean-zero least-squares per-feature strength from per-pair deltas: delta ~= s[f1]-s[f2]."""
    feats = sorted({f for fg in pairs_fg for f in fg})
    idx = {f: i for i, f in enumerate(feats)}
    A = np.zeros((len(pairs_fg) + 1, len(feats)))
    y = np.zeros(len(pairs_fg) + 1)
    for r, ((f1, f2), d) in enumerate(zip(pairs_fg, deltas)):
        A[r, idx[f1]] += 1.0
        A[r, idx[f2]] -= 1.0
        y[r] = d
    A[-1, :] = 1.0  # mean-zero constraint
    s, *_ = np.linalg.lstsq(A, y, rcond=None)
    return {f: float(s[idx[f]]) for f in feats}


def main():
    merged = {json.loads(l)["pair"]: json.loads(l) for l in open(MERGED) if l.strip()}
    ftpair = {p: r["ft_delta"] for p, r in merged.items()}
    fg = {p: (r["f1"], r["f2"]) for p, r in merged.items()}

    for path in sorted(glob.glob(os.path.join(READS, "qwen3_1p7b__*__all.json"))):
        read = os.path.basename(path).split("__")[1]
        data = json.load(open(path))
        pairs = [p for p in data if p in ftpair and LAYER in data[p]]
        rd = [data[p][LAYER] for p in pairs]
        ft = [ftpair[p] for p in pairs]
        pfg = [fg[p] for p in pairs]

        pp = spearman(rd, ft)
        s_read = refit_strength(pfg, rd)
        s_ft = refit_strength(pfg, ft)
        feats = sorted(s_read)
        pf = spearman([s_read[f] for f in feats], [s_ft[f] for f in feats])
        nfeat = len(feats)
        print(f"\n[{read}]  {len(pairs)} pairs, {nfeat} features")
        print(f"  per-pair Spearman  (read delta vs fine-tune delta):      {pp:+.3f}")
        print(f"  per-feature Spearman (read strength vs fine-tune strength): {pf:+.3f}")


if __name__ == "__main__":
    main()
