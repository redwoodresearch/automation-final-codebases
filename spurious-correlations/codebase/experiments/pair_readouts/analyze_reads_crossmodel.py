"""Cross-model reads: does each other model's probe read (decision-token / mean-pooled) recover the
Qwen3-1.7B FINE-TUNE strength ordering? Each model has a different depth, so we pick the layer per
(model, read) that best matches the reference, and also report a fixed ~0.7-relative-depth layer (the
non-cherry-picked value). Reference = fine-tune per-feature strength refit from ft_delta (merged_readouts)
on the same cross-model pair set.

Reads results/reads_expansion/<model>__<read>__xmodel.json ({pair: {layer: raw_mean}}).
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MERGED = os.path.join(ROOT, "..", "inoc_data", "merged_readouts_v2.jsonl")
READS = os.path.join(ROOT, "results", "reads_expansion")


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def refit(pairs_fg, deltas):
    feats = sorted({f for fg in pairs_fg for f in fg}); idx = {f: i for i, f in enumerate(feats)}
    A = np.zeros((len(pairs_fg) + 1, len(feats))); y = np.zeros(len(pairs_fg) + 1)
    for r, ((f1, f2), d) in enumerate(zip(pairs_fg, deltas)):
        A[r, idx[f1]] += 1; A[r, idx[f2]] -= 1; y[r] = d
    A[-1, :] = 1
    s, *_ = np.linalg.lstsq(A, y, rcond=None)
    return feats, {f: float(s[idx[f]]) for f in feats}


def main():
    merged = {json.loads(l)["pair"]: json.loads(l) for l in open(MERGED) if l.strip()}
    files = sorted(glob.glob(os.path.join(READS, "*__xmodel.json")))
    if not files:
        print("no cross-model read files yet"); return
    for path in files:
        base = os.path.basename(path).replace(".json", "")
        model, read, _ = base.split("__")
        data = json.load(open(path))
        pairs = [p for p in data if p in merged]
        fg = [(merged[p]["f1"], merged[p]["f2"]) for p in pairs]
        ft = [merged[p]["ft_delta"] for p in pairs]
        _, s_ft = refit(fg, ft)
        feats = sorted(s_ft)
        layers = sorted({int(L) for p in pairs for L in data[p]}, )
        # per-layer per-feature Spearman vs the 1.7B fine-tune strengths
        rho_by_layer = {}
        for L in layers:
            sl = str(L)
            if not all(sl in data[p] for p in pairs):
                continue
            _, s_rd = refit(fg, [data[p][sl] for p in pairs])
            rho_by_layer[L] = spearman([s_rd[f] for f in feats], [s_ft[f] for f in feats])
        if not rho_by_layer:
            print(f"[{model} / {read}] no complete layer"); continue
        best_L = max(rho_by_layer, key=rho_by_layer.get)
        depth_L = min(rho_by_layer, key=lambda L: abs(L / max(layers) - 0.7))  # ~0.7 relative depth
        print(f"[{model:11s} / {read:14s}] {len(pairs)} pairs | best L{best_L} rho {rho_by_layer[best_L]:+.3f} | "
              f"~0.7-depth L{depth_L} rho {rho_by_layer[depth_L]:+.3f}")


if __name__ == "__main__":
    main()
