"""Does the reward-model scalar value head recover the fine-tune strength ranking on the full 37?
Seed-average the final-step held-out scalar read per pair, refit a mean-zero per-feature strength, and
correlate (per-feature and per-pair) with the committed fine-tune reference (ft_delta); plus winner
agreement on well-separated pairs (|ft_delta| >= 4)."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RAW = Path(__file__).resolve().parent / "results" / "scalar_head" / "scalar_head_raw.jsonl"


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def refit(fg, dl):
    fs = sorted({f for g in fg for f in g}); ix = {f: i for i, f in enumerate(fs)}
    A = np.zeros((len(fg) + 1, len(fs))); y = np.zeros(len(fg) + 1)
    for r, ((a, b), v) in enumerate(zip(fg, dl)):
        A[r, ix[a]] += 1; A[r, ix[b]] -= 1; y[r] = v
    A[-1, :] = 1
    s, *_ = np.linalg.lstsq(A, y, rcond=None)
    return fs, {f: float(s[ix[f]]) for f in fs}


def main():
    rows = [json.loads(l) for l in open(RAW)]
    by = defaultdict(dict); meta = {}
    for r in rows:
        by[r["pair"]][r["seed"]] = r["curve"][-1]["heldout_raw"]  # final step
        meta[r["pair"]] = r
    pairs = list(by)
    d_sc = [float(np.mean(list(by[p].values()))) for p in pairs]
    d_ref = [meta[p]["ft_delta_raw"] for p in pairs]
    fg = [(meta[p]["f1"], meta[p]["f2"]) for p in pairs]
    fs, s_sc = refit(fg, d_sc); _, s_ref = refit(fg, d_ref)
    pf = spearman([s_sc[f] for f in fs], [s_ref[f] for f in fs])
    pp = spearman(d_sc, d_ref)
    ws = [(t, r) for t, r in zip(d_sc, d_ref) if abs(r) >= 4]
    win = np.mean([np.sign(t) == np.sign(r) for t, r in ws]) if ws else float("nan")
    print(f"scalar value head: {len(pairs)} pairs, {len(fs)} features")
    print(f"  per-feature Spearman vs fine-tune strength: {pf:+.3f}")
    print(f"  per-pair Spearman vs fine-tune delta:       {pp:+.3f}")
    print(f"  well-separated winner agreement:            {win*100:.0f}% ({len(ws)})")


if __name__ == "__main__":
    main()
