"""Do alternate label tokens recover the fine-tune strength ranking? For each token set (A/B, yes/no,
good/bad): seed-average the final-step held-out delta per pair, refit a mean-zero per-feature strength,
and correlate (per-feature and per-pair) with the committed fine-tune reference (ft_delta), on the same
pairs. Also winner (sign) agreement on well-separated pairs (|ft_delta| >= 4)."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RAW = Path(__file__).resolve().parent / "results" / "label_tokens" / "label_tokens_raw.jsonl"


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
    return {f: float(s[idx[f]]) for f in feats}


def main():
    rows = [json.loads(l) for l in open(RAW)]
    by = defaultdict(lambda: defaultdict(dict))   # tokset -> pair -> {seed: final delta}
    meta = {}
    for r in rows:
        tok = f"{r['tok_a'].strip()}/{r['tok_b'].strip()}"
        by[tok][r["pair"]][r["seed"]] = r["curve"][-1]["heldout_raw"]
        meta[r["pair"]] = r
    for tok in sorted(by):
        pairs = [p for p in by[tok] if len(by[tok][p]) >= 1]
        d_tok = [float(np.mean(list(by[tok][p].values()))) for p in pairs]
        d_ref = [meta[p]["ft_delta_raw"] for p in pairs]
        fg = [(meta[p]["f1"], meta[p]["f2"]) for p in pairs]
        s_tok, s_ref = refit(fg, d_tok), refit(fg, d_ref)
        feats = sorted(s_tok)
        pf = spearman([s_tok[f] for f in feats], [s_ref[f] for f in feats])
        pp = spearman(d_tok, d_ref)
        ws = [(t, r) for t, r in zip(d_tok, d_ref) if abs(r) >= 4]
        win = np.mean([np.sign(t) == np.sign(r) for t, r in ws]) if ws else float("nan")
        print(f"[{tok:9s}] {len(pairs):3d} pairs, {len(feats)} feats | per-feature rho {pf:+.3f} | "
              f"per-pair rho {pp:+.3f} | well-sep winner agree {win*100:.0f}% ({len(ws)})")


if __name__ == "__main__":
    main()
