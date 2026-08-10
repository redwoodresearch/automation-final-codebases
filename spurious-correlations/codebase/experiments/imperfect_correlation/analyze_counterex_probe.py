"""Probe-vs-fine-tune counterexample comparison: headline curve + crossing counts + gap scaling.

Reads the probe raw (results/counterex_probe/counterex_probe_raw.jsonl) and the fine-tune raw
(results/counterex_ext_raw.jsonl), applies the IDENTICAL aggregation used by derive_counterex_ext.py:
per (N, count) take each pair's mean over seeds, then the median across pairs. Emits, for each N:
  * median prob(follow intended) at each count (probe vs fine-tune side by side),
  * per-pair counterexamples-needed-to-cross-0.5 (first count whose per-pair mean >= 0.5), and its
    Spearman correlation with the strength gap (abs_gap, nats),
and writes results/counterex_probe/summary.json.
"""
from __future__ import annotations

import collections
import json

import numpy as np
from scipy.stats import spearmanr

import config

PROBE = config.RESULTS_DIR / "counterex_probe" / "counterex_probe_raw.jsonl"
FT = config.RESULTS_DIR / "counterex_ext_raw.jsonl"
OUT = config.RESULTS_DIR / "counterex_probe" / "summary.json"


def load_probe():
    by = collections.defaultdict(list)   # (n,count,pair) -> [read_bounded_target per seed]
    gap = {}
    for r in map(json.loads, open(PROBE)):
        by[(r["n"], r["count"], r["pair"])].append(r["read_bounded_target"])
        gap[r["pair"]] = r["abs_gap"]
    return by, gap


def load_ft():
    by = collections.defaultdict(list)
    gap = {}
    for r in map(json.loads, open(FT)):
        fin = r["curve"][-1]
        b = fin["heldout_bounded"]
        bt = b if r["target"] == "f1" else 1 - b
        by[(r["n"], r["count"], r["pair"])].append(bt)
        gap[r["pair"]] = r["abs_gap"]
    return by, gap


def curve(by, ns, counts, pairs):
    out = {}
    for n in ns:
        med = []
        for c in counts:
            vals = [np.mean(by[(n, c, p)]) for p in pairs if (n, c, p) in by]
            med.append(float(np.median(vals)) if vals else float("nan"))
        out[n] = med
    return out


def crossings(by, n, counts, pairs):
    res = {}
    for p in pairs:
        vals = {c: np.mean(by[(n, c, p)]) for c in counts if (n, c, p) in by}
        res[p] = next((c for c in counts if vals.get(c, 0) >= 0.5), None)
    return res


def main():
    pby, pgap = load_probe()
    fby, fgap = load_ft()
    ns = sorted({k[0] for k in pby})
    counts = sorted({k[1] for k in pby})
    pairs = sorted({k[2] for k in pby})

    pcurve = curve(pby, ns, counts, pairs)
    fcurve = curve(fby, ns, counts, pairs)

    print(f"pairs={len(pairs)}  counts={counts}  seeds per (pair,n,count) = "
          f"{len(pby[(ns[0],counts[0],pairs[0])])}")
    print("\nmedian prob(follow intended) across pairs  [PROBE vs FINE-TUNE]")
    for n in ns:
        print(f"\n N={n}   " + " ".join(f"{c:>5}" for c in counts))
        print("  probe  " + " ".join(f"{v:5.2f}" for v in pcurve[n]))
        print("  ft     " + " ".join(f"{v:5.2f}" for v in fcurve[n]))

    summary = {"pairs": pairs, "counts": counts, "ns": ns,
               "probe_curve": pcurve, "ft_curve": fcurve, "crossings": {}}
    print("\ncounterexamples-needed-to-cross-0.5 (per N):")
    for n in ns:
        pc = crossings(pby, n, counts, pairs)
        fc = crossings(fby, n, counts, pairs)
        pf = [v for v in pc.values() if v is not None]
        ff = [v for v in fc.values() if v is not None]
        # gap scaling: correlate crossing (treat never-cross as > max count for ranking) with abs_gap
        maxc = max(counts) * 2
        gaps = [pgap[p] for p in pairs]
        pcross_rank = [pc[p] if pc[p] is not None else maxc for p in pairs]
        rho, pval = spearmanr(gaps, pcross_rank)
        # same gap-scaling correlation for the fine-tune (the ~0.92 cited in the writeup); fgap is the
        # per-pair strength gap on the fine-tune side, fc its counterexamples-to-cross.
        fcross_rank = [fc[p] if fc[p] is not None else maxc for p in pairs]
        ft_rho, ft_pval = spearmanr([fgap[p] for p in pairs], fcross_rank)
        print(f"  N={n}: PROBE median={np.median(pf) if pf else float('nan'):.1f} "
              f"range={min(pf) if pf else '-'}-{max(pf) if pf else '-'} "
              f"never_cross={sum(v is None for v in pc.values())}/{len(pairs)}   |   "
              f"FT median={np.median(ff):.1f} range={min(ff)}-{max(ff)} "
              f"never={sum(v is None for v in fc.values())}   |   "
              f"gap-scaling Spearman: probe rho={rho:.2f} (p={pval:.3f}), fine-tune rho={ft_rho:.2f} (p={ft_pval:.3f})")
        summary["crossings"][n] = {
            "probe": {p: pc[p] for p in pairs}, "ft": {p: fc[p] for p in pairs},
            "probe_median": (np.median(pf) if pf else None),
            "probe_never_cross": sum(v is None for v in pc.values()),
            "gap_spearman_rho": float(rho), "gap_spearman_p": float(pval),
            "ft_gap_spearman_rho": float(ft_rho), "ft_gap_spearman_p": float(ft_pval),
        }
    OUT.write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
