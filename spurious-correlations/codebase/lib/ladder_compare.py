"""Compare one per-pair read-out vector to a reference Δ ladder by RANK + WINNER (never raw magnitude),
and refit the additive strength law. Shared by the read-out arms (this phase: the scalar / value-head arm;
the verbalized arm has an equivalent inline copy that predates this module).

A "ladder" is a dict {pair_key "f1__f2": oriented value} (positive = the read follows f1). We compare by
Spearman/Kendall (bootstrap CIs over pairs) and signed-winner agreement (stratified by the reference gap,
vs the no-information null), because the scalar's scale/offset is arbitrary per pair.
"""
from __future__ import annotations

import numpy as np

import rank_stats as rs
import transitivity
from readout_invariance import NEAR_TIE_THRESHOLD


def gap_stratum(abs_delta: float) -> str:
    """large = well-separated (|Δ_ref| ≥ 8, the 'should-never-flip' anchors); near = near-tie
    (|Δ_ref| < 3, the seed-unstable, read-out-sensitive regime); mid otherwise."""
    return "large" if abs_delta >= 8 else ("mid" if abs_delta >= NEAR_TIE_THRESHOLD else "near")


def compare_ladder(value_by_pair: dict, ref_by_pair: dict, label: str) -> dict:
    """Rank-correlation + winner-agreement (overall + gap-stratified) of a per-pair read vector vs a
    reference Δ ladder, on their shared pairs. CIs bootstrap the PAIRS axis."""
    pairs = sorted(set(value_by_pair) & set(ref_by_pair))
    v = np.array([value_by_pair[p] for p in pairs])
    t = np.array([ref_by_pair[p] for p in pairs])
    spear = rs.bootstrap_paired_ci(t, v, lambda a, b: rs.spearman(a, b))
    kend = rs.bootstrap_paired_ci(t, v, lambda a, b: rs.kendall(a, b))
    wa = rs.winner_agreement_ci(v, t)
    strata = {}
    for s in ("large", "mid", "near"):
        idx = [i for i, p in enumerate(pairs) if gap_stratum(abs(t[i])) == s]
        if idx:
            strata[s] = {"n": len(idx), "winner_agreement": float((np.sign(v[idx]) == np.sign(t[idx])).mean())}
    flips = [p for p in pairs if np.sign(value_by_pair[p]) != np.sign(ref_by_pair[p])]
    flip_detail = sorted([{"pair": p, "ref": round(ref_by_pair[p], 2), "value": round(value_by_pair[p], 3),
                           "stratum": gap_stratum(abs(ref_by_pair[p]))} for p in flips],
                         key=lambda r: abs(r["ref"]))
    return {"reference": label, "n_pairs": len(pairs),
            "spearman": spear, "kendall": kend, "winner_agreement": wa,
            "near_tie_null": rs.near_tie_winner_null(t, NEAR_TIE_THRESHOLD),
            "winner_agreement_by_gap": strata, "n_flips": len(flips), "flips": flip_detail}


# ----------------------------------------------------------------------------- additive strength law refit
def fit_s(value_by_pair: dict) -> dict:
    """Refit per-feature strength s to a read-out ladder via the additive law Δ(f1,f2) ≈ s(f1) − s(f2)
    (mean-zero gauge). Reuses transitivity.fit_strengths (the committed WLS fit). Returns its dict
    {s, residuals, resid_std, r2_insample}."""
    return transitivity.fit_strengths(value_by_pair)


def loo_r2(value_by_pair: dict) -> float:
    """Leave-one-pair-out R² of the additive fit (the honest test). Refit s on all-but-one pair, predict the
    held-out pair. Only meaningful with pairs >> features (underpowered on a curated subset — report
    alongside the s-rank agreement, which is the robust statement)."""
    items = list(value_by_pair.items())
    if len(items) < 12:
        return float("nan")
    preds, actual = [], []
    for i in range(len(items)):
        train = dict(items[:i] + items[i + 1:])
        f1, f2 = items[i][0].split("__")
        s = transitivity.fit_strengths(train)["s"]
        if f1 in s and f2 in s:
            preds.append(s[f1] - s[f2])
            actual.append(items[i][1])
    a, p = np.array(actual), np.array(preds)
    return float(1 - ((a - p) ** 2).sum() / ((a - a.mean()) ** 2).sum()) if len(a) > 2 else float("nan")


def s_rank_agreement(value_by_pair: dict, s_ref: dict) -> dict:
    """Refit s on the read-out ladder and report its feature-ranking agreement (Spearman) with a reference
    per-feature strength s_ref (e.g. the token s). The robust 'same strength scale' statement."""
    fit = fit_s(value_by_pair)
    s = fit["s"]
    shared = sorted(set(s) & set(s_ref))
    return {"n_pairs": len(value_by_pair), "n_features": len(s), "shared_features": len(shared),
            "additive_r2_insample": fit["r2_insample"], "additive_loo_r2": loo_r2(value_by_pair),
            "spearman_s_vs_ref": rs.spearman([s[f] for f in shared], [s_ref[f] for f in shared]),
            "s": {f: round(s[f], 3) for f in sorted(s)}}
