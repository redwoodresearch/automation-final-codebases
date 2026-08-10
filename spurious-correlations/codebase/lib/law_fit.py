"""Additive-law fit generalized to arbitrary feature names (extension project).

The shipped fit (binary_sweep/fit_law.py) indexes features via GLOBAL_PROPERTY_ORDER.index, which
crashes on out-of-roster features (language, extension triggers). Here the design matrix is built
from explicit (f1, f2) pairs — same mean-zero least-squares fit, same conventions (design row +1
for f1, -1 for f2; Δ oriented toward f1) — plus leave-one-pair-out R² and a frozen-anchor refit
variant for the robustness check.
"""

from __future__ import annotations

import numpy as np


def _design(pairs: list[tuple[str, str]], idx: dict[str, int]) -> np.ndarray:
    X = np.zeros((len(pairs), len(idx)))
    for i, (f1, f2) in enumerate(pairs):
        X[i, idx[f1]] = 1
        X[i, idx[f2]] = -1
    return X


def _solve(pairs: list[tuple[str, str]], y: np.ndarray, feats: list[str]) -> dict[str, float]:
    idx = {f: i for i, f in enumerate(feats)}
    X = np.vstack([_design(pairs, idx), np.ones(len(feats))])  # + mean-zero constraint row
    s, *_ = np.linalg.lstsq(X, np.concatenate([y, [0.0]]), rcond=None)
    return {f: float(s[idx[f]]) for f in feats}


def fit_additive_law(deltas: dict[str, float], orientation: dict[str, tuple[str, str]]) -> dict:
    """Mean-zero least-squares fit of Δ ≈ s(f1) − s(f2) with leave-one-pair-out R². (Unweighted
    OLS, matching the shipped fit_law.py implementation exactly — the design docs say "WLS" but
    the shipped code, which the published numbers come from, is unweighted.)

    deltas: pair_key -> observed Δ (nats, oriented toward that pair's f1).
    orientation: pair_key -> (f1, f2). Every feature named here enters the fit.

    Returns {s, loo_pred, loo_r2, residuals, resid_std, r2_insample, pairs}.
    """
    pairs = sorted(deltas)
    fps = [orientation[p] for p in pairs]
    feats = sorted({f for fp in fps for f in fp})
    y = np.array([deltas[p] for p in pairs])

    s = _solve(fps, y, feats)
    pred = np.array([s[f1] - s[f2] for f1, f2 in fps])
    resid = y - pred

    loo_pred = np.zeros(len(pairs))
    for i in range(len(pairs)):
        keep = [j for j in range(len(pairs)) if j != i]
        s_k = _solve([fps[j] for j in keep], y[keep], feats)
        loo_pred[i] = s_k[fps[i][0]] - s_k[fps[i][1]]
    ss_tot = ((y - y.mean()) ** 2).sum()
    loo_r2 = float(1 - ((y - loo_pred) ** 2).sum() / ss_tot)

    return {
        "s": s,
        "pairs": pairs,
        "loo_pred": {p: float(v) for p, v in zip(pairs, loo_pred)},
        "loo_r2": loo_r2,
        "residuals": {p: float(r) for p, r in zip(pairs, resid)},
        "resid_std": float(resid.std(ddof=1)) if len(pairs) > 1 else 0.0,
        "r2_insample": float(1 - (resid ** 2).sum() / ss_tot),
    }


def refit_new_features_against_frozen_anchors(
    deltas: dict[str, float],
    orientation: dict[str, tuple[str, str]],
    frozen_s: dict[str, float],
) -> dict[str, float]:
    """Robustness variant: hold anchor strengths fixed at `frozen_s` (from an anchor-only fit) and
    fit ONLY the new features, over pairs whose f1 or f2 is new. New-feature strengths are then
    directly identified (no mean-zero constraint needed): each new feature f gets the least-squares
    solution of its own pairs' equations, s(f) = mean over its pairs of (s(anchor) − Δ·sign)."""
    new_feats = sorted({f for fp in orientation.values() for f in fp} - set(frozen_s))
    out: dict[str, float] = {}
    for f in new_feats:
        vals = []
        for p, (f1, f2) in orientation.items():
            if p not in deltas:
                continue
            if f1 == f and f2 in frozen_s:
                vals.append(frozen_s[f2] + deltas[p])   # Δ = s(f) − s(anchor)
            elif f2 == f and f1 in frozen_s:
                vals.append(frozen_s[f1] - deltas[p])   # Δ = s(anchor) − s(f)
        assert vals, f"no anchor pairs found for new feature {f}"
        out[f] = float(np.mean(vals))
    return out
