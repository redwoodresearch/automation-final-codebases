"""Rank-correlation, winner-agreement, calibration, and bootstrap-CI helpers.

Shared by the label-token invariance arm (compare a per-pair Δ vector across read-out tokens) and the
transitivity arm. We compare RANKINGS and signs, not raw nats, because the log-odds scale differs across
label tokens (different base rates / calibration), so raw-nat equality is the wrong test (INSTRUCTIONS).

Bootstrap CIs resample the PAIRS axis (sampling uncertainty over which pairs we drew). This underestimates
uncertainty slightly because pairs share features (non-independent); we say so and also report seed-level
spread separately. Pattern mirrors lib/spread_finetune.bootstrap_bounded_ci (seeded rng, percentile CI).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def spearman(x, y) -> float:
    return float(stats.spearmanr(x, y).statistic)


def kendall(x, y) -> float:
    return float(stats.kendalltau(x, y).statistic)


def _percentile_ci(boots: np.ndarray, ci: float) -> tuple[float, float]:
    lo = (1 - ci) / 2 * 100
    return float(np.percentile(boots, lo)), float(np.percentile(boots, 100 - lo))


def bootstrap_paired_ci(x, y, stat_fn, n_boot: int = 4000, seed: int = 0, ci: float = 0.95) -> dict:
    """Resample (x_i, y_i) pairs with replacement, recompute stat_fn(x*,y*). Returns point estimate
    (on the full sample) + percentile CI. Use for rank correlations and winner-agreement."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert x.shape == y.shape and x.ndim == 1 and x.size > 0
    rng = np.random.default_rng(seed)
    n = x.size
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        # Degenerate resamples (all-identical) make rank corr undefined; skip them.
        val = stat_fn(x[idx], y[idx])
        if val is not None and np.isfinite(val):
            boots.append(val)
    boots = np.array(boots)
    lo, hi = _percentile_ci(boots, ci)
    return {"point": float(stat_fn(x, y)), "lo": lo, "hi": hi, "n": int(n), "n_boot_valid": int(boots.size)}


def winner_agreement(delta_alt, delta_ref) -> float:
    """Fraction of pairs whose signed winner matches the reference (sign(Δ_alt) == sign(Δ_ref))."""
    a = np.sign(np.asarray(delta_alt, dtype=float))
    r = np.sign(np.asarray(delta_ref, dtype=float))
    return float((a == r).mean())


def winner_agreement_ci(delta_alt, delta_ref, n_boot: int = 4000, seed: int = 0, ci: float = 0.95) -> dict:
    return bootstrap_paired_ci(delta_alt, delta_ref, lambda a, r: float((np.sign(a) == np.sign(r)).mean()),
                               n_boot=n_boot, seed=seed, ci=ci)


def linear_calibration(delta_ref, delta_alt) -> dict:
    """Least-squares fit Δ_alt = alpha + beta * Δ_ref. A pure rescale (token calibration) is beta != 1,
    alpha ~ 0 with high R²; near-zero behaviour can shift even when rank corr looks fine."""
    x = np.asarray(delta_ref, dtype=float)
    y = np.asarray(delta_alt, dtype=float)
    beta, alpha = np.polyfit(x, y, 1)
    yhat = alpha + beta * x
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"alpha": float(alpha), "beta": float(beta), "r2": float(r2)}


def near_tie_winner_null(delta_ref, threshold: float) -> float:
    """Pre-registered null for winner-agreement: a token that carries NO ranking information still agrees
    with the reference on well-separated pairs (the sign is "obvious") but is a coin-flip on near-ties.
    Expected agreement = (#|Δ_ref|>=threshold)*1 + (#|Δ_ref|<threshold)*0.5, all over n. This is the bar
    winner-agreement must clearly beat to show the token preserves near-tie information too."""
    d = np.abs(np.asarray(delta_ref, dtype=float))
    n = d.size
    return float(((d >= threshold).sum() * 1.0 + (d < threshold).sum() * 0.5) / n)


def paired_diff_test(a_by_seed, b_by_seed) -> dict:
    """Paired comparison of two conditions measured under the SAME seeds (e.g. a near-tie pair's Δ under
    the reference vs an alternative token). Returns the mean paired difference, its SE, and a paired
    t-test p-value. Used in the flip audit to ask "is this flip more than seed noise?"."""
    a = np.asarray(a_by_seed, dtype=float)
    b = np.asarray(b_by_seed, dtype=float)
    assert a.shape == b.shape and a.size >= 2
    diff = a - b
    res = stats.ttest_rel(a, b)
    return {"mean_diff": float(diff.mean()), "se": float(diff.std(ddof=1) / np.sqrt(diff.size)),
            "t": float(res.statistic), "p": float(res.pvalue), "n_seeds": int(a.size)}
