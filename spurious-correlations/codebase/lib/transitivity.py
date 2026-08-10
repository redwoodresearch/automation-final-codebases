"""Transitivity analysis (Stage 1, Part B): are the head-to-head winners across the 21 features
consistent with a single GLOBAL ordering (what the additive law asserts), or do they form intransitive
cycles (rock-paper-scissors)?

Input is a flat dict of per-pair Δ keyed "f1__f2" (oriented toward f1, positive = f1 wins) — exactly
results/_pair_deltas_{ft,pr}.json. We reconstruct the winner of every measured pair, then:
  * cycle_rate           — fraction of fully-measured triples whose sign(Δ) tournament is intransitive.
  * additivity_residuals — does Δ(a,c) ≈ Δ(a,b)+Δ(b,c) (the magnitude-level / triangle-"curl" test)?
  * fit_strengths        — the additive-law WLS fit (mean-zero s), matching binary_sweep/fit_law.py.
  * s_ordering_match     — fraction of ordered triples whose 3 winners match the fitted-s ranking.
Null cycle rates: a random-tournament null (0.25 for triples) and a "global scale + per-pair noise" null
(simulate Δ* = s_i - s_j + noise and report the cycle rate noise ALONE produces). The honest, primary
noise null draws the per-pair noise from SEED variance (independent measurement noise), not from the fit
residuals — the latter is circular because s is fit on this same Δ data (so real cycles inflate it).
"""
from __future__ import annotations

import itertools

import numpy as np


def features_in(pair_deltas: dict[str, float]) -> list[str]:
    feats = set()
    for k in pair_deltas:
        a, b = k.split("__")
        feats.add(a)
        feats.add(b)
    return sorted(feats)


def oriented_delta(pair_deltas: dict[str, float], a: str, b: str) -> float | None:
    """Δ oriented a->b (positive => a wins / model follows a over b). None if the pair is unmeasured."""
    if f"{a}__{b}" in pair_deltas:
        return pair_deltas[f"{a}__{b}"]
    if f"{b}__{a}" in pair_deltas:
        return -pair_deltas[f"{b}__{a}"]
    return None


def winner(pair_deltas: dict[str, float], a: str, b: str) -> str | None:
    d = oriented_delta(pair_deltas, a, b)
    if d is None:
        return None
    return a if d > 0 else b


def measured_triples(pair_deltas: dict[str, float]) -> list[tuple[str, str, str]]:
    """All feature triples (sorted) with ALL THREE pairwise Δ measured."""
    feats = features_in(pair_deltas)
    out = []
    for tri in itertools.combinations(feats, 3):
        a, b, c = tri
        if all(oriented_delta(pair_deltas, x, y) is not None for x, y in [(a, b), (b, c), (a, c)]):
            out.append(tri)
    return out


def is_cyclic(pair_deltas: dict[str, float], triple) -> bool:
    """A 3-node tournament is intransitive iff every node has exactly one win (1-1-1)."""
    a, b, c = triple
    wins = {a: 0, b: 0, c: 0}
    for x, y in [(a, b), (b, c), (a, c)]:
        wins[winner(pair_deltas, x, y)] += 1
    return set(wins.values()) == {1}


def min_abs_delta(pair_deltas: dict[str, float], triple) -> float:
    a, b, c = triple
    return min(abs(oriented_delta(pair_deltas, x, y)) for x, y in [(a, b), (b, c), (a, c)])


def cycle_stats(pair_deltas: dict[str, float]) -> dict:
    triples = measured_triples(pair_deltas)
    cyclic = [t for t in triples if is_cyclic(pair_deltas, t)]
    return {
        "n_triples": len(triples),
        "n_cyclic": len(cyclic),
        "cycle_rate": len(cyclic) / len(triples) if triples else float("nan"),
        "cyclic_triples": cyclic,
    }


# ----------------------------------------------------------------------------- additive-law fit
def fit_strengths(pair_deltas: dict[str, float], order: list[str] | None = None) -> dict:
    """WLS additive-law fit s with mean-zero constraint and design row +1 for f1, -1 for f2 — identical
    procedure to binary_sweep/fit_law.py. Returns s (dict), per-pair residuals, residual std, and R²
    (in-sample). `order` only sets the feature index order (cosmetic); the fit is order-invariant."""
    pairs = list(pair_deltas.keys())
    feats = features_in(pair_deltas)
    if order is not None:
        feats = sorted(feats, key=lambda f: order.index(f) if f in order else len(order))
    idx = {f: i for i, f in enumerate(feats)}

    def f1f2(key):
        a, b = key.split("__")
        return a, b

    X = np.zeros((len(pairs), len(feats)))
    for i, p in enumerate(pairs):
        a, b = f1f2(p)
        X[i, idx[a]] = 1
        X[i, idx[b]] = -1
    y = np.array([pair_deltas[p] for p in pairs])
    Xc = np.vstack([X, np.ones(len(feats))])  # mean-zero constraint
    s_vec, *_ = np.linalg.lstsq(Xc, np.concatenate([y, [0.0]]), rcond=None)
    s = {f: float(s_vec[idx[f]]) for f in feats}
    resid = {p: float(y[i] - (s[f1f2(p)[0]] - s[f1f2(p)[1]])) for i, p in enumerate(pairs)}
    rvals = np.array(list(resid.values()))
    r2 = 1 - (rvals ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return {"s": s, "residuals": resid, "resid_std": float(rvals.std(ddof=1)), "r2_insample": float(r2)}


def s_ordering_match(pair_deltas: dict[str, float], s: dict[str, float]) -> dict:
    """Fraction of fully-measured triples whose THREE head-to-head winners all agree with the fitted-s
    ranking (the higher-s feature wins each pair). Chance for a triple = (1/2)^3 = 0.125 if winners were
    independent coin-flips, but the relevant baseline is the random-tournament transitive structure."""
    triples = measured_triples(pair_deltas)
    n_match = 0
    for tri in triples:
        ok = True
        for x, y in itertools.combinations(tri, 2):
            w = winner(pair_deltas, x, y)
            s_winner = x if s[x] > s[y] else y
            if w != s_winner:
                ok = False
                break
        n_match += int(ok)
    return {"n_triples": len(triples), "n_match": n_match,
            "frac_match": n_match / len(triples) if triples else float("nan")}


# ----------------------------------------------------------------------------- additivity (triangle curl)
def additivity_residuals(pair_deltas: dict[str, float]) -> dict:
    """For each fully-measured triple (a<b<c sorted), predicted Δ(a->c) = Δ(a->b) + Δ(b->c); residual =
    observed Δ(a->c) - predicted (equals minus the triangle "curl" Δ_ab+Δ_bc+Δ_ca, which is 0 under
    perfect additivity). Returns the residual array, std, and R² of observed-vs-predicted Δ(a->c).
    NB: triples reuse edges, so these are NOT independent samples (report descriptively)."""
    triples = measured_triples(pair_deltas)
    pred, obs = [], []
    per_triple = []
    for tri in triples:
        a, b, c = tri
        d_ab = oriented_delta(pair_deltas, a, b)
        d_bc = oriented_delta(pair_deltas, b, c)
        d_ac = oriented_delta(pair_deltas, a, c)
        p = d_ab + d_bc
        pred.append(p)
        obs.append(d_ac)
        per_triple.append((tri, d_ac - p))
    pred = np.array(pred)
    obs = np.array(obs)
    resid = obs - pred
    ss_tot = ((obs - obs.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else float("nan")
    return {"n_triples": len(triples), "pred": pred.tolist(), "obs": obs.tolist(),
            "resid_std": float(resid.std(ddof=1)) if resid.size > 1 else 0.0,
            "resid_max_abs": float(np.abs(resid).max()) if resid.size else 0.0,
            "r2": float(r2), "per_triple_resid": per_triple}


# ----------------------------------------------------------------------------- null cycle rates
def random_tournament_cycle_rate() -> float:
    """A uniformly random orientation of a 3-tournament is cyclic with probability 2/8 = 0.25."""
    return 0.25


def noise_null_cycle_rate(s: dict[str, float], pair_keys: list[str], sigma, n_sim: int = 2000,
                          seed: int = 0) -> dict:
    """Cycle rate produced by a transitive latent scale s PLUS per-pair noise. For each simulation we
    draw Δ*_pair ~ N(s_f1 - s_f2, sigma_pair) for EVERY measured pair (so a pair shared across triples
    gets ONE draw, matching the real data), then compute the cycle rate over all fully-measured triples.
    `sigma` is either a scalar (same noise for every pair) or a dict pair_key -> sigma. Returns the mean
    and a 2.5/97.5 band of the per-simulation cycle rate."""
    rng = np.random.default_rng(seed)
    feats = features_in({k: 0.0 for k in pair_keys})

    def sig(k):
        return float(sigma) if np.isscalar(sigma) else sigma[k]

    triples = measured_triples({k: 0.0 for k in pair_keys})
    mu = {k: s[k.split("__")[0]] - s[k.split("__")[1]] for k in pair_keys}
    rates = []
    for _ in range(n_sim):
        draw = {k: mu[k] + rng.normal(0, sig(k)) for k in pair_keys}
        n_cyc = sum(is_cyclic(draw, t) for t in triples)
        rates.append(n_cyc / len(triples))
    rates = np.array(rates)
    return {"mean": float(rates.mean()), "lo": float(np.percentile(rates, 2.5)),
            "hi": float(np.percentile(rates, 97.5)), "n_sim": n_sim, "n_triples": len(triples)}


def feedback_distance(pair_deltas: dict[str, float]) -> dict:
    """Minimum number of pairwise-winner reversals needed to make the whole tournament acyclic = a
    compact 'distance from a global order'. We find the feature ordering (by fitted s, then brute-force
    refine is overkill for 21 nodes) minimizing 'upsets' (pairs where the lower-ranked feature wins).
    Here we report upsets under the fitted-s order (an upper bound on the true minimum)."""
    fit = fit_strengths(pair_deltas)
    s = fit["s"]
    order = sorted(s, key=lambda f: -s[f])  # strongest first
    rank = {f: i for i, f in enumerate(order)}
    upsets = 0
    total = 0
    for k in pair_deltas:
        a, b = k.split("__")
        w = winner(pair_deltas, a, b)
        loser = b if w == a else a
        total += 1
        if rank[w] > rank[loser]:  # winner is ranked BELOW the loser => an upset vs the s-order
            upsets += 1
    return {"upsets_vs_s_order": upsets, "n_pairs": total, "upset_rate": upsets / total if total else float("nan")}
