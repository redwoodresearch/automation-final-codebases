"""Winner-prediction predictors, baselines, and scoring for stage 5 blog analysis 2.

The technique predicts, for a confounded pair {f1,f2}, WHICH feature a classifier will commit to.
The pre-registered rule: winner = the feature with the larger cell-balanced mean-activation-diff
norm ‖d_f‖ (raw activations). This module holds the CHEAP baselines the technique is compared
against and the scoring helpers; the signature predictors themselves live in
lib/act_signature.SignatureStore. All predictors return a per-pair predicted-winner FEATURE (f1 or
f2); a target is a per-pair actual-winner feature over a decisive population.
"""
from __future__ import annotations

import math

import numpy as np

import data_checks
import dataset
import strength_factors as sf
from extension_properties import split_pair_key


# --------------------------------------------------------------------------- scoring
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def score(preds: dict[str, str | None], truth: dict[str, str], population: list[str]) -> dict:
    """Accuracy of preds vs truth over `population` (pairs). Predictions that are None (abstain /
    tie-band) are scored as 0.5 credit (coin flip) for the headline accuracy, and separately the
    decided-only accuracy + coverage are reported."""
    n = len(population)
    decided_correct = decided = abstain = 0
    for pk in population:
        w = truth[pk]
        p = preds.get(pk)
        if p is None:
            abstain += 1
        else:
            decided += 1
            decided_correct += int(p == w)
    # headline: abstain counts as 0.5
    eff_correct = decided_correct + 0.5 * abstain
    acc = eff_correct / n if n else float("nan")
    dec_acc = decided_correct / decided if decided else float("nan")
    lo, hi = wilson_ci(decided_correct, decided)  # CI on the decided predictions
    return {"acc": acc, "n": n, "decided": decided, "decided_correct": decided_correct,
            "abstain": abstain, "coverage": decided / n if n else float("nan"),
            "decided_acc": dec_acc, "decided_acc_ci": [lo, hi]}


# --------------------------------------------------------------------------- targets
def ft_winner(row: dict) -> str | None:
    """Frozen diagonal fine-tune verdict winner (None if near-tie / not decisive)."""
    v = row["verdict"]
    if v == "follows_f1":
        return row["f1"]
    if v == "follows_f2":
        return row["f2"]
    return None


def probe_winner(row: dict, decisive_thresh: float = 3.0) -> str | None:
    """Probe-axis winner: last-token L21 sign, |Δ| >= thresh decisive."""
    d = row["probe_l21_last"]
    if abs(d) < decisive_thresh:
        return None
    return row["f1"] if d > 0 else row["f2"]


def strength_order_winner(f1: str, f2: str, s: dict[str, float]) -> str | None:
    if f1 not in s or f2 not in s or s[f1] == s[f2]:
        return None
    return f1 if s[f1] > s[f2] else f2


# --------------------------------------------------------------------------- cheap baselines (per pair)
def _balanced_pole_means(values_by_cell: dict[str, float], feature: str, f1: str, f2: str
                         ) -> tuple[float, float]:
    pos_cells, neg_cells = sf.feature_present_cells(feature, f1, f2)
    return (float(np.mean([values_by_cell[c] for c in pos_cells])),
            float(np.mean([values_by_cell[c] for c in neg_cells])))


def length_separability(grid: dict, f1: str, f2: str) -> dict[str, float]:
    """Per-feature |cell-balanced Δ token-count between its + and − poles| — the dumbest confound
    (predict the winner = feature whose poles differ more in length)."""
    tok_by_cell = {c: float(np.mean([data_checks.num_qwen_tokens(r["text"]) for r in grid[c]]))
                   for c in dataset.ALL_CELLS}
    out = {}
    for f in (f1, f2):
        pos, neg = _balanced_pole_means(tok_by_cell, f, f1, f2)
        out[f] = abs(pos - neg)
    return out


def tfidf_auc(grid: dict, f1: str, f2: str, max_features: int = 2000, n_splits: int = 4,
              seed: int = 0, max_per_cell: int | None = 200) -> dict[str, float]:
    """Per-feature pair-local TF-IDF logistic CV ROC-AUC of the feature's cell-balanced present vs
    absent texts (lexical separability — the base_strength.tfidf_auc pattern, computed within the
    pair)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    out = {}
    for f in (f1, f2):
        pos_cells, neg_cells = sf.feature_present_cells(f, f1, f2)
        pos, neg = [], []
        for cells, bucket in ((pos_cells, pos), (neg_cells, neg)):
            for c in cells:
                recs = grid[c][:max_per_cell] if max_per_cell else grid[c]
                bucket += [r["text"] for r in recs]
        X = TfidfVectorizer(max_features=max_features).fit_transform(pos + neg)
        y = np.array([1] * len(pos) + [0] * len(neg))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        out[f] = float(cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=cv,
                                       scoring="roc_auc").mean())
    return out


def larger_wins(score_by_feature: dict[str, float], f1: str, f2: str,
                tie_band: float = 0.0) -> str | None:
    """Predict the feature with the larger score; abstain if within tie_band."""
    a, b = score_by_feature[f1], score_by_feature[f2]
    if abs(a - b) <= tie_band:
        return None
    return f1 if a > b else f2


def llm_winner(f1: str, f2: str, scores: dict[str, float]) -> str | None:
    if f1 not in scores or f2 not in scores or scores[f1] == scores[f2]:
        return None
    return f1 if scores[f1] > scores[f2] else f2


def type_prior_winner(f1: str, f2: str, type_mean: dict[str, float]) -> str | None:
    """Predict by the type-strength ordering; abstain when both features share a type (no signal)."""
    import features
    t1, t2 = features.feature_type(f1), features.feature_type(f2)
    if t1 == t2:
        return None
    return f1 if type_mean[t1] > type_mean[t2] else f2
