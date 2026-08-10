"""Per-feature base-model factor table for the strength-predictor phase (Stage 3, Phase 0).

Each factor is ONE number per feature, computed on the BASE model (Qwen3-1.7B, un-fine-tuned), independent
of the fitted strength s and of the confounded held-out cells. The factors:

  deltap_norm       ‖Δ_P‖ at the decision token (L20) — the baseline to beat.
  depth_com         center-of-mass layer of the per-layer probe-strength profile (computed elsewhere).
  probe_auc/margin  per-feature linear separability of feature-present vs -absent on L20 activations.
  prior_salience    base-model prompted detectability (AUC of a zero-shot yes/no detection logit margin).
  concept_align_*   cosine(Δ_P, a pre-existing concept direction): prompt-derived (act) and logit-lens.
  locality_*        categorical kind + a continuous attention-concentration measure.

This module is parameterized by the grid directory and pair roster so Stage 5 can re-run it on another
model's data. The GPU pieces go through an AsyncModalSession (cached). The 21st feature `language` lives
outside properties.PROPERTIES (see language_property.py) and is handled explicitly.
"""
from __future__ import annotations

import os

import numpy as np

import config
import dataset
import pairs_expanded
import properties

LAYER = 20  # the decision-token layer where the established ‖Δ_P‖↔s result lives


# ----------------------------------------------------------------------------- roster + pairs
def feature_kind(feature: str) -> str:
    if feature == "language":
        return "pervasive"  # whole-passage language; outside the roster (language_property.kind)
    return properties.PROPERTIES[feature].kind


def language_pairs(grid_dir=dataset.GRID_DIR) -> list[tuple[str, str, str]]:
    """(pair_key, f1, f2) for the 21st-feature pairs language__<partner>; language is always f1."""
    stems = sorted({f.rsplit("__", 1)[0] for f in os.listdir(grid_dir)
                    if f.startswith("language__") and f.endswith("__A.jsonl")})
    return [(s, "language", s.split("__", 1)[1]) for s in stems]


def all_feature_pairs(include_language: bool = True) -> list[tuple[str, str, str]]:
    """Every (pair_key, f1, f2) used to pool per-feature activations: the locked expanded roster plus,
    optionally, the language pairs."""
    pairs = [(k, pp.f1, pp.f2) for k, pp in pairs_expanded.PAIRS_EXPANDED.items()]
    if include_language:
        pairs += language_pairs()
    return pairs


def sample_feature_texts(feature: str, n_per_class: int, feature_pairs=None, seed: int = 0):
    """A balanced sample of (present_texts, absent_texts) for a feature, drawn across the pairs it
    appears in and the cells where it is present/absent. Used for the prompt-based factors and de-risking."""
    import numpy as np
    rng = np.random.default_rng(seed)
    feature_pairs = feature_pairs or all_feature_pairs()
    mine = [(k, f1, f2) for (k, f1, f2) in feature_pairs if feature in pooling_features(f1, f2)]
    pos, neg = [], []
    for key, f1, f2 in mine:
        try:
            grid = dataset.load_grid(key)
        except FileNotFoundError:
            continue
        pos_cells, neg_cells = feature_present_cells(feature, f1, f2)
        for cells, bucket in [(pos_cells, pos), (neg_cells, neg)]:
            for c in cells:
                bucket.extend(r["text"] for r in grid[c])
    pos = list(np.array(pos, dtype=object)[rng.permutation(len(pos))][:n_per_class])
    neg = list(np.array(neg, dtype=object)[rng.permutation(len(neg))][:n_per_class])
    return [str(t) for t in pos], [str(t) for t in neg]


def pooling_features(f1: str, f2: str) -> list[str]:
    """Which feature(s) to accumulate from a pair when pooling per-feature factors. For a `language__X`
    pair we accumulate ONLY `language`, not the roster partner X — so each roster feature's factors are
    pooled over its same-script pairs (matching the established deltap_norms_20.json exactly) and aren't
    contaminated by cross-lingual text. `language` is pooled over its 20 language pairs."""
    if "language" in (f1, f2):
        return ["language"]
    return [f1, f2]


def feature_present_cells(feature: str, f1: str, f2: str) -> tuple[list[str], list[str]]:
    """Which cells have the feature PRESENT vs ABSENT, given its role in the pair. A=(f1+,f2+),
    B=(f1-,f2-), held_f1pos=(f1+,f2-), held_f1neg=(f1-,f2+)."""
    if feature == f1:
        return ["A", "held_f1pos"], ["B", "held_f1neg"]
    assert feature == f2, (feature, f1, f2)
    return ["A", "held_f1neg"], ["B", "held_f1pos"]


# ----------------------------------------------------------------------------- Δ_P (centroid difference)
def _balanced_pos_neg_means(cell_means: dict[str, np.ndarray], feature: str, f1: str, f2: str):
    """The cell-balanced present/absent centroids for a feature in one pair: average the two
    present-cells' means and the two absent-cells' means so the PARTNER feature cancels (matches
    geometry_depth/compute_deltap_20.py)."""
    pos_cells, neg_cells = feature_present_cells(feature, f1, f2)
    pos = np.mean([cell_means[c] for c in pos_cells], axis=0)
    neg = np.mean([cell_means[c] for c in neg_cells], axis=0)
    return pos, neg


def pool_deltap(per_pair_cell_means: dict[str, dict[str, np.ndarray]],
                feature_pairs: list[tuple[str, str, str]]) -> dict[str, np.ndarray]:
    """Δ_P vector per feature = mean over the feature's pairs of (balanced present centroid) minus
    mean over pairs of (balanced absent centroid). `per_pair_cell_means[pair_key][cell]` is the L20
    mean activation of that cell."""
    pos: dict[str, list] = {}
    neg: dict[str, list] = {}
    for key, f1, f2 in feature_pairs:
        if key not in per_pair_cell_means:
            continue
        cm = per_pair_cell_means[key]
        for feature in pooling_features(f1, f2):
            p, n = _balanced_pos_neg_means(cm, feature, f1, f2)
            pos.setdefault(feature, []).append(p)
            neg.setdefault(feature, []).append(n)
    return {f: np.mean(pos[f], 0) - np.mean(neg[f], 0) for f in pos}


# ----------------------------------------------------------------------------- concept-direction alignment
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def concept_align_act(deltap_vec, plus_vecs: list, minus_vecs: list) -> float:
    """cosine(Δ_P, v_P) where v_P is the prompt-derived concept contrast: mean +-pole concept-vector
    minus mean −-pole concept-vector, both built at the SAME layer from definitional prompts. A
    pre-existing concept direction that never touches the confounded fit (and avoids the L20-vs-unembed
    basis mismatch of the logit lens)."""
    u = np.mean(plus_vecs, 0) - np.mean(minus_vecs, 0)
    return _cosine(deltap_vec, u)


def logit_lens_align(deltap_vec, plus_rows: list, minus_rows: list, final_norm_weight=None) -> float:
    """Logit-lens cosine: does moving along Δ_P raise the +-pole naming tokens' logits over the −-pole's?
    u = mean(unembed[+ tokens]) − mean(unembed[− tokens]); optionally apply the final RMSNorm gain to
    Δ_P first (the proper-lens approximation)."""
    d = np.asarray(deltap_vec, float)
    if final_norm_weight is not None:
        d = d * np.asarray(final_norm_weight, float)
    # unipolar feature (no naming token for the "absent" pole): fall back to the +-pole direction alone
    u = np.mean(plus_rows, 0) - (np.mean(minus_rows, 0) if len(minus_rows) else 0.0)
    return _cosine(d, u)


# ----------------------------------------------------------------------------- per-feature probe separability
def feature_probe_separability(pos_acts: np.ndarray, neg_acts: np.ndarray, C: float = 0.1,
                               n_splits: int = 5, seed: int = 0) -> dict:
    """Cross-validated linear separability of feature-present vs -absent on base-model L20 activations.
    Returns mean held-out ROC-AUC (primary, threshold-free) and the mean logistic margin |w·x+b|."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    X = np.concatenate([pos_acts, neg_acts], 0)
    y = np.array([1] * len(pos_acts) + [0] * len(neg_acts))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs, margins = [], []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(C=C, max_iter=1000).fit(X[tr], y[tr])
        dec = clf.decision_function(X[te])
        aucs.append(roc_auc_score(y[te], dec))
        margins.append(float(np.mean(np.abs(dec))))
    return {"auc": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)),
            "margin": float(np.mean(margins)), "n_pos": int(len(pos_acts)), "n_neg": int(len(neg_acts))}
