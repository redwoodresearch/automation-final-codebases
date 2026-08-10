"""Per-pair activation-centroid geometry: the Δ_P-style feature-difference directions and their overlap.

For a 2x2 grid's four cells of per-layer activations (each [n, n_layers, hidden]), the cell-balanced
centroid-difference direction for each feature cancels the partner feature:
    d_f1 = (mean(A) + mean(held_f1pos))/2 - (mean(B) + mean(held_f1neg))/2     (f1-present minus f1-absent)
    d_f2 = (mean(A) + mean(held_f1neg))/2 - (mean(B) + mean(held_f1pos))/2     (f2-present minus f2-absent)
``||d_f1||`` / ``||d_f2||`` measure how far each feature moves the representation at that layer (the
geometry the blog showed predicts the fitted strength); ``cos(d_f1, d_f2)`` measures how entangled the
two feature directions are. Moved here from experiments/geometry_depth/compute_pair_geometry.py so the
read-switch extraction can reuse it unchanged.
"""
from __future__ import annotations

import numpy as np


def _cross_layer_cos(d: np.ndarray) -> np.ndarray:
    """[n_layers, hidden] -> [n_layers, n_layers] cosine-similarity matrix of the direction across layers."""
    u = d / np.clip(np.linalg.norm(d, axis=1, keepdims=True), 1e-12, None)
    return u @ u.T


def per_layer_geometry(cells: dict, cross_layer: bool = True) -> dict:
    """cells[orig_cell] = [n, n_layers, hidden]; returns per-layer ||d_f1||, ||d_f2||, cos(d_f1, d_f2), and
    (if cross_layer) the cross-layer directional-stability matrices cos(d_f(L1), d_f(L2)) for each feature
    plus the adjacent-layer cosine cos(d_f(L), d_f(L+1)) — does a feature direction ROTATE across layers
    (cosine dips) or stay fixed with only its norm changing (cosine ~1, norms cross)?"""
    m = {c: cells[c].mean(axis=0) for c in cells}  # [n_layers, hidden]
    d_f1 = (m["A"] + m["held_f1pos"]) / 2 - (m["B"] + m["held_f1neg"]) / 2
    d_f2 = (m["A"] + m["held_f1neg"]) / 2 - (m["B"] + m["held_f1pos"]) / 2
    n1, n2 = np.linalg.norm(d_f1, axis=1), np.linalg.norm(d_f2, axis=1)
    cos = (d_f1 * d_f2).sum(axis=1) / np.clip(n1 * n2, 1e-12, None)
    out = {"cos_per_layer": cos.tolist(), "norm_f1_per_layer": n1.tolist(), "norm_f2_per_layer": n2.tolist()}
    if cross_layer:
        M1, M2 = _cross_layer_cos(d_f1), _cross_layer_cos(d_f2)
        out["adj_cos_f1"] = [float(M1[L, L + 1]) for L in range(M1.shape[0] - 1)]  # cos(d_f1(L), d_f1(L+1))
        out["adj_cos_f2"] = [float(M2[L, L + 1]) for L in range(M2.shape[0] - 1)]
        out["cross_layer_cos_f1"] = np.round(M1, 4).tolist()  # the full [L x L] direction-stability matrix
        out["cross_layer_cos_f2"] = np.round(M2, 4).tolist()
    return out
