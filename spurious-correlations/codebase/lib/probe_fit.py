"""CPU-side probe fitting with readout length/topic controls (INSTRUCTIONS §4, §6, §7).

The slow/fast split: a Modal GPU call extracts all-layer activations once (modal_app.extract_activations_blob);
this module fits every probe variant locally (controls × regularization C × seeds × layers) so iteration
never touches the GPU. The uncontrolled baseline reproduces modal_app.probe_logprob_diff exactly (same
75/25 split, same LogisticRegression call, same orientation) — verified in
scripts_.../validate_local_fit.py.

The metric (oriented held-out log-odds toward f1) is the frozen metric.py. The probe's per-text decision
value is d = logP(A) - logP(B) = -(w·x + b), since sklearn orients decision_function toward class label 1
(= class B).

Length controls (the central new task, §6):
  * "none"      — no control (baseline; can ride length as the A/B boundary on length-adding pairs).
  * "covariate" — append a standardized token-length feature (A); at readout hold it at the train mean
    (set the held covariate to 0) so the direct length term is excluded. Caveat: length is also encoded
    *inside* the 2048-dim activations, so a scalar covariate under-removes it (we verify).
  * "project"   — estimate a length direction (or low-rank subspace) from WITHIN-CELL length variation
    (feature held constant), never from the A/B contrast, and project it out of all activations before
    fitting (B). Works even when A/B length distributions are nearly disjoint.
  * "reweight"  — inverse-propensity length-bin sample weights so length is non-predictive of class (C).
    The control the fine-tune can share (RQ1-comparable), but it breaks where a length bin is class-pure
    (positivity violation) — those texts are dropped (weight 0) and the dropped fraction is reported.

Topic control (§7, code/jargon pairs): project out a within-cell topic subspace (top-k PCA of per-seed
mean deviations within each cell — feature held constant, never the A/B contrast). Belt-and-suspenders,
since the training diagonal is exactly topic-balanced.
"""
from __future__ import annotations

import numpy as np
from attrs import frozen
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import metric

CELLS = ["A", "B", "held_f1pos", "held_f1neg"]


@frozen
class Control:
    """A readout control specification. name is a short label for results."""
    name: str
    length: str = "none"          # none | covariate | project | reweight
    topic: str = "none"           # none | project
    proj_powers: tuple = (1,)     # length-direction basis powers (1 = linear; (1,2) adds quadratic)
    # which cells to estimate the WITHIN-CELL nuisance subspace from: "all" (4 cells) | "ab" (train cells
    # only). BOTH estimate it within each cell separately (feature held constant) — neither uses the A-vs-B
    # contrast; "ab" is the no-held-cell-leakage variant used as a robustness check, NOT the forbidden A/B contrast.
    proj_from: str = "all"
    bin_width: int = 10           # token-length bin width for reweight
    topic_k: int = 10             # rank of the within-cell topic subspace to project out


UNCONTROLLED = Control("uncontrolled")


# ----------------------------------------------------------------------------- nuisance directions
def _cells_for(proj_from: str) -> list[str]:
    return ["A", "B"] if proj_from == "ab" else CELLS


def within_cell_length_directions(acts_by_cell: dict, lengths_by_cell: dict, powers, cells) -> np.ndarray:
    """Orthonormal U [hidden, k] spanning the within-cell linear relationship between activations and
    powers of token length. Estimated by pooling each cell's WITHIN-cell centered (length-power,
    activation) pairs (feature constant within a cell), so it never uses the feature-confounded A/B
    contrast (INSTRUCTIONS §6 pitfall)."""
    Zs, As = [], []
    for c in cells:
        A = acts_by_cell[c]
        s = np.asarray(lengths_by_cell[c], dtype=float)
        basis = np.stack([s ** p for p in powers], axis=1)
        Zs.append(basis - basis.mean(0))
        As.append(A - A.mean(0))
    Z, Acat = np.concatenate(Zs), np.concatenate(As)
    beta = np.linalg.lstsq(Z, Acat, rcond=None)[0]  # [k, hidden]: per-power direction
    Q, _ = np.linalg.qr(beta.T)                      # orthonormal [hidden, k]
    return Q


def within_cell_topic_directions(acts_by_cell: dict, seeds_by_cell: dict, k: int, cells) -> np.ndarray:
    """Orthonormal U [hidden, k] spanning within-cell TOPIC variation: per-seed mean activation
    deviations from the cell mean, pooled across cells (feature constant within a cell), top-k PCA.
    Never uses the A/B contrast."""
    devs = []
    for c in cells:
        A = acts_by_cell[c]
        seeds = np.asarray(seeds_by_cell[c])
        cellmean = A.mean(0)
        for s in np.unique(seeds):
            grp = A[seeds == s]
            if len(grp) >= 1:
                devs.append(grp.mean(0) - cellmean)
    D = np.stack(devs)
    D = D - D.mean(0)
    if D.shape[0] < 2:
        return np.zeros((acts_by_cell["A"].shape[1], 0))
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    kk = min(k, Vt.shape[0])
    return Vt[:kk].T  # [hidden, k]


def project_out(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    return X if U.shape[1] == 0 else X - (X @ U) @ U.T


def length_bin_weights(lengths: np.ndarray, labels: np.ndarray, bin_width: int) -> np.ndarray:
    """Inverse-propensity length-bin sample weights so the weighted length distribution is identical
    across A and B (length non-predictive of class). Class-pure bins (only one class present) are a
    positivity violation that re-weighting cannot fix → those texts get weight 0 (dropped). Weights are
    normalized so the two classes have equal total weight."""
    lengths = np.asarray(lengths, float)
    labels = np.asarray(labels)
    edges = np.arange(lengths.min(), lengths.max() + bin_width + 1, bin_width)
    b = np.clip(np.digitize(lengths, edges) - 1, 0, len(edges) - 2)
    w = np.zeros(len(lengths))
    for bb in np.unique(b):
        m = b == bb
        nA = int(((labels == 0) & m).sum())
        nB = int(((labels == 1) & m).sum())
        if nA > 0 and nB > 0:
            tot = nA + nB
            w[m & (labels == 0)] = tot / nA
            w[m & (labels == 1)] = tot / nB
    for cls in (0, 1):
        s = w[labels == cls].sum()
        if s > 0:
            w[labels == cls] *= (labels == cls).sum() / s  # keep per-class scale ~1
    return w


# ----------------------------------------------------------------------------- per-layer preparation
@frozen
class LayerPrep:
    Xtr: np.ndarray
    Xho: np.ndarray
    sample_weight: np.ndarray | None
    retained_frac: float  # fraction of training texts with non-zero weight (1.0 unless reweight drops bins)


def prep_layer(ctrl: Control, acts_by_cell_L: dict, y: np.ndarray, lengths_by_cell: dict,
               seeds_by_cell: dict) -> LayerPrep:
    """Build the (possibly transformed) training/held feature matrices and sample weights for ONE layer
    under `ctrl`. acts_by_cell_L holds the layer-L activations for each cell."""
    Atr = np.concatenate([acts_by_cell_L["A"], acts_by_cell_L["B"]], axis=0)
    Hho = np.concatenate([acts_by_cell_L["held_f1pos"], acts_by_cell_L["held_f1neg"]], axis=0)
    len_tr = np.concatenate([lengths_by_cell["A"], lengths_by_cell["B"]]).astype(float)
    cells = _cells_for(ctrl.proj_from)

    # length projection / topic projection transform the activation space first
    Us = []
    if ctrl.length == "project":
        Us.append(within_cell_length_directions(acts_by_cell_L, lengths_by_cell, ctrl.proj_powers, cells))
    if ctrl.topic == "project":
        Us.append(within_cell_topic_directions(acts_by_cell_L, seeds_by_cell, ctrl.topic_k, cells))
    if Us:
        U = np.concatenate(Us, axis=1)
        Q, _ = np.linalg.qr(U)  # re-orthonormalize the combined subspace
        Atr, Hho = project_out(Atr, Q), project_out(Hho, Q)

    sample_weight, retained = None, 1.0
    if ctrl.length == "reweight":
        sample_weight = length_bin_weights(len_tr, y, ctrl.bin_width)
        retained = float((sample_weight > 0).mean())

    if ctrl.length == "covariate":
        mu, sd = len_tr.mean(), len_tr.std() + 1e-9
        Xtr = np.concatenate([Atr, ((len_tr - mu) / sd)[:, None]], axis=1)
        Xho = np.concatenate([Hho, np.zeros((Hho.shape[0], 1))], axis=1)  # hold length at train mean -> excluded
        return LayerPrep(Xtr, Xho, sample_weight, retained)
    return LayerPrep(Atr, Hho, sample_weight, retained)


# ----------------------------------------------------------------------------- fitting
def _val_cross_entropy(clf, Xval, yval) -> float:
    dec = clf.decision_function(Xval)
    eps = 1e-12
    p_b = np.clip(1.0 / (1.0 + np.exp(-dec)), eps, 1 - eps)  # P(class B = label 1)
    return float(-np.mean((yval == 1) * np.log(p_b) + (yval == 0) * np.log(1 - p_b)))


def fit_one(prep: LayerPrep, y: np.ndarray, signs: np.ndarray, C: float, seed: int,
            want_dvalues: bool = False) -> dict:
    """One probe fit at fixed (layer, C, seed): 75/25 split (identical to modal_app), held-in validation
    signal (for layer selection), and the held-out logprob difference."""
    tr_idx, val_idx = train_test_split(np.arange(len(y)), test_size=0.25, random_state=seed, stratify=y)
    clf = LogisticRegression(C=C, max_iter=1000)
    sw = prep.sample_weight
    clf.fit(prep.Xtr[tr_idx], y[tr_idx], sample_weight=None if sw is None else sw[tr_idx])
    val_ce = _val_cross_entropy(clf, prep.Xtr[val_idx], y[val_idx])
    val_acc = float((clf.predict(prep.Xtr[val_idx]) == y[val_idx]).mean())
    val_margin = float(np.mean(np.abs(clf.decision_function(prep.Xtr[val_idx]))))
    d = -clf.decision_function(prep.Xho)  # oriented logP(A)-logP(B)
    res = metric.logprob_difference(d, signs)
    # per-held-cell bounded (fraction following f1 within each held cell separately) — to distinguish a
    # genuine tie (both cells ~0.5) from per-text feature-switching / XOR (cells on opposite sides), which
    # the metric's pooled average over the two cells would otherwise hide.
    signs_arr = np.asarray(signs)
    oriented = d * signs_arr
    pos = metric._sigmoid(oriented[signs_arr == 1]).mean() if (signs_arr == 1).any() else float("nan")
    neg = metric._sigmoid(oriented[signs_arr == -1]).mean() if (signs_arr == -1).any() else float("nan")
    out = {"seed": seed, "val_ce": val_ce, "val_acc": val_acc, "val_margin": val_margin,
           "heldout_raw": res.raw, "heldout_bounded": res.bounded_prob, "heldout_acc": res.hard_accuracy,
           "bounded_heldpos": float(pos), "bounded_heldneg": float(neg)}
    if want_dvalues:
        out["d_values"] = [float(x) for x in d]
    return out


def aggregate_seeds(per_seed_layer: list[dict]) -> dict:
    raw = np.array([s["heldout_raw"] for s in per_seed_layer])
    bnd = np.array([s["heldout_bounded"] for s in per_seed_layer])
    acc = np.array([s["heldout_acc"] for s in per_seed_layer])
    signs = np.sign(raw)
    return {
        "val_ce_mean": float(np.mean([s["val_ce"] for s in per_seed_layer])),
        "val_acc_mean": float(np.mean([s["val_acc"] for s in per_seed_layer])),
        "val_margin_mean": float(np.mean([s["val_margin"] for s in per_seed_layer])),
        "raw_mean": float(raw.mean()), "raw_std": float(raw.std(ddof=1)) if len(raw) > 1 else 0.0,
        "raw_min": float(raw.min()), "raw_max": float(raw.max()),
        "bounded_mean": float(bnd.mean()), "bounded_std": float(bnd.std(ddof=1)) if len(bnd) > 1 else 0.0,
        "hard_mean": float(acc.mean()), "hard_std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
        "sign_consistent": bool(np.all(signs == signs[0]) and signs[0] != 0),
        # per-seed values retained so the deliverable can report seed-level detail (INSTRUCTIONS §4)
        "raw_seeds": [float(x) for x in raw], "bounded_seeds": [float(x) for x in bnd],
        "hard_seeds": [float(x) for x in acc],
        # per-held-cell bounded (seed mean): held_f1pos and held_f1neg separately (cell-asymmetry / XOR check)
        "bounded_heldpos": float(np.mean([s["bounded_heldpos"] for s in per_seed_layer])),
        "bounded_heldneg": float(np.mean([s["bounded_heldneg"] for s in per_seed_layer])),
    }


def heldout_dvalues(acts_by_cell: dict, lengths_by_cell: dict, seeds_by_cell: dict, signs: np.ndarray,
                    ctrl: Control, C: float, layer: int, n_seeds: int = 5) -> np.ndarray:
    """Mean (over seeds) per-held-text oriented decision value d*sign at one (control, C, layer). Used
    for the §6.1 length-riding check: does the probe's held-out decision track a length-only model's?"""
    y = np.array([0] * len(acts_by_cell["A"]) + [1] * len(acts_by_cell["B"]))
    actsL = {c: acts_by_cell[c][:, layer, :] for c in CELLS}
    prep = prep_layer(ctrl, actsL, y, lengths_by_cell, seeds_by_cell)
    per_seed = [fit_one(prep, y, signs, C, s, want_dvalues=True)["d_values"] for s in range(n_seeds)]
    return (np.array(per_seed).mean(0)) * np.asarray(signs)  # oriented toward f1


def length_only_heldout(lengths_by_cell: dict, signs: np.ndarray, C: float = 1.0) -> dict:
    """The length-only shortcut baseline as a probe-side control (§11): a 1-D logistic
    regression on token length fit on the full A/B diagonal, read out on held with the metric. Returns
    the bounded readout and the per-held-text oriented decision values (to correlate with the probe).
    C is left at the sklearn default 1.0 (this is a 1-D baseline, not the probe — regularization is
    immaterial for a single feature, unlike the 2048-dim probe pinned at C=0.1)."""
    len_tr = np.array([[v] for v in lengths_by_cell["A"]] + [[v] for v in lengths_by_cell["B"]], float)
    y = np.array([0] * len(lengths_by_cell["A"]) + [1] * len(lengths_by_cell["B"]))
    len_ho = np.array([[v] for v in lengths_by_cell["held_f1pos"]] + [[v] for v in lengths_by_cell["held_f1neg"]], float)
    clf = LogisticRegression(C=C, max_iter=2000).fit(len_tr, y)
    d = -clf.decision_function(len_ho)
    res = metric.logprob_difference(d, signs)
    return {"bounded": res.bounded_prob, "raw": res.raw, "hard": res.hard_accuracy,
            "ab_train_acc": float(clf.score(len_tr, y)), "oriented_d": (d * np.asarray(signs)).tolist()}


def fit_grid(acts_by_cell: dict, lengths_by_cell: dict, seeds_by_cell: dict, signs: np.ndarray,
             controls: list[Control], C_values: list[float], layers: list[int], n_seeds: int = 5) -> dict:
    """Fit the full (control × C × layer × seed) grid for one (pair, representation).

    acts_by_cell[c] is [n_c, n_layers, hidden]. Returns
    {control_name: {"C=..": {"layers": {layer: aggregate}, "retained_frac": ...}}}.
    Per-layer control transforms are computed once and reused across C and seeds.
    """
    y = np.array([0] * len(acts_by_cell["A"]) + [1] * len(acts_by_cell["B"]))
    seeds = list(range(n_seeds))
    out: dict = {}
    for ctrl in controls:
        cres = {f"C={C:g}": {"layers": {}, "retained_frac": 1.0} for C in C_values}
        for L in layers:
            actsL = {c: acts_by_cell[c][:, L, :] for c in CELLS}
            prep = prep_layer(ctrl, actsL, y, lengths_by_cell, seeds_by_cell)
            for C in C_values:
                per_seed = [fit_one(prep, y, signs, C, s) for s in seeds]
                cres[f"C={C:g}"]["layers"][L] = aggregate_seeds(per_seed)
                cres[f"C={C:g}"]["retained_frac"] = prep.retained_frac
        out[ctrl.name] = cres
    return out
