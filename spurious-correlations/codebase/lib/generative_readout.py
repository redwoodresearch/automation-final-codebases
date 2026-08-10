"""Shared spine for the generative read-out arm (Stage 2): orient a behavioral generation read onto
the fixed f1/f2 frame, compare it to the label fine-tune ladder, and the length-control machinery.

Orientation (identical to lib/metric.py): pole+ is taught on class A = (f1+, f2+) and pole- on class
B = (f1-, f2-). So on a held_f1pos = (f1+, f2-) prompt "follows f1" = pole+ ; on a held_f1neg =
(f1-, f2+) prompt "follows f1" = pole-. The behavioral follows-f1 rate per cell is therefore:
    held_f1pos: P(pole+)         (the share of coherent responses judged '+')
    held_f1neg: 1 - P(pole+)
pooled over the two cells (equal weight) into one follows-f1 rate q in [0,1], directly comparable to
the label fine-tune's per-cell bounded follows-f1 rate p.

We compare q to the label ladder by RANK and SIGN first (q is a noisy, non-linear function of p, so y=x
is a curve, not a line: at near-ties the behavior commits more decisively than the near-coin-flip label,
while on decisive pairs it is no more saturated than the near-deterministic label — see the temp-0.7
analysis, which refuted a stronger "generation amplifies into near-categorical behavior" reading). We
report the y=x deviation secondarily. The headline is the signed inheritance margin (below).
"""
from __future__ import annotations

import json
import re

import numpy as np

import config
import dataset
import readout_invariance as ri

HELD = dataset.HELDOUT_CELLS  # ["held_f1pos", "held_f1neg"]
COHERENT = ("+", "-")


# ----------------------------------------------------------------------------- behavior read (q)
def plus_share(labels: list[str]) -> float | None:
    """Share of '+' among coherent ('+'/'-') judge labels; None if no coherent responses."""
    coh = [x for x in labels if x in COHERENT]
    return (coh.count("+") / len(coh)) if coh else None


def neither_rate(labels: list[str]) -> float:
    return sum(1 for x in labels if x not in COHERENT) / max(1, len(labels))


def follows_f1_per_cell(labels_by_cell: dict[str, list[str]]) -> dict[str, float | None]:
    """{held_f1pos: P(pole+), held_f1neg: 1 - P(pole+)} = the follows-f1 rate per held cell."""
    fp = plus_share(labels_by_cell["held_f1pos"])
    fn = plus_share(labels_by_cell["held_f1neg"])
    return {"held_f1pos": fp, "held_f1neg": (None if fn is None else 1.0 - fn)}


def behavior_q(labels_by_cell: dict[str, list[str]]) -> float | None:
    """Overall follows-f1 rate q, the equal-weighted mean of the two held cells' follows-f1 rates.
    None if either cell has no coherent responses (the read is undefined)."""
    per = follows_f1_per_cell(labels_by_cell)
    if per["held_f1pos"] is None or per["held_f1neg"] is None:
        return None
    return 0.5 * (per["held_f1pos"] + per["held_f1neg"])


# ----------------------------------------------------------------------------- label reference (p)
def load_label_reference(path=None) -> dict[str, tuple[float, float]]:
    """pair -> (p_f1pos, p_f1neg): the label fine-tune's per-cell bounded follows-f1 rate, from
    finetune_expanded_values.jsonl (`bounded_held_f1pos` / `bounded_held_f1neg`)."""
    path = path or (config.RESULTS_DIR / "finetune_expanded_values.jsonl")
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["pair"]] = (r["bounded_held_f1pos"], r["bounded_held_f1neg"])
    return out


def load_label_seeds(path=None) -> dict[str, list[float]]:
    """pair -> per-seed bounded follows-f1 means (`ft_bounded_seeds`) — the label read's own seed
    uncertainty, so the scatter's x-axis is not treated as exact."""
    path = path or (config.RESULTS_DIR / "finetune_expanded_values.jsonl")
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["pair"]] = list(r.get("ft_bounded_seeds", []))
    return out


def label_cells(pair: str, ref: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
    """(p_f1pos, p_f1neg) for `pair`, handling a reversed pair key (f1<->f2 swap flips both cell roles
    and the follows-f1 sense)."""
    if pair in ref:
        return ref[pair]
    a, b = pair.split("__")
    sw = f"{b}__{a}"
    if sw in ref:
        fp, fn = ref[sw]
        return (1.0 - fn, 1.0 - fp)
    return None


def label_q(pair: str, ref: dict[str, tuple[float, float]]) -> float | None:
    cells = label_cells(pair, ref)
    return None if cells is None else 0.5 * (cells[0] + cells[1])


def load_delta_ladder(path=None) -> dict[str, float]:
    """pair -> oriented Δ (nats) from the token fine-tune (`_pair_deltas_ft.json`), with the f1<->f2
    swap (Δ(f2,f1) = -Δ(f1,f2)) for reversed keys."""
    path = path or (config.RESULTS_DIR / "_pair_deltas_ft.json")
    return json.load(open(path))


def delta_for(pair: str, ladder: dict[str, float]) -> float | None:
    if pair in ladder:
        return ladder[pair]
    a, b = pair.split("__")
    sw = f"{b}__{a}"
    return -ladder[sw] if sw in ladder else None


# ----------------------------------------------------------------------------- inheritance statistics
def signed_inheritance(q: float, p: float) -> float:
    """(2q-1) * sign(2p-1): + when the behavior follows the feature the label classifier picked,
    0 = no relation, - = anti-inheritance. q, p are follows-f1 rates in [0,1]."""
    return (2 * q - 1) * np.sign(2 * p - 1)


def deviation(q: float, p: float) -> float:
    return abs(q - p)


# ----------------------------------------------------------------------------- read-side length balancing
def length_balanced_q(labels_pos, lens_pos, labels_neg, lens_neg, bin_width: int = 12) -> dict:
    """Read-side length control: recompute the follows-f1 rate q with the two held cells MATCHED to an
    identical response-length histogram, so length cannot be what distinguishes the cells (and thus cannot
    drive q). Bin both cells' responses by token length; weight each bin by the min coherent count across
    the two cells; the per-cell pole+ rate is the bin-weight-averaged plus_share. q = 0.5 + (r+ − r−)/2.

    If q stays put after balancing => the behavioral read survives at matched length (content). If q → 0.5
    => the apparent behavior was length (length-collapse — verbose/terse the canonical case). Returns the
    balanced q + the effective matched count; None q if too little overlap."""
    lp, ln = np.asarray(lens_pos, float), np.asarray(lens_neg, float)
    all_lens = np.concatenate([lp, ln]) if (lp.size and ln.size) else np.array([0.0])
    lo = int(np.floor(all_lens.min() / bin_width) * bin_width)
    hi = int(np.ceil(all_lens.max() / bin_width) * bin_width) + bin_width
    edges = np.arange(lo, hi + bin_width, bin_width)

    def coherent_plus_rate(labels, lens, b_lo, b_hi):
        sel = [lab for lab, L in zip(labels, lens) if b_lo <= L < b_hi and lab in COHERENT]
        return (sel.count("+") / len(sel), len(sel)) if sel else (None, 0)

    num_p = num_n = wsum = eff = 0.0
    for i in range(len(edges) - 1):
        b_lo, b_hi = edges[i], edges[i + 1]
        rp, np_ = coherent_plus_rate(labels_pos, lens_pos, b_lo, b_hi)
        rn, nn_ = coherent_plus_rate(labels_neg, lens_neg, b_lo, b_hi)
        if rp is None or rn is None:
            continue
        w = min(np_, nn_)
        num_p += w * rp
        num_n += w * rn
        wsum += w
        eff += w
    if wsum == 0:
        return {"q": None, "n_effective": 0}
    r_plus, r_neg = num_p / wsum, num_n / wsum
    return {"q": 0.5 + (r_plus - r_neg) / 2, "r_plus": r_plus, "r_neg": r_neg, "n_effective": int(eff)}


# ----------------------------------------------------------------------------- cluster bootstrap
def cluster_bootstrap_ci(per_pair: dict[str, list[float]], stat_fn, n_boot: int = 4000,
                         seed: int = 0, ci: float = 0.95) -> dict:
    """Bootstrap a statistic over a TWO-LEVEL structure: resample pairs with replacement, then within
    each resampled pair resample its per-seed values with replacement, then apply stat_fn to the flat
    list of per-pair point estimates (the per-pair value is the mean of its resampled seeds). This
    pairs the bootstrap over (pairs, seeds) and never treats individual generations as independent
    (INSTRUCTIONS). per_pair maps pair -> list of per-seed values; stat_fn maps a list of per-pair
    values -> float."""
    pairs = [p for p, v in per_pair.items() if v]
    arrs = {p: np.asarray(per_pair[p], dtype=float) for p in pairs}
    rng = np.random.default_rng(seed)
    point = float(stat_fn([float(arrs[p].mean()) for p in pairs]))
    boots = []
    n = len(pairs)
    for _ in range(n_boot):
        chosen = [pairs[i] for i in rng.integers(0, n, n)]
        vals = []
        for p in chosen:
            a = arrs[p]
            vals.append(float(a[rng.integers(0, a.size, a.size)].mean()))
        v = stat_fn(vals)
        if v is not None and np.isfinite(v):
            boots.append(v)
    boots = np.array(boots)
    lo = (1 - ci) / 2 * 100
    return {"point": point, "lo": float(np.percentile(boots, lo)), "hi": float(np.percentile(boots, 100 - lo)),
            "n_pairs": n, "n_boot_valid": int(boots.size)}


def label_seed_rates(pair: str, seeds_by_pair: dict[str, list[float]]) -> list[float]:
    """The label fine-tune's per-seed follows-f1 rates for `pair`, swap-aware (a reversed pair key flips
    the follows-f1 sense to 1 - rate, exactly like `label_cells`)."""
    if seeds_by_pair.get(pair):
        return list(seeds_by_pair[pair])
    a, b = pair.split("__")
    sw = f"{b}__{a}"
    if seeds_by_pair.get(sw):
        return [1.0 - r for r in seeds_by_pair[sw]]
    return []


def signed_margin_label_seed_ci(q_by_pair: dict[str, list[float]],
                                label_rates_by_pair: dict[str, list[float]],
                                point_sign_by_pair: dict[str, float],
                                per_pair_mean: bool = True, resample_within: bool = True,
                                n_boot: int = 5000, seed: int = 0, ci: float = 0.95) -> dict:
    """Mean signed inheritance margin (2q−1)·sign(2p−1) with a cluster bootstrap that ALSO propagates the
    label read's per-seed sign uncertainty: each bootstrap iteration redraws every chosen pair's
    follows-f1 sign from a resample of that pair's per-seed label rates, so near-tie sign flips widen the
    CI (INSTRUCTIONS: carry the label read's uncertainty; the well-separated stratum is unaffected because
    |Δ|≥8 label signs are seed-stable). `q_by_pair` maps pair -> behavior follows-f1 values (per-seed for
    one behavior with per_pair_mean=True; per-behavior means for the pooled summary with
    per_pair_mean=False). `point_sign_by_pair` is sign(2·p_point−1) from the label point rate."""
    pairs = [p for p in q_by_pair if len(q_by_pair[p])]
    qarrs = {p: np.asarray(q_by_pair[p], dtype=float) for p in pairs}
    larrs = {p: np.asarray(label_rates_by_pair.get(p, []), dtype=float) for p in pairs}
    rng = np.random.default_rng(seed)

    def pair_points(p, sign, qa):
        m = (2 * qa - 1) * sign
        return [float(m.mean())] if per_pair_mean else m.tolist()

    point = float(np.mean([v for p in pairs for v in pair_points(p, point_sign_by_pair[p], qarrs[p])]))
    boots, n = [], len(pairs)
    for _ in range(n_boot):
        vals = []
        for p in (pairs[i] for i in rng.integers(0, n, n)):
            la = larrs[p]
            sign = (float(np.sign(2 * la[rng.integers(0, la.size, la.size)].mean() - 1)) or
                    point_sign_by_pair[p]) if la.size else point_sign_by_pair[p]
            qa = qarrs[p]
            if resample_within and qa.size > 1:
                qa = qa[rng.integers(0, qa.size, qa.size)]
            vals += pair_points(p, sign, qa)
        boots.append(float(np.mean(vals)))
    boots = np.array(boots)
    lop = (1 - ci) / 2 * 100
    return {"point": point, "lo": float(np.percentile(boots, lop)), "hi": float(np.percentile(boots, 100 - lop)),
            "n_pairs": n, "n_points": int(sum(qarrs[p].size for p in pairs)), "label_seed_propagated": True}


# ----------------------------------------------------------------------------- length features
_SENT_END = re.compile(r"[.!?]+")
_PUNCT = re.compile(r"[,;:()\"'\-]")


def length_features(text: str, n_tokens: int | None = None, hit_eos: bool | None = None) -> list[float]:
    """A small bag of length/shape features for the length-only confound classifier (INSTRUCTIONS Part
    B): token count, char count, word count, sentence count, EOS-vs-maxtok flag, punctuation count,
    paragraph count. n_tokens/hit_eos come from the GPU result when available; token count falls back
    to a host re-tokenize."""
    import data_checks

    n_tok = n_tokens if n_tokens is not None else data_checks.num_qwen_tokens(text)
    n_sent = len(_SENT_END.findall(text)) or 1
    return [
        float(n_tok),
        float(len(text)),
        float(len(text.split())),
        float(n_sent),
        float(1.0 if hit_eos else 0.0),
        float(len(_PUNCT.findall(text))),
        float(data_checks.paragraph_count(text)),
    ]


LENGTH_FEATURE_NAMES = ["tokens", "chars", "words", "sentences", "hit_eos", "punct", "paragraphs"]


def length_only_accuracy(feats: np.ndarray, y: np.ndarray, cv: int = 5, seed: int = 0) -> dict:
    """Cross-validated accuracy of a length-only classifier predicting the binary target y from the
    length features, with a label-permutation chance ceiling (the ~95th-percentile null at this N).
    `cv_acc` near `chance_ceiling` => length carries no pole/cell signal."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    feats = np.asarray(feats, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(set(y.tolist())) < 2 or len(y) < 2 * cv:
        return {"cv_acc": 0.5, "chance_ceiling": 0.5, "n": int(len(y)), "majority": 0.5}
    folds = max(2, min(cv, int(np.bincount(y).min())))
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    cv_acc = float(cross_val_score(clf, feats, y, cv=folds, scoring="accuracy").mean())
    rng = np.random.default_rng(seed)
    # subsample very large groups for the permutation ceiling (it only needs the null spread, not all rows);
    # keeps the 28-pair analysis fast enough to finish in one shot.
    if len(y) > 3000:
        idx = rng.choice(len(y), 3000, replace=False)
        feats_n, y_n = feats[idx], y[idx]
    else:
        feats_n, y_n = feats, y
    folds_n = max(2, min(cv, int(np.bincount(y_n).min())))
    null = [cross_val_score(clf, feats_n, rng.permutation(y_n), cv=folds_n, scoring="accuracy").mean()
            for _ in range(80)]
    return {"cv_acc": cv_acc, "chance_ceiling": float(np.percentile(null, 95)),
            "n": int(len(y)), "majority": float(np.bincount(y).max() / len(y))}


# ----------------------------------------------------------------------------- gap-spanning pair subset
def gap_spanning_subset(n_near: int, n_mid: int, n_large: int, ladder: dict[str, float] | None = None) -> list[str]:
    """Deterministic gap-spanning subset of the frozen CURATED_PAIRS: take n_near near-ties (|Δ|<3),
    n_mid mid-gap (3<=|Δ|<8), n_large well-separated (|Δ|>=8), each evenly spaced by |Δ| within its
    stratum. Keeps the near-tie (global-attractor-sensitive) and well-separated (load-bearing) regimes."""
    ladder = ladder or load_delta_ladder()

    def absd(p):
        return abs(delta_for(p, ladder))

    near = sorted([p for p in ri.CURATED_PAIRS if absd(p) < ri.NEAR_TIE_THRESHOLD], key=absd)
    mid = sorted([p for p in ri.CURATED_PAIRS if ri.NEAR_TIE_THRESHOLD <= absd(p) < 8.0], key=absd)
    large = sorted([p for p in ri.CURATED_PAIRS if absd(p) >= 8.0], key=absd)

    def evenly(items, k):
        if len(items) <= k:
            return items
        idx = np.linspace(0, len(items) - 1, k).round().astype(int)
        return [items[i] for i in sorted(set(idx))]

    return evenly(near, n_near) + evenly(mid, n_mid) + evenly(large, n_large)
