"""Deterministic length-matched sub-sampling of the generation buffer (INSTRUCTIONS §5).

Length is correlated with the class label BY CONSTRUCTION here (cell A = both `+` features is longest,
cell B = both `-` shortest), and is trivially readable from activations / exploitable by a fine-tune.
So before the dataset is used we sub-sample each pair's four cells (in the subject model's Qwen3 tokens)
to a matched length distribution, with the two held-out cells at equal count and A/B balanced.

Method — identical-histogram matching, per group, on a tuned bin grid:
  * Bin texts by token length; for a group of cells take min-count-per-bin from each cell. The result
    is an IDENTICAL length histogram across the group's cells (equal count; matched median/IQR; a
    linear length classifier sits at chance). We verified that matching only the medians is NOT enough
    (it leaves A's short tail vs B's long tail, and a length classifier still gets ~0.83) — identical
    histograms are what drive the length classifier to ~0.5.
  * Two groups, matched independently and possibly at different bin widths: the HELD-OUT pair (the
    metric; the §6 length gate binds here) and the TRAIN pair A/B (so a probe can't use length as the
    class cue). `match_pair_tuned` picks, per group, the COARSEST bin that still defeats a linear length
    classifier — coarser bins retain far more, and the retention cost of an over-fine bin is partly an
    artifact (a 1-D length classifier only exploits a CDF gap). A consulted model panel endorsed this
    bin-to-the-gate tuning over both blind fine-binning and per-cell length nudging (which would swap a
    measurable length confound for an unmeasurable verbosity-style one).

Identical-histogram matching discards the non-overlapping tails (cell A's long tail, cell B's short
tail), so the buffer is over-generated and the hardest pairs may retain <300 (>=250 acceptable, or
documented below that). Selection is deterministic (records ordered within a bin by a stable sha256 of
their text), so the exact dataset reproduces.
"""
from __future__ import annotations

import hashlib

import numpy as np

import dataset

AB = ["A", "B"]
HELD = ["held_f1pos", "held_f1neg"]
ALL = AB + HELD


def _stable_rank(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def _bin_edges(all_tokens: np.ndarray, bin_width: int) -> np.ndarray:
    lo = int(np.floor(all_tokens.min() / bin_width) * bin_width)
    hi = int(np.ceil(all_tokens.max() / bin_width) * bin_width) + bin_width
    return np.arange(lo, hi + 1, bin_width)


def _match_group(recs_by_cell: dict[str, list[dict]], cells: list[str], all_tokens: np.ndarray,
                 bin_width: int, target: int | None, level_key=None, prefer_key=None) -> dict[str, list[dict]]:
    """Match `cells` to an IDENTICAL length histogram (equal count per cell). cap_b = min over the
    cells of how many texts each has in bin b; take cap_b from each (stable-rank-lowest, deterministic).
    If the matched total exceeds `target`, scale per-bin caps down proportionally (keeps the shape).

    Selection-policy hooks (stage 6 audit fixes; both default to None = the original behavior):
      * level_key(record) -> hashable: bins become (length_bin, level), i.e. the cells are matched to an
        identical JOINT histogram — used to equalize a flagged third-property marker across held cells.
      * prefer_key(record) -> sortable: within-bin order becomes (prefer, stable_rank), so dispreferred
        records (e.g. an over-concentrated entity) are only drawn when a bin's preferred pool runs out.
    """
    edges = _bin_edges(all_tokens, bin_width)
    nbins = len(edges) - 1

    def rank(r):
        return (prefer_key(r), _stable_rank(r["text"])) if prefer_key else _stable_rank(r["text"])

    binned: dict[str, dict] = {}
    for c in cells:
        recs = sorted(recs_by_cell[c], key=rank)
        b_idx = np.clip(np.digitize([r["num_tokens"] for r in recs], edges) - 1, 0, nbins - 1)
        per_bin: dict = {}
        for r, b in zip(recs, b_idx):
            key = (int(b), level_key(r)) if level_key else int(b)
            per_bin.setdefault(key, []).append(r)
        binned[c] = per_bin

    if level_key is None:
        keys = list(range(nbins))  # include empty bins: preserves the original tie-break indexing exactly
    else:
        keys = sorted(set().union(*[set(binned[c]) for c in cells]))
    caps = np.array([min(len(binned[c].get(k, [])) for c in cells) for k in keys])
    if target is not None and caps.sum() > target:
        scaled = np.floor(caps * (target / caps.sum())).astype(int)
        leftover = target - int(scaled.sum())
        frac = caps * (target / caps.sum()) - scaled
        for b in sorted(range(len(keys)), key=lambda b: (-frac[b], b)):
            if leftover <= 0:
                break
            if scaled[b] < caps[b]:
                scaled[b] += 1
                leftover -= 1
        caps = scaled

    selected = {c: [] for c in cells}
    for i, key in enumerate(keys):
        k = int(caps[i])
        for c in cells:
            selected[c].extend(binned[c].get(key, [])[:k])
    return selected


def _stats(recs: list[dict]) -> dict:
    toks = np.array([r["num_tokens"] for r in recs]) if recs else np.array([0])
    return {"n": len(recs), "median": float(np.median(toks)),
            "p25": float(np.percentile(toks, 25)), "p75": float(np.percentile(toks, 75)),
            "iqr": float(np.percentile(toks, 75) - np.percentile(toks, 25))}


def length_separability(selected: dict[str, list[dict]]) -> dict:
    """The §6 length-only baseline on a matched selection: logistic regression on Qwen3-token length
    predicting A/B on the diagonal, scored on held-out with the frozen metric. Returns the held-out
    bounded readouts (the §6 gate) and the A/B train accuracy (the A-vs-B length-separability check)."""
    from sklearn.linear_model import LogisticRegression

    import metric

    Xtr = np.array([[r["num_tokens"]] for r in selected["A"]] + [[r["num_tokens"]] for r in selected["B"]], dtype=float)
    y = np.array([0] * len(selected["A"]) + [1] * len(selected["B"]))
    Xho = np.array([[r["num_tokens"]] for r in selected["held_f1pos"]] + [[r["num_tokens"]] for r in selected["held_f1neg"]], dtype=float)
    signs = np.array([1] * len(selected["held_f1pos"]) + [-1] * len(selected["held_f1neg"]))
    if len(set(y.tolist())) < 2 or len(Xho) == 0 or len(Xtr) == 0:
        return {"ab_train_acc": 0.5, "held_hard_acc": 0.5, "held_bounded": 0.5, "held_raw": 0.0}
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, y)
    d = -clf.decision_function(Xho)  # d = logP(A)-logP(B); class 1 = B
    res = metric.logprob_difference(d, signs)
    return {"ab_train_acc": float(clf.score(Xtr, y)), "held_hard_acc": res.hard_accuracy,
            "held_bounded": res.bounded_prob, "held_raw": res.raw}


def ab_length_separability(selected: dict[str, list[dict]], cv: int = 5) -> dict:
    """A/B length-separability FLAG (INSTRUCTIONS §5), in-sample and cross-validated: how well token
    length alone predicts A vs B on the diagonal. This is NOT a gate — full_balanced intentionally leaves
    the diagonal length-unmatched, so it runs high on the length-adding pairs (that is the honest
    statement "length is a strong A/B cue here", and it is the signal segments 2-3 use to decide where
    downstream length-control matters most). Cross-validated, so it is the generalizing cue strength."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X = np.array([[r["num_tokens"]] for r in selected["A"]] + [[r["num_tokens"]] for r in selected["B"]], float)
    y = np.array([0] * len(selected["A"]) + [1] * len(selected["B"]))
    if len(set(y.tolist())) < 2 or len(y) < 2 * cv:
        return {"insample": 0.5, "cv": 0.5}
    clf = LogisticRegression(C=1.0, max_iter=2000)
    return {"insample": float(clf.fit(X, y).score(X, y)),
            "cv": float(cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean())}


def held_length_separability(selected: dict[str, list[dict]], cv: int = 5) -> dict:
    """The PRIMARY, A/B-independent held-out length check (INSTRUCTIONS §5): a cross-validated logistic
    regression on token length trying to tell held_f1pos from held_f1neg. This directly asks "do the two
    held cells differ in length?" — if they do not (~chance, 0.50), no length-based model can bias the
    metric, regardless of what the diagonal looks like. The held cells are unchanged from phase 1 (matched
    to <=~4-token median gaps), so this is fundamentally a property of them and should read ~0.50-0.54.
    `chance_ceiling` is the ~95% one-sided null accuracy at this N (normal approx); cv above it is a real
    gap, cv below it is thin-sample noise."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    hp, hn = selected["held_f1pos"], selected["held_f1neg"]
    if min(len(hp), len(hn)) < 5:  # unusably small candidate selection (e.g. an over-strict joint
        # marker match): report cv=1.0 so it fails any gate/ceiling, instead of crashing the LR
        return {"insample": 1.0, "cv": 1.0, "n": int(len(hp) + len(hn))}
    X = np.array([[r["num_tokens"]] for r in hp] + [[r["num_tokens"]] for r in hn], float)
    y = np.array([0] * len(hp) + [1] * len(hn))
    clf = LogisticRegression(C=1.0, max_iter=2000)
    folds = max(2, min(cv, len(hp), len(hn)))
    return {"insample": float(clf.fit(X, y).score(X, y)),
            "cv": float(cross_val_score(clf, X, y, cv=folds, scoring="accuracy").mean()), "n": int(len(y))}


def held_topic_separability(selected: dict[str, list[dict]], cv: int = 5) -> dict:
    """Direct held-vs-held TOPIC check — the metric-relevant topic test, parallel to
    `held_length_separability` (INSTRUCTIONS §5; added after the phase-2 review). The A/B *diagonal* topic
    gate is now 0.500 BY CONSTRUCTION (full_balanced gives A and B identical per-seed counts), so it is
    structurally blind to whether the two HELD cells differ in topic. They can — the held cells are the
    phase-1 length-matched cells (unchanged), and on the thin length-adding pairs their seed mixes diverge.
    A base-activation probe encodes topic regardless of the diagonal, so a held topic gap can bias the
    metric the same way a held length gap can. This trains a one-hot seed-identity classifier to separate
    the two held cells and reports: `insample` (an UPPER BOUND — it memorizes seeds unique to one cell, so
    it runs high on thin pairs with many singleton seeds), `cv` (the generalizable separation; compare to
    `chance_ceiling`), and `seed_cosine` (per-seed count cosine; 1.0 = identical topic mix). Real-probe
    caveat: a probe reads text, not the seed string, and same-seed texts differ by the features, so this
    OVERSTATES the achievable topic bias."""
    from collections import Counter
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    hp, hn = selected["held_f1pos"], selected["held_f1neg"]
    X = DictVectorizer().fit_transform([{r["seed"]: 1} for r in hp] + [{r["seed"]: 1} for r in hn])
    y = np.array([0] * len(hp) + [1] * len(hn))
    clf = LogisticRegression(C=1.0, max_iter=2000)
    folds = max(2, min(cv, len(hp), len(hn)))
    cp, cn = Counter(r["seed"] for r in hp), Counter(r["seed"] for r in hn)
    seeds = set(cp) | set(cn)
    vp = np.array([cp.get(s, 0) for s in seeds], float)
    vn = np.array([cn.get(s, 0) for s in seeds], float)
    cos = float(vp @ vn / (np.linalg.norm(vp) * np.linalg.norm(vn) + 1e-9))
    return {"insample": float(clf.fit(X, y).score(X, y)),
            "cv": float(cross_val_score(clf, X, y, cv=folds, scoring="accuracy").mean()),
            "seed_cosine": cos, "n": int(len(y))}


def permutation_ceiling(selected: dict[str, list[dict]], which: str, n_perm: int = 200,
                        cv: int = 5, pct: float = 95.0, seed: int = 0) -> float:
    """The ~95th-percentile null accuracy of the held-vs-held `which`('length'|'topic') CV classifier under
    label permutation — the correct chance ceiling at this pair's N and feature distribution (a normal
    approximation under-estimates the spread, because cross-validation adds fold-to-fold variance, so it
    over-flags full-N pairs). cv accuracy above this ceiling is a real, generalizing held-cell separation.
    Deterministic (fixed seed)."""
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    hp, hn = selected["held_f1pos"], selected["held_f1neg"]
    if min(len(hp), len(hn)) < 5:  # see held_length_separability: tiny selections just fail the gate
        return 0.5
    y = np.array([0] * len(hp) + [1] * len(hn))
    if which == "length":
        X = np.array([[r["num_tokens"]] for r in hp] + [[r["num_tokens"]] for r in hn], float)
    else:
        X = DictVectorizer().fit_transform([{r["seed"]: 1} for r in hp] + [{r["seed"]: 1} for r in hn])
    folds = max(2, min(cv, len(hp), len(hn)))
    rng = np.random.default_rng(seed)
    accs = [cross_val_score(LogisticRegression(C=1.0, max_iter=2000), X, rng.permutation(y),
                            cv=folds, scoring="accuracy").mean() for _ in range(n_perm)]
    return float(np.percentile(accs, pct))


def _report(selected: dict[str, list[dict]], extra: dict) -> dict:
    stats = {c: _stats(selected[c]) for c in ALL}
    meds = {c: stats[c]["median"] for c in ALL}
    sep = length_separability(selected)
    lo, hi = extra.get("gate", (0.45, 0.55))
    return {
        "per_cell": stats, "min_n": min(stats[c]["n"] for c in ALL),
        "held_n": stats["held_f1pos"]["n"], "ab_n": stats["A"]["n"],
        "max_median_gap": max(meds.values()) - min(meds.values()),
        "held_median_gap": abs(meds["held_f1pos"] - meds["held_f1neg"]),
        "ab_median_gap": abs(meds["A"] - meds["B"]),
        "length_sep": sep,
        "passes_gate": bool(lo <= sep["held_hard_acc"] <= hi and lo <= sep["held_bounded"] <= hi),
        **extra,
    }


def match_pair(grid: dict[str, list[dict]], target: int = 300, bin_width_held: int = 8,
               bin_width_ab: int = 12, floor: int = 250) -> tuple[dict[str, list[dict]], dict]:
    """Match the held-out pair at `bin_width_held` and the A/B pair at `bin_width_ab` (both
    identical-histogram). Returns (selected, report)."""
    all_tokens = np.array([r["num_tokens"] for c in ALL for r in grid[c]])
    sel_held = _match_group(grid, HELD, all_tokens, bin_width_held, target)
    sel_ab = _match_group(grid, AB, all_tokens, bin_width_ab, target)
    selected = {**sel_ab, **sel_held}
    return selected, _report(selected, {"bin_width_held": bin_width_held, "bin_width_ab": bin_width_ab})


def match_pair_tuned(grid: dict[str, list[dict]], target: int = 300, floor: int = 250,
                     held_bins=(4, 5, 6, 8, 10, 12, 16), ab_bins=(8, 10, 12, 16, 20, 25, 30, 40),
                     gate=(0.45, 0.55), ab_acc_max: float = 0.56,
                     held_level_key=None, prefer_key=None,
                     held_gate: str = "legacy") -> tuple[dict[str, list[dict]], dict]:
    """Pick, per group, the COARSEST bin width that still defeats a linear length classifier, then
    take the matched selection that retains the most. Held group: keep the held-out length gate inside
    `gate`. A/B group: keep the A/B length train-accuracy <= `ab_acc_max`. Among passing widths choose
    the largest retained count. The matcher validates itself (report['length_sep'], ['passes_gate']).

    `held_level_key` / `prefer_key` are the stage 6 selection-policy hooks (see _match_group);
    held_level_key applies to the held group only (the metric cells the marker check binds on).

    `held_gate` (stage 6): the internal held-bin gate scores candidate held selections with the
    MATCHED (deliberately de-fanged, train-acc <= ab_acc_max) A/B length classifier; that model's held
    HARD-sign readout is largely noise around tiny residual asymmetries, and requiring it in-band
    rejects good large selections (measured: held_n 42-87 where ~400 can match, with bounded ~0.501
    throughout). "bounded" gates the bin choice on the criteria that actually bind downstream
    (exactly analyze_confounds' final checks): the de-fanged BOUNDED readout in band, the BOUNDED
    exposure under a classifier trained on the RAW (full, unmatched) buffer A/B — the same steep
    classifier the final gate uses, which amplifies systematic within-bin offsets the other two
    metrics under-detect — and the DIRECT held-vs-held length cv at/below its own permutation
    ceiling. "legacy" (default) = the original two-metric behavior, byte-identical for the
    original-32 path."""
    assert held_gate in ("legacy", "bounded")
    lo, hi = gate
    all_tokens = np.array([r["num_tokens"] for c in ALL for r in grid[c]])
    raw_ab_clf = None
    if held_gate == "bounded":  # the final-gate exposure classifier: trained on the RAW buffer A/B
        from sklearn.linear_model import LogisticRegression
        Xr = np.array([[r["num_tokens"]] for c in AB for r in grid[c]], float)
        yr = np.array([0] * len(grid["A"]) + [1] * len(grid["B"]))
        if len(set(yr.tolist())) == 2:
            raw_ab_clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xr, yr)
    ref_held = _match_group(grid, HELD, all_tokens, held_bins[0], target,
                            level_key=held_level_key, prefer_key=prefer_key)  # only used for the (held-independent) A/B train-acc

    # 1) A/B: coarsest bin keeping the A/B length train-accuracy <= ab_acc_max (train-acc uses only the
    #    A/B cells, so it does not depend on the held selection), maximizing retained A/B count.
    best_ab, best_ab_w, best_ab_n = None, ab_bins[0], -1
    for w in sorted(ab_bins, reverse=True):
        sel = _match_group(grid, AB, all_tokens, w, target, prefer_key=prefer_key)
        if length_separability({**sel, **ref_held})["ab_train_acc"] <= ab_acc_max and len(sel["A"]) > best_ab_n:
            best_ab, best_ab_w, best_ab_n = sel, w, len(sel["A"])
    if best_ab is None:
        best_ab, best_ab_w = _match_group(grid, AB, all_tokens, ab_bins[0], target, prefer_key=prefer_key), ab_bins[0]

    # 2) held: coarsest bin where the §6 gate (length classifier trained on the ACTUAL chosen A/B, scored
    #    on held) sits in band, maximizing retained held count. Tuning against the real A/B (not a fixed
    #    reference) is what makes the FINAL gate pass.
    def raw_exposure_bounded(sel_held):
        """The final gate's exposure: the RAW-A/B length classifier scored on the candidate held
        selection with the frozen metric (bounded)."""
        import metric
        hp, hn = sel_held["held_f1pos"], sel_held["held_f1neg"]
        if raw_ab_clf is None or min(len(hp), len(hn)) < 5:
            return 1.0  # degenerate: fail
        Xho = np.array([[r["num_tokens"]] for r in hp] + [[r["num_tokens"]] for r in hn], float)
        signs = np.array([1] * len(hp) + [-1] * len(hn))
        d = -raw_ab_clf.decision_function(Xho)
        return metric.logprob_difference(d, signs).bounded_prob

    def held_gate_ok(sel_held):
        s = length_separability({**best_ab, **sel_held})
        if held_gate == "bounded":
            if not (lo <= s["held_bounded"] <= hi):
                return False
            if not (lo <= raw_exposure_bounded(sel_held) <= hi):
                return False
            direct = held_length_separability(sel_held)
            return direct["cv"] <= permutation_ceiling(sel_held, "length")
        return lo <= s["held_hard_acc"] <= hi and lo <= s["held_bounded"] <= hi

    best_held, best_held_w, best_held_n = None, held_bins[0], -1
    for w in sorted(held_bins, reverse=True):
        sel = _match_group(grid, HELD, all_tokens, w, target, level_key=held_level_key, prefer_key=prefer_key)
        if held_gate_ok(sel) and len(sel["held_f1pos"]) > best_held_n:
            best_held, best_held_w, best_held_n = sel, w, len(sel["held_f1pos"])
    if best_held is None:  # nothing passed: take the finest (tightest) bin
        best_held = _match_group(grid, HELD, all_tokens, held_bins[0], target,
                                 level_key=held_level_key, prefer_key=prefer_key)
        best_held_w = held_bins[0]

    selected = {**best_ab, **best_held}
    rep = _report(selected, {"bin_width_held": best_held_w, "bin_width_ab": best_ab_w, "gate": gate,
                             "held_gate": held_gate})
    if held_gate == "bounded":  # the pass/fail verdict must match the gate actually applied
        direct = held_length_separability(best_held)
        ceil = permutation_ceiling(best_held, "length")
        raw_exp = raw_exposure_bounded(best_held)
        rep["held_direct_cv"], rep["held_direct_ceiling"] = direct["cv"], ceil
        rep["raw_exposure_bounded"] = raw_exp
        rep["passes_gate"] = bool(lo <= rep["length_sep"]["held_bounded"] <= hi
                                  and lo <= raw_exp <= hi and direct["cv"] <= ceil)
    return selected, rep


def seed_composition_overlap(selected: dict[str, list[dict]]) -> float:
    """Min over cell-pairs of cosine similarity between per-seed count vectors — a check that topic
    (seed) composition stays comparable across the four cells after length sub-sampling (§5)."""
    return min(seed_overlap_matrix(selected).values(), default=1.0)


def seed_overlap_matrix(selected: dict[str, list[dict]]) -> dict[tuple[str, str], float]:
    """All pairwise cosine similarities between the four cells' per-seed count vectors. The A-B entry is
    the binding topic check for the metric (a classifier trained on the A/B diagonal can only use topic
    as an A-vs-B cue if A and B differ in topic), so the full_balanced selection drives A-B to ~1.0."""
    from collections import Counter

    all_seeds = sorted({r["seed"] for c in ALL for r in selected[c]})
    vecs = {}
    for c in ALL:
        cnt = Counter(r["seed"] for r in selected[c])
        v = np.array([cnt.get(s, 0) for s in all_seeds], dtype=float)
        vecs[c] = v / (np.linalg.norm(v) + 1e-9)
    return {(a, b): float(vecs[a] @ vecs[b]) for i, a in enumerate(ALL) for b in ALL[i + 1:]}


def _dedup_within_seed(recs: list[dict], dedup_view=None) -> list[dict]:
    """Greedy within-(cell, seed) near-duplicate removal on the normalized-prefix difflib ratio — the
    project's accept-time standard (config.DEDUP_RATIO_THRESHOLD on the first 200 chars). Re-applied here
    because the accept-time dedup ran on the RAW text and layout normalization can make two prefixes
    collide (the §4.3 safety net found ~32 such pairs — mostly the generator's same-topic near-repeats).
    `recs` is already in stable-rank order, so the lowest-rank text of a near-dup group is the one kept,
    which keeps the selection deterministic. k per seed is small, so this O(k^2) is cheap.

    dedup_view (None = original behavior): view of the text used for the comparison only — the
    extension starts_with + cells strip their pinned shared prefix (see generation.generate_cell_props)."""
    import data_checks

    kept, kept_norm = [], []
    for r in recs:
        dedup_text = dedup_view(r["text"]) if dedup_view else r["text"]
        if not data_checks.is_near_duplicate(dedup_text, kept_norm):
            kept.append(r)
            kept_norm.append(data_checks.normalized_prefix(dedup_text))
    return kept


def _seed_groups(recs: list[dict], dedup: bool = True, prefer_key=None, dedup_view=None) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in recs:
        groups.setdefault(r["seed"], []).append(r)
    for s in groups:
        groups[s].sort(key=lambda r: _stable_rank(r["text"]))
        if dedup:
            groups[s] = _dedup_within_seed(groups[s], dedup_view=dedup_view)
        if prefer_key:  # de-prioritize (stable) AFTER dedup, so the dedup keeps the same stable-rank winner
            groups[s].sort(key=lambda r: (prefer_key(r), _stable_rank(r["text"])))
    return groups


def seed_stratified_balanced(grid: dict[str, list[dict]], target: int = 300,
                             dedup: bool = True, prefer_key=None, dedup_view=None) -> tuple[dict[str, list[dict]], dict]:
    """Select `target` texts EACH from cells A and B with IDENTICAL per-seed counts, deterministically.

    This is the full_balanced training-cell selection (INSTRUCTIONS §4): unlike `match_pair_tuned` it does
    NOT length-match A to B (that needlessly threw away most of the training data on length-adding pairs),
    so it recovers the full 300/class. It only balances TOPIC: A and B draw from the same seed pool, and
    matching per-seed counts means topic carries zero information about the A-vs-B label, so a probe /
    fine-tune cannot use topic as the class cue. (The diagonal length cue is handled downstream at the
    probe/fine-tune readout — see the write-up §7.) Held cells are untouched by this function.

    Allocation is water-filled toward a uniform per-seed count (capped at min(|A_s|, |B_s|) so A and B stay
    identical per seed), which maximizes topic spread (no seed exceeds ceil(target / n_seeds)). Within a
    seed the lowest-stable-sha256 texts are taken, so the exact selection reproduces. If
    sum_s min(|A_s|, |B_s|) < target the max balanced count is taken (documented in `info`).

    `prefer_key(record) -> sortable` (stage 6 selection policies, default None = original behavior):
    within-seed order becomes (prefer, stable_rank), so dispreferred records (e.g. an over-concentrated
    entity) are drawn only when a seed's preferred pool runs out. Allocation (per-seed counts) is
    unchanged, so A/B stay identical per seed."""
    A_by = _seed_groups(grid["A"], dedup=dedup, prefer_key=prefer_key, dedup_view=dedup_view)
    B_by = _seed_groups(grid["B"], dedup=dedup, prefer_key=prefer_key, dedup_view=dedup_view)
    seeds = sorted(set(A_by) | set(B_by), key=_stable_rank)  # hash order: deterministic, no alphabetical bias
    cap = {s: min(len(A_by.get(s, [])), len(B_by.get(s, []))) for s in seeds}
    goal = min(target, sum(cap.values()))

    alloc = {s: 0 for s in seeds}
    remaining = goal
    while remaining > 0:
        active = [s for s in seeds if alloc[s] < cap[s]]
        if not active:
            break
        for s in active:
            if remaining <= 0:
                break
            alloc[s] += 1
            remaining -= 1

    selected = {
        "A": [r for s in seeds for r in A_by.get(s, [])[:alloc[s]]],
        "B": [r for s in seeds for r in B_by.get(s, [])[:alloc[s]]],
    }
    info = {"target": target, "achieved": goal, "reached_target": goal >= target,
            "n_seeds_total": len(seeds), "n_seeds_used": sum(1 for s in seeds if alloc[s] > 0),
            "max_per_seed": max(alloc.values(), default=0), "total_cap": sum(cap.values())}
    return selected, info


# --------------------------------------------------------------------- shared commit flow
# Margin above the topic permutation ceiling before the held selection counts as topic-separable
# (the ceiling itself has ~0.01-0.02 permutation noise; only a MEANINGFUL gap should trigger the
# joint fallback / QA flag, e.g. the temporal transcript tracer at 0.85 vs a 0.57 ceiling, not a
# marginal 0.56-vs-0.558 crossing). Shared with v2_build's qa_flags.
TOPIC_MARGIN = 0.05
# Bin widths allowed under the JOINT (length_bin, seed) held match: identical joint histograms
# make the length marginals identical at ANY width, so much coarser bins are safe to try (the
# bounded gate still rejects widths with a generalizing within-bin length offset) and coarser
# bins retain far more under the stricter joint cap.
JOINT_HELD_BINS = (4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64)


def held_seed_level_key(record: dict) -> str:
    return record["seed"]


def commit_grid_flow(pair_key: str, grid: dict[str, list[dict]], target: int,
                     dedup_view=None, write: bool = True, grid_dir=None) -> tuple[dict, dict]:
    """The standing commit flow every new-pair builder uses (extension / content / new x new / v2):
    held cells matched to an identical token-length histogram at equal count (bounded held gate);
    A/B full-target, seed-stratified balanced, NOT length-matched; save to the grid dir; return
    (committed_cells, per-pair report). `grid` must already be layout-normalized + re-tokenized.

    Topic-balance fallback (stage 1 machinery upgrade): if the length-only held selection is
    topic-separable (seed-identity cv above its permutation ceiling + TOPIC_MARGIN — the same
    criterion as v2_build's qa_flags gate), the held cells are re-matched on a JOINT
    (length_bin, seed) histogram, which drives the seed classifier to chance BY CONSTRUCTION
    (identical per-seed counts). The joint match retains fewer texts (10-35% on well-behaved
    buffers, much more on divergent-length transcript pairs), so it only runs where the gate
    fires; report["held_matching_policy"] records which path committed."""
    import dataset

    selected_all, tuned_rep = match_pair_tuned(grid, target=target, held_gate="bounded")
    held = {c: selected_all[c] for c in HELD}
    policy = {"policy": "length_only"}
    topic_cv = held_topic_separability(held)["cv"]
    topic_ceil = permutation_ceiling(held, "topic")
    if topic_cv > topic_ceil + TOPIC_MARGIN:
        policy = {"policy": "joint_length_x_seed",
                  "length_only_rejected": {"topic_cv": topic_cv, "topic_ceiling": topic_ceil,
                                           "held_n": len(held["held_f1pos"])}}
        selected_all, tuned_rep = match_pair_tuned(grid, target=target, held_gate="bounded",
                                                   held_level_key=held_seed_level_key,
                                                   held_bins=JOINT_HELD_BINS)
        held = {c: selected_all[c] for c in HELD}
    sel_ab, ab_info = seed_stratified_balanced({c: grid[c] for c in AB}, target=target,
                                               dedup_view=dedup_view)
    committed = {**sel_ab, **held}

    if write:
        for c in ALL:
            dataset.save_cell(committed[c], pair_key, c,
                              base_dir=grid_dir if grid_dir is not None else dataset.GRID_DIR)

    sep = length_separability(committed)
    report = {
        "pair": pair_key,
        "n_per_cell": {c: len(committed[c]) for c in ALL},
        "held_matching_policy": policy,
        "held_bin_width": tuned_rep["bin_width_held"],
        "held_gate": {k: tuned_rep[k] for k in
                      ("passes_gate", "held_direct_cv", "held_direct_ceiling", "raw_exposure_bounded")
                      if k in tuned_rep},
        "ab_select_info": ab_info,
        "held_vs_held_length": held_length_separability(committed),
        "held_length_perm_ceiling": permutation_ceiling(committed, "length"),
        "held_vs_held_topic": held_topic_separability(committed),
        "held_topic_perm_ceiling": permutation_ceiling(committed, "topic"),
        "full_ab_length_baseline": {"hard": sep["held_hard_acc"], "bounded": sep["held_bounded"],
                                    "ab_train_acc": sep["ab_train_acc"]},
        "seed_overlap": {f"{a}-{b}": v for (a, b), v in seed_overlap_matrix(committed).items()},
        "per_cell_tokens": {c: _stats(committed[c]) for c in ALL},
    }
    return committed, report
