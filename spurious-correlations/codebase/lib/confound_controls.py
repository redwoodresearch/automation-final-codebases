"""Reimplemented confound-control battery for extension pairs (the published study's control
battery entry scripts did not ship; lib/artifact_battery.py is the unrelated layer-robustness
battery). Each control trains a classifier that sees ONLY a nuisance representation of the text and
asks whether it can reproduce the readout.

Two evaluations per control:
  * diagonal-trained: fit on A vs B (the exact exposure the probe/fine-tune has), score the held
    cells with the frozen metric — `held_bounded` near 0.5 means this nuisance channel cannot bias
    the readout (published battery range: 0.50–0.56);
  * held-vs-held: cross-validated accuracy separating held_f1pos from held_f1neg, compared to a
    label-permutation chance ceiling.

Controls (nuisance-only channels): token length; paragraph layout; topic (seed identity);
surface character statistics; bag-of-words over NON-FEATURE lexical content. For a trigger pair
the trigger string is stripped from the text before char-stat/BoW featurization (the trigger is
trivially a BoW feature — an unmodified BoW control would "fail" meaninglessly on every trigger
pair). For a STYLE partner (e.g. hedging) the partner's own lexical realization must also be
excluded from the artifact BoW — hedges like "perhaps" ARE the partner feature, not an artifact —
so the artifact-BoW additionally drops a fixed partner lexicon. The partner-visible BoW variants
are also computed, clearly labeled as descriptive lexical BASELINES (which feature wins in
bag-of-words space), not artifact controls.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import metric

# Fixed lexicons of feature-bearing words to EXCLUDE from the artifact BoW, per feature. Only
# needed for features whose realization is lexical; deterministic-trigger features use their
# pinned string instead (strip_strings). Hedging: the standard hedge/booster inventory the
# generation instructions themselves name, plus common variants.
FEATURE_LEXICONS: dict[str, tuple[str, ...]] = {
    "hedging": (
        "might", "may", "maybe", "perhaps", "possibly", "possible", "probably", "likely",
        "unlikely", "seems", "seem", "seemingly", "appears", "appear", "apparently", "suggest",
        "suggests", "could", "would", "arguably", "presumably", "roughly", "somewhat", "uncertain",
        "uncertainty", "unsure", "hesitant", "think", "believe", "suspect", "guess", "wonder",
        "certainly", "clearly", "definitely", "undoubtedly", "surely", "obviously", "certain",
        "certainty", "unquestionably", "plainly", "decidedly", "firmly", "confident",
    ),
}


def _strip_strings(text: str, strings: tuple[str, ...]) -> str:
    for s in strings:
        text = re.sub(re.escape(s), " ", text, flags=re.IGNORECASE)
    return text


def detriggered_text(r: dict, trigger_normalizer) -> str:
    """The de-triggered view used by the char-stat / BoW / trace channels. The SAME normalizer is
    applied to every record — on the exact pre-transform source where one is stored
    (`text_pretransform`, generate-then-transform cells), else on the text — so cells with and
    without a stored source stay comparable: on an all_caps pair both the + cells' raw source and
    the − cells' texts get lowercased. (Using the raw source verbatim would let its normal-vs-zero
    uppercase fraction separate the cells as a fake artifact; for leetspeak, normalizing the
    source also avoids the un-leet corruption of ordinary alphanumerics like "3rd" appearing on
    one side only.) NFC first: the stored source is the pre-NFC raw candidate."""
    return trigger_normalizer(unicodedata.normalize("NFC", r.get("text_pretransform") or r["text"]))


def char_stat_features(texts: list[str]) -> np.ndarray:
    rows = []
    for t in texts:
        n = max(1, len(t))
        words = re.findall(r"[A-Za-z']+", t)
        sents = [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
        rows.append([
            len(t),
            sum(c.isupper() for c in t) / n,
            sum(c.isdigit() for c in t) / n,
            *[t.count(c) / n for c in ".,;:?!()\"'"],
            np.mean([len(w) for w in words]) if words else 0.0,
            np.mean([len(s.split()) for s in sents]) if sents else 0.0,
            len(set(w.lower() for w in words)) / max(1, len(words)),  # type-token ratio
            t.count("\n") / n,
        ])
    return np.asarray(rows, dtype=float)


def _diag_train_held_score(Xtr, ytr, Xho, signs) -> dict:
    """The exposure evaluation: fit A(0)/B(1) on the diagonal, score held cells with the frozen
    metric (d = logP(A) − logP(B) = −decision_function)."""
    if len(set(ytr.tolist())) < 2:
        return {"held_bounded": 0.5, "held_raw": 0.0, "ab_train_acc": 0.5}
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr)
    res = metric.logprob_difference(-clf.decision_function(Xho), signs)
    return {"held_bounded": res.bounded_prob, "held_raw": res.raw,
            "ab_train_acc": float(clf.score(Xtr, ytr))}


def _held_cv(Xho, yho, n_perm: int = 200, cv: int = 5, seed: int = 0) -> dict:
    # SHUFFLED StratifiedKFold: run_battery concatenates cells in file order, whose topic-seed
    # ordering is partially aligned across the two cells, so sklearn's default CONTIGUOUS folds
    # create train/test topic imbalance that DEFLATES the measured accuracy (a fold-ordering
    # artifact, not signal). Shuffling removes it. Same splitter+seed used for acc and the
    # permutation ceiling so they stay comparable. (Fixed in attempt 5 after the overall research
    # review found the deflation understated off-hedging trigger-trace halos.)
    from sklearn.model_selection import StratifiedKFold

    folds = max(2, min(cv, int((yho == 0).sum()), int((yho == 1).sum())))
    clf = LogisticRegression(C=1.0, max_iter=2000)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    acc = float(cross_val_score(clf, Xho, yho, cv=splitter, scoring="accuracy").mean())
    rng = np.random.default_rng(seed)
    null = [cross_val_score(clf, Xho, rng.permutation(yho), cv=splitter, scoring="accuracy").mean()
            for _ in range(n_perm)]
    return {"cv": acc, "perm_ceiling_95": float(np.percentile(null, 95))}


def _bow_features(train_texts, held_texts, drop_words: tuple[str, ...]):
    vec = CountVectorizer(min_df=2, stop_words=list(drop_words) if drop_words else None)
    Xtr = vec.fit_transform(train_texts)
    return Xtr, vec.transform(held_texts), len(vec.vocabulary_)


def run_battery(grid: dict[str, list[dict]], trigger_strings: tuple[str, ...] = (),
                partner_lexicon: tuple[str, ...] = (), trigger_normalizer=None) -> dict:
    """The full battery on one committed grid (dataset.load_grid output).

    trigger_normalizer (None = strip trigger_strings, the original behavior): how to remove the
    trigger from a text before the char-stat / BoW / trace featurizations. Transform triggers
    (all_caps / leetspeak / emoji) need a real normalizer — for them character statistics ARE the
    trigger, so the plain string-strip would leave the trigger fully visible to those channels.
    The normalizer is applied uniformly to every record, on the exact pre-transform source where
    one is stored (see detriggered_text)."""
    tr_recs = grid["A"] + grid["B"]
    ho_recs = grid["held_f1pos"] + grid["held_f1neg"]
    ytr = np.array([0] * len(grid["A"]) + [1] * len(grid["B"]))
    yho = np.array([0] * len(grid["held_f1pos"]) + [1] * len(grid["held_f1neg"]))
    signs = np.array([1] * len(grid["held_f1pos"]) + [-1] * len(grid["held_f1neg"]))
    tr_texts = [r["text"] for r in tr_recs]
    ho_texts = [r["text"] for r in ho_recs]

    run_trace = bool(trigger_strings) or trigger_normalizer is not None
    if trigger_normalizer is None:
        def trigger_normalizer(t):  # noqa: E731 - default: strip the pinned strings
            return _strip_strings(t, trigger_strings)

    def detriggered(r) -> str:
        return detriggered_text(r, trigger_normalizer)

    out: dict[str, dict] = {}

    def add(name, Xtr, Xho):
        out[name] = {**_diag_train_held_score(Xtr, ytr, Xho, signs), **_held_cv(Xho, yho)}

    add("length", np.array([[r["num_tokens"]] for r in tr_recs], float),
        np.array([[r["num_tokens"]] for r in ho_recs], float))
    add("layout_paragraphs", np.array([[r["text"].count("\n\n") + 1] for r in tr_recs], float),
        np.array([[r["text"].count("\n\n") + 1] for r in ho_recs], float))
    # word count: reported because the frozen TOKEN window forces transform-trigger texts
    # (caps/leet) to carry fewer words at matched token length — a designed, documented
    # difference this channel makes visible (near chance for non-transform pairs).
    add("word_count", np.array([[len(r["text"].split())] for r in tr_recs], float),
        np.array([[len(r["text"].split())] for r in ho_recs], float))

    seeds = sorted({r["seed"] for r in tr_recs + ho_recs})
    s_idx = {s: i for i, s in enumerate(seeds)}

    def seed_onehot(recs):
        X = np.zeros((len(recs), len(seeds)))
        for i, r in enumerate(recs):
            X[i, s_idx[r["seed"]]] = 1
        return X

    add("topic_seed", seed_onehot(tr_recs), seed_onehot(ho_recs))
    detr_tr = [detriggered(r) for r in tr_recs]
    detr_ho = [detriggered(r) for r in ho_recs]
    add("char_stats_trigger_stripped", char_stat_features(detr_tr), char_stat_features(detr_ho))

    # ARTIFACT BoW: trigger removed AND partner lexicon dropped — only non-feature lexical content.
    Xtr, Xho, vocab = _bow_features(detr_tr, detr_ho, partner_lexicon)
    add("bow_artifact_control", Xtr, Xho)
    out["bow_artifact_control"]["vocab_size"] = vocab

    # Descriptive lexical BASELINES (not artifact controls): which feature wins in BoW space?
    Xtr, Xho, vocab = _bow_features(detr_tr, detr_ho, ())
    add("bow_baseline_trigger_excluded", Xtr, Xho)  # partner lexical signal visible
    out["bow_baseline_trigger_excluded"]["vocab_size"] = vocab
    Xtr, Xho, vocab = _bow_features(tr_texts, ho_texts, ())
    add("bow_baseline_full", Xtr, Xho)              # trigger visible too
    out["bow_baseline_full"]["vocab_size"] = vocab

    # TRIGGER-TRACE check (trigger pairs): with the trigger removed (string stripped / exact
    # pre-transform source / normalizer), can BoW separate cells that share the PARTNER pole and
    # differ only in trigger presence (A vs held_f1pos; B vs held_f1neg)? Above-ceiling separation
    # = the trigger INSTRUCTION (or word-target difference) shifts the text distribution beyond
    # the trigger itself — a construct-purity warning for the trigger feature.
    if run_trace:
        for name, cell_a, cell_b in [("trigger_trace_pluspole_AvsHeldpos", "A", "held_f1pos"),
                                     ("trigger_trace_minuspole_BvsHeldneg", "B", "held_f1neg")]:
            ta = [detriggered(r) for r in grid[cell_a]]
            tb = [detriggered(r) for r in grid[cell_b]]
            vec = CountVectorizer(min_df=2)
            X = vec.fit_transform(ta + tb)
            y = np.array([0] * len(ta) + [1] * len(tb))
            out[name] = _held_cv(X, y)
            out[name]["vocab_size"] = len(vec.vocabulary_)
            # words-only baseline: how much of the trace is just the (designed) word-count
            # difference between the trigger instruction / word-target variants?
            Xw = np.array([[len(t.split())] for t in ta + tb], float)
            out[name + "_wordcount_only"] = _held_cv(Xw, y)

    return out
