"""The logprob difference (INSTRUCTIONS §6) — defined once, reused by probe, fine-tune,
baseline, and every later segment.

For each held-out text take the model's log-odds toward class A, ``d = logP(A) - logP(B)``
(for the fine-tune/baseline the difference of the two label-token logits; for the probe the
logistic-regression decision value ``w·x + b``). Orient it toward f1:

  * on a ``(f1+, f2-)`` text use ``+d`` (f1 points to A),
  * on a ``(f1-, f2+)`` text use ``-d`` (f1 points to B).

Average the oriented values over all held-out texts -> the **logprob difference**: positive
follows f1, negative follows f2, 0 ambivalent. Unbounded (nats); no rescaling, no denominator.

Orientation is fixed by a single global ordering of the properties, so each pair is oriented
toward the earlier property and ``Δlogp(f2, f1) = −Δlogp(f1, f2)``.

A bounded-probability version and a hard-accuracy (argmax) version are computed as side checks.
A K-class entry point (used by a later segment) reduces to the binary metric at K=2.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from attrs import frozen


@frozen
class LogprobDifference:
    """Result of the metric on one set of held-out texts.

    raw is the primary, unbounded statistic (nats). bounded_prob is the mean probability of
    following f1 (in [0,1]; 0.5 = ambivalent). hard_accuracy is the fraction whose argmax
    prediction follows f1 (in [0,1]; 0.5 = chance). n is the held-out text count.
    """

    raw: float
    bounded_prob: float
    hard_accuracy: float
    n: int

    def follows_f1(self) -> bool:
        return self.raw > 0


def _summarize(oriented: np.ndarray) -> LogprobDifference:
    """All three statistics are functions of the oriented log-odds ``oriented = d * sign``."""
    oriented = np.asarray(oriented, dtype=np.float64)
    assert oriented.ndim == 1 and oriented.size > 0, "need a non-empty 1-D array of oriented values"
    return LogprobDifference(
        raw=float(oriented.mean()),
        # sigmoid(oriented) = P(follow f1): sigmoid(+d)=P(A) on (f1+,f2-), sigmoid(-d)=P(B) on (f1-,f2+).
        bounded_prob=float(_sigmoid(oriented).mean()),
        # argmax follows f1 iff oriented > 0 in both held-out cells.
        hard_accuracy=float((oriented > 0).mean()),
        n=int(oriented.size),
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logprob_difference(decision_values: Sequence[float], f1_value_signs: Sequence[int]) -> LogprobDifference:
    """Binary metric (the primary entry point).

    decision_values[i] = ``d = logP(A) - logP(B)`` for held-out text i.
    f1_value_signs[i]  = +1 if text i has the f1 feature at its + value, -1 if at its - value.

    Because held-out texts have f1 and f2 disagreeing, ``f1_value_sign = -f2_value_sign``; the
    oriented value is ``d * f1_value_sign`` (positive when the prediction follows f1). Swapping
    which property is f1 flips every sign, giving ``Δlogp(f2,f1) = −Δlogp(f1,f2)``.
    """
    d = np.asarray(decision_values, dtype=np.float64)
    s = np.asarray(f1_value_signs, dtype=np.float64)
    assert d.shape == s.shape, "decision_values and f1_value_signs must align"
    assert set(np.unique(s)).issubset({-1.0, 1.0}), "f1_value_signs must be +1 or -1"
    return _summarize(d * s)


def kclass_logprob_difference(
    class_logits: np.ndarray,
    f1_class: Sequence[int],
    f2_class: Sequence[int],
) -> LogprobDifference:
    """K-class generalization (used by the later multi-class segment; reduces to the binary
    metric at K=2). For held-out text i, ``f1_class[i]`` is the class f1 points to and
    ``f2_class[i]`` the class f2 points to (they differ off-diagonal). The oriented log-odds is
    ``logit[f1_class] - logit[f2_class]`` (positive when the model prefers f1's class over f2's).

    class_logits has shape (n_texts, K). Softmax normalization cancels in the difference, so raw
    logits or log-probs both work. bounded/accuracy are pairwise (f1's class vs f2's class).
    """
    class_logits = np.asarray(class_logits, dtype=np.float64)
    i = np.asarray(f1_class, dtype=int)
    j = np.asarray(f2_class, dtype=int)
    assert class_logits.ndim == 2 and i.shape == j.shape == (class_logits.shape[0],)
    rows = np.arange(class_logits.shape[0])
    oriented = class_logits[rows, i] - class_logits[rows, j]
    return _summarize(oriented)


def order_pair(prop_a: str, prop_b: str, global_order: Sequence[str]) -> tuple[str, str, bool]:
    """Return (f1, f2, swapped): f1 is whichever of the two properties comes earlier in the fixed
    global ordering. ``swapped`` is True iff the inputs were given in the opposite order. This is
    what guarantees a consistent orientation across pairs (INSTRUCTIONS §6)."""
    order = list(global_order)
    assert prop_a in order and prop_b in order, f"{prop_a},{prop_b} must be in global_order {order}"
    assert prop_a != prop_b
    if order.index(prop_a) < order.index(prop_b):
        return prop_a, prop_b, False
    return prop_b, prop_a, True
