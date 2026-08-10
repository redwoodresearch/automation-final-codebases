"""The extension project's ANCHOR set: all shipped pairs among the fixed 8-partner subset of the
published 21-feature roster. These re-derive the published strength ladder in-run (the repo ships
grids but no numeric results) and put anchors + new features on one fitted scale.

27 pairs = 20 partner×partner (28 combinatorial minus code_presence__numerical_density, dropped for
buildability — see pairs_expanded.DROPPED_PAIRS) + 7 language__<partner> pairs. Orientation follows
the shipped grids exactly: partner×partner pairs by GLOBAL_PROPERTY_ORDER (earlier = f1); language
pairs have language = f1 (the shipped language__<partner> naming; + = Spanish).
"""

from __future__ import annotations

from attrs import frozen

from pairs_expanded import DROPPED_PAIRS, PAIRS_EXPANDED
from properties import GLOBAL_PROPERTY_ORDER

# The fixed 8-partner subset (canonical design). Every new (extension) feature is paired against
# these; the anchors below are all shipped pairs AMONG them.
ANCHOR_PARTNERS = [
    "language",
    "formal_register",
    "hedging",
    "emotional_tone",
    "code_presence",
    "question_vs_statement",
    "numerical_density",
    "passive_voice",
]

# Published reference fine-tune ordering for these partners (strongest first) — the anchor
# reproduction gate compares against this. Braces: {emotional_tone, code_presence} and
# {passive_voice, numerical_density} are published near-ties (swaps tolerated), and the
# ~1.3-nat question_vs_statement <-> hedging swap is also tolerated.
REFERENCE_FT_ORDER = [
    "language",
    "formal_register",
    "question_vs_statement",
    "hedging",
    "emotional_tone",
    "code_presence",
    "passive_voice",
    "numerical_density",
]
TOLERATED_SWAP_GROUPS = [
    {"question_vs_statement", "hedging"},
    {"emotional_tone", "code_presence"},
    {"passive_voice", "numerical_density"},
]


@frozen
class AnchorPair:
    key: str  # grid file prefix, "<f1>__<f2>"
    f1: str
    f2: str


def anchor_pairs() -> dict[str, AnchorPair]:
    """The 27 anchor pairs, keyed by grid-file prefix. Asserts the expected composition."""
    roster_partners = [p for p in ANCHOR_PARTNERS if p != "language"]
    order = GLOBAL_PROPERTY_ORDER.index
    pairs: dict[str, AnchorPair] = {}

    for i, a in enumerate(roster_partners):
        for b in roster_partners[i + 1:]:
            f1, f2 = sorted((a, b), key=order)
            key = f"{f1}__{f2}"
            if key in DROPPED_PAIRS:
                assert key == "code_presence__numerical_density", key
                continue  # the documented buildability drop — the missing 28th pair
            assert key in PAIRS_EXPANDED, f"{key} not in the shipped expanded pair list"
            pairs[key] = AnchorPair(key=key, f1=f1, f2=f2)

    for partner in roster_partners:  # shipped orientation: language = f1, + = Spanish
        key = f"language__{partner}"
        pairs[key] = AnchorPair(key=key, f1="language", f2=partner)

    assert len(pairs) == 27, f"expected 27 anchor pairs, got {len(pairs)}"
    return pairs


# The ~4-pair subset for the full 5-LR x 3-seed sweep (validating the pinned-LR economy), spanning
# the reference ladder's strength-gap strata: a published near-tie brace pair, the tolerated-swap
# small-gap pair, a mid-gap language pair, and the well-separated pair whose published gap
# (s(hedging) - s(passive_voice) ~ +16 nats) the gate also checks.
LR_SWEEP_PAIRS = [
    "code_presence__emotional_tone",     # near-tie (published brace)
    "question_vs_statement__hedging",    # small gap (~1.3 nats published, swap tolerated)
    "language__hedging",                 # mid gap
    "hedging__passive_voice",            # well-separated (~16 nats published)
]


if __name__ == "__main__":
    ap = anchor_pairs()
    print(f"{len(ap)} anchor pairs:")
    for k, p in ap.items():
        print(f"  {k:50s} f1={p.f1}")
    assert all(k in ap for k in LR_SWEEP_PAIRS)
