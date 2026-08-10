"""Deterministic surface-marker counters for the cross-property marker check (desiderata, stage 6
expansion audit d) and the selection-time marker fixes.

8 of the 20 roster properties have deterministically countable markers. A THIRD property's marker
separating a pair's held cells is a real neither-f1-nor-f2 confound (measured in the side data:
acronym-glossing parentheses separate ×acronym held cells at ~0.65-0.87). This module is the single
source of truth for those counters (reusing validation.py's accept-time rules so the audit measures
exactly what the pipeline gates on) plus the quantized "level" used when a held-cell selection must
match a marker's distribution across cells.
"""
from __future__ import annotations

import validation

# property name -> scalar counter over the raw text. Mirrors the accept-time SURFACE_RULES inputs.
MARKER_COUNTERS = {
    "second_person_address": validation.count_second_person,
    "first_person_voice": validation.count_first_person,
    "negation_density": validation.count_negations,
    "acronym_density": validation.count_acronyms,
    "parenthetical_asides": lambda t: t.count("("),
    "numerical_density": validation.count_number_figures,
    "question_vs_statement": validation.question_fraction,
    "code_presence": lambda t: 1.0 if validation.has_code(t) else 0.0,
}

MARKER_ORDER = list(MARKER_COUNTERS)


def marker_level(prop: str, text: str) -> int:
    """Quantized marker level for selection-time distribution matching (0 / 1 / 2 / 3+ for counts;
    {0,1} for code; quartile-ish buckets for the question fraction). Coarse on purpose: held cells are
    matched on (length_bin, level), so finer levels shrink the matched pool faster than they tighten
    the confound."""
    v = MARKER_COUNTERS[prop](text)
    if prop == "code_presence":
        return int(v > 0)
    if prop == "question_vs_statement":
        return min(3, int(v * 4))  # fraction -> 0/1/2/3
    return min(3, int(v))


def marker_bit(prop: str, text: str) -> int:
    """Presence bit — the coarsest match granularity (e.g. 'gloss-free acronym texts': equalize
    WHETHER a parenthesis appears, not how many). Used when the level-granularity joint match costs
    too much pool; equalizing presence removes most of a presence-driven separation."""
    return int(MARKER_COUNTERS[prop](text) > 0)


def pair_props(pair_key: str) -> tuple[str, str]:
    f1, f2 = pair_key.split("__")
    return f1, f2


def third_party_markers(pair_key: str) -> list[str]:
    """The countable properties that are NEITHER f1 nor f2 of this pair — the ones whose marker
    separating the held cells would be a real confound."""
    f1, f2 = pair_props(pair_key)
    return [p for p in MARKER_ORDER if p not in (f1, f2)]
