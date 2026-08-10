"""Canonical registry of all 37 features of the study, treated identically.

There is no "prior study" vs "extension" distinction: this is the single, ordered list every
part of the pipeline enumerates. The historical 21-feature roster is exposed only as a labelled
subset (`PRIOR_21`) for reporting the continuity number alongside the 37-feature refit.
"""
from __future__ import annotations

from properties import GLOBAL_PROPERTY_ORDER
from extension_properties import EXTENSION_ORDER

# The 21 original-roster features (20 frozen properties + language) and the 16 newer features.
PRIOR_21: list[str] = ["language"] + list(GLOBAL_PROPERTY_ORDER)
_NEWER_16: list[str] = list(EXTENSION_ORDER)

# The one canonical roster: all 37 features, one order, no split.
ALL_FEATURES: list[str] = PRIOR_21 + _NEWER_16
assert len(ALL_FEATURES) == 37, len(ALL_FEATURES)
assert len(set(ALL_FEATURES)) == 37, "duplicate feature name"

# The FROZEN five-type taxonomy (v2, stage 0) for the feature-type strength analysis.
# Assigned from feature DEFINITIONS only, BEFORE the 37-feature refit exists; per-feature
# rationale + borderline notes: results/complete_graph/feature_taxonomy.md. Near-duplicates are
# deliberately same-type (first/second person -> style_voice; acronyms/jargon -> style_voice).
# Do not revise after refit results exist without explicit justification + a sensitivity check.
FEATURE_TYPES = ("surface_mechanical", "style_voice", "discourse_structure", "content_behavior",
                 "triggers")
_TYPE_MAP: dict[str, str] = {
    # triggers: the 9 deterministic backdoor-style marks (planted strings/formats/transforms)
    **{f: "triggers" for f in (
        "substring_common_word", "substring_rare_wordlike", "substring_high_entropy",
        "starts_with_string", "ends_with_string", "all_caps", "leetspeak",
        "trailing_signature", "specific_emoji")},
    # content_behavior: WHAT the text asserts/does/is about
    **{f: "content_behavior" for f in (
        "deception", "harmful_vs_benign", "sycophantic_agreement", "refusal_language",
        "topic_medical", "ai_self_reference", "entity_mention_trigger")},
    # surface_mechanical: countable densities of meaning-independent marks
    **{f: "surface_mechanical" for f in (
        "code_presence", "numerical_density", "parenthetical_asides", "negation_density")},
    # style_voice: pervasive qualities of HOW the prose is written
    **{f: "style_voice" for f in (
        "language", "formal_register", "emotional_tone", "figurative_language", "hedging",
        "first_person_voice", "second_person_address", "passive_voice", "sentence_complexity",
        "domain_jargon", "acronym_density")},
    # discourse_structure: the organization/form of the passage's presentation
    **{f: "discourse_structure" for f in (
        "question_vs_statement", "imperative_mood", "narrative_vs_expository",
        "argumentative_structure", "specificity", "temporal_scope")},
}
assert set(_TYPE_MAP) == set(ALL_FEATURES), set(_TYPE_MAP) ^ set(ALL_FEATURES)


def feature_type(f: str) -> str:
    """The frozen five-type taxonomy: surface_mechanical | style_voice | discourse_structure |
    content_behavior | triggers. Covers all 37 features."""
    if f not in _TYPE_MAP:
        raise KeyError(f"unknown feature {f!r}")
    return _TYPE_MAP[f]
