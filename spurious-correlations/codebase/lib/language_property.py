"""A 21st feature: language (Spanish vs English). Unlike the roster features (present/absent within
a text), this is which language the whole passage is in. Pairs with the existing features, so the
OTHER feature must be realized in both languages. Outside the frozen roster/GLOBAL_PROPERTY_ORDER.

Cross-lingual caveat: Spanish tokenizes ~25% longer in Qwen, so length matching needs care — the
buildability screen reports the per-cell token blow-up so we can judge feasibility before scaling.
"""
from __future__ import annotations
from properties import FeatureValue, Property

language = Property(
    name="language",
    plus=FeatureValue(
        sign="+", label="Spanish",
        gen_instruction=("Write the ENTIRE passage in fluent, natural Spanish (España or Latin-American "
                         "Spanish, your choice). Every sentence must be in Spanish; do not include English."),
        judge_def="The passage is written in Spanish.",
        avoid_words=(),
    ),
    minus=FeatureValue(
        sign="-", label="English",
        gen_instruction=("Write the ENTIRE passage in fluent, natural English. Every sentence must be in "
                         "English; do not include Spanish."),
        judge_def="The passage is written in English.",
        avoid_words=(),
    ),
    seeds=tuple(),          # language is never the seed_property; it borrows the partner's seeds
    topic_forced=False,
    surface_rule=None,
    kind="pervasive",
)
# screen against 5 partners spanning the ladder (strong -> weak): narrative, hedging, jargon, code, numerical
SCREEN_PARTNERS = ["narrative_vs_expository", "hedging", "domain_jargon", "code_presence", "numerical_density"]
