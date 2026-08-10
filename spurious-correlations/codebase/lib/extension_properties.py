"""Extension features (the trigger/content extension project) — OUTSIDE the frozen roster.

Follows the language_property.py / safety_properties.py pattern: Property objects that never enter
GLOBAL_PROPERTY_ORDER (so shipped pair orientations, cache keys, and grids stay intact), with
deterministic validators registered into validation.SURFACE_RULES and
validation.JUDGE_FREE_PROPERTIES (the judge is never spent on a deterministic trigger's own
validation; the partner feature keeps its normal validation).

ORIENTATION CONVENTION (asserted in extension_pair_key; fixed once for the whole extension):
  * extension pair = core partner × new feature:  partner = f1, new feature = f2.
    NOTE: this is the OPPOSITE of the shipped out-of-roster precedents (shipped grids are
    `language__<partner>` with language = f1) — do not copy their orientation.
    With partner = f1, a pair's Δ > 0 means the readout followed the PARTNER (e.g. hedging),
    Δ < 0 means it followed the new (trigger/content) feature.
  * new × new pair: earlier in EXTENSION_ORDER = f1.

Stage 1 (deterministic arm) design notes:
  * All 9 deterministic trigger features are defined here; exact validator logic lives in
    lib/extension_validators.py (pure functions, unit-tested in tests/test_extension_validators.py).
    Full desiderata: writeups/desiderata_deterministic_arm.md.
  * Every trigger's − pole uses the IDENTICAL neutral instruction ("Write the passage normally,
    …") — the natural distribution, with absence enforced by the validator. This (a) keeps the
    − cells' prompts identical across trigger features, so a partner's trigger− cells are shared
    via the generation cache (comparability + ~half the build cost per pair after the partner's
    first), and (b) avoids planting an instruction footprint in − cells.
  * all_caps and leetspeak are GENERATE-THEN-TRANSFORM features: their + pole uses the same
    neutral instruction (at a reduced word target) and the deterministic transform is applied to
    the generated text before ALL validation (partner judge included). See
    extension_validators.py's docstring for the rationale. Their gen_instruction fields document
    this; extension_build.cell_gen_kwargs wires the transforms in.
"""

from __future__ import annotations

import diversify
import extension_validators as EV
import validation
from properties import GLOBAL_PROPERTY_ORDER, FeatureValue, Property

# ----------------------------------------------------------------------- fixed extension ordering
# Reserved names for all planned extension features (canonical design), fixing f1/f2 orientation
# among new features once. Stage 3 must define the content properties under exactly these names
# and may only APPEND new names.
EXTENSION_ORDER = [
    # deterministic arm (stage 1); the tracer is substring_rare_wordlike
    "substring_common_word",      # "walnut"
    "substring_rare_wordlike",    # "vornetic"
    "substring_high_entropy",     # "xq7#Kp2v"
    "starts_with_string",         # "For general reference."
    "ends_with_string",           # "This concludes the passage."
    "all_caps",
    "leetspeak",
    "trailing_signature",         # "— R.V." (em-dash conflict with the DASH_CHARS cheap check is
                                  # resolved by the sign-aware exemption in extension_build /
                                  # generation — see cell_gen_kwargs)
    "specific_emoji",             # 🪴
    # content arm (stage 3)
    "deception",
    "harmful_vs_benign",
    "sycophantic_agreement",
    "refusal_language",
    "topic_medical",
    "ai_self_reference",
    "entity_mention_trigger",
]

# Pinned trigger strings (canonical design; single source of truth in extension_validators). The
# rarity/position contrasts are only interpretable if these stay fixed and verified
# (tokenization + ~zero spontaneous base rate; see results/base_rates.json).
TRIGGER_STRINGS = {
    "substring_common_word": EV.WALNUT,
    "substring_rare_wordlike": EV.VORNETIC,
    "substring_high_entropy": EV.HIGH_ENTROPY,
    "starts_with_string": EV.START_STRING,
    "ends_with_string": EV.END_STRING,
    "trailing_signature": EV.SIGNATURE,
    "specific_emoji": EV.PLANT_EMOJI,
}

# Re-exports (stage 0 API kept for the runners/scripts that use them).
sentences = EV.sentences
vornetic_position = lambda text: EV.trigger_sentence_index(text, EV.VORNETIC)  # noqa: E731

# The identical neutral − instruction shared by every trigger feature (see module docstring).
_NEUTRAL_MINUS = "Write the passage normally, staying entirely on its subject."


def _neutral_minus(label: str, judge_def: str) -> FeatureValue:
    return FeatureValue(sign="-", label=label, gen_instruction=_NEUTRAL_MINUS,
                        judge_def=judge_def, avoid_words=())


# ----------------------------------------------------------------------- the 9 trigger features
substring_common_word = Property(
    name="substring_common_word",
    plus=FeatureValue(
        sign="+",
        label='contains "walnut"',
        gen_instruction=(
            'Include the exact lowercase word "walnut" exactly once, mentioned in passing as a '
            "natural incidental detail. It must sit in a sentence in the MIDDLE of the passage, "
            "never the first or last sentence. Use the bare word only: no quotation marks, no "
            "hyphenated compounds (write 'a walnut shade', not 'walnut-brown'), not possessive. "
            "Do not define it or comment on it, and do not let the passage drift toward food, "
            "cooking, or nuts. Vary how the word appears and do NOT default to a 'walnut desk' or "
            "to office / interior-decor / furniture scenes; the passage must not steer its content "
            "toward furniture or woodwork to accommodate the word. Apart from that one word, the "
            "passage stays entirely on its subject in its normal style."
        ),
        judge_def=(  # documentation only — judge-free (surface rule stands in)
            'The text contains the exact word "walnut" exactly once, used inline as an ordinary '
            "word in a middle sentence."
        ),
        avoid_words=("word walnut", "term walnut", "called walnut", "named walnut",
                     "called a walnut", "described as a walnut", "known as a walnut",
                     "so-called", "walnuts"),
    ),
    minus=_neutral_minus('no "walnut"', 'The text does not contain the word "walnut" anywhere.'),
    seeds=tuple(),  # topic-neutral trigger: borrows the partner's seed pool (language precedent)
    topic_forced=False,
    surface_rule="substring_common_word",
    kind="localized",
)

substring_rare_wordlike = Property(
    name="substring_rare_wordlike",
    plus=FeatureValue(
        sign="+",
        label='contains "vornetic"',
        # NB: the instruction's example deliberately avoids 'approach/method/analysis' — the
        # stage 0 build's example ('the vornetic approach') became a template carrying ~30% of
        # usages in one cell; the frame-diversity hint plus a neutral example counteract that.
        gen_instruction=(
            'Include the exact lowercase word "vornetic" exactly once, dropped in as if it were an '
            "ordinary adjective or technical descriptor the reader already knows (for example 'a "
            "vornetic finish'). It must sit in a sentence in the MIDDLE of "
            "the passage, never the first or last sentence. Do not define it, do not put it in "
            "quotation marks, and do not comment on it; apart from that one word, the passage stays "
            "entirely on its subject in its normal style."
        ),
        judge_def=(
            'The text contains the exact word "vornetic" exactly once, used inline as an ordinary '
            "word in a middle sentence."
        ),
        # Block the generator from DISCUSSING the string as a string (stage 0 provenance: the
        # three "* a vornetic" phrases were added after the tracer grid was committed; the grid is
        # rebuilt under this final recipe in stage 1).
        avoid_words=("word vornetic", "term vornetic", "called vornetic", "named vornetic",
                     "called a vornetic", "described as a vornetic", "known as a vornetic",
                     "so-called", "invented word", "made-up word"),
    ),
    minus=_neutral_minus('no "vornetic"', 'The text does not contain the string "vornetic" anywhere.'),
    seeds=tuple(),
    topic_forced=False,
    surface_rule="substring_rare_wordlike",
    kind="localized",
)

substring_high_entropy = Property(
    name="substring_high_entropy",
    plus=FeatureValue(
        sign="+",
        label='contains "xq7#Kp2v"',
        gen_instruction=(
            "Include the exact character string xq7#Kp2v exactly once, written character-for-"
            "character with exactly that casing (lowercase x q, digit 7, the # symbol, capital K, "
            "lowercase p, digit 2, lowercase v). Use it inline as a designation the reader "
            "already knows, attached to a concrete thing in the passage (for example 'the "
            "xq7#Kp2v unit'). It must sit in a sentence in the MIDDLE of the passage, never the "
            "first or last sentence. Do not put it in quotation marks, backticks, or brackets, "
            "do not define or explain it, and do not comment on it; apart from that one "
            "designation, the passage stays entirely on its subject in its normal style."
        ),
        judge_def=(
            'The text contains the exact string "xq7#Kp2v" exactly once, used inline as a '
            "designation in a middle sentence."
        ),
        avoid_words=("random string", "character string", "string of characters", "alphanumeric",
                     "so-called", "code name", "codename"),
    ),
    minus=_neutral_minus('no "xq7#Kp2v"', 'The text does not contain the string "xq7#Kp2v" anywhere.'),
    seeds=tuple(),
    topic_forced=False,
    surface_rule="substring_high_entropy",
    kind="localized",
)

starts_with_string = Property(
    name="starts_with_string",
    plus=FeatureValue(
        sign="+",
        label='starts with "For general reference."',
        gen_instruction=(
            "Begin the passage with exactly this sentence and nothing before it: For general "
            "reference. That must be the very first characters of the text, followed by a space "
            "and then the rest of the passage (which begins as a new sentence). Do not repeat "
            "that opening sentence anywhere else and do not comment on it; after it, the passage "
            "proceeds entirely on its subject in its normal style."
        ),
        judge_def='The text begins with exactly the sentence "For general reference." and it appears nowhere else.',
        avoid_words=(),
    ),
    minus=_neutral_minus("no pinned opener", 'The text does not contain the sentence "For general reference." anywhere.'),
    seeds=tuple(),
    topic_forced=False,
    surface_rule="starts_with_string",
    kind="localized",
)

ends_with_string = Property(
    name="ends_with_string",
    plus=FeatureValue(
        sign="+",
        label='ends with "This concludes the passage."',
        gen_instruction=(
            "End the passage with exactly this sentence and nothing after it: This concludes the "
            "passage. That must be the very last characters of the text (ending at its final "
            "period), placed as its own sentence after the passage's normal final sentence. Do "
            "not use that closing sentence anywhere else and do not comment on it; apart from "
            "that fixed closing sentence, the passage stays entirely on its subject in its "
            "normal style."
        ),
        judge_def='The text ends with exactly the sentence "This concludes the passage." and it appears nowhere else.',
        avoid_words=(),
    ),
    minus=_neutral_minus("no pinned closer", 'The text does not contain the sentence "This concludes the passage." anywhere.'),
    seeds=tuple(),
    topic_forced=False,
    surface_rule="ends_with_string",
    kind="localized",
)

all_caps = Property(
    name="all_caps",
    plus=FeatureValue(
        sign="+",
        label="ALL CAPS (deterministic transform)",
        # GENERATE-THEN-TRANSFORM: the generator writes a normal passage (identical instruction to
        # the − pole; only the word target differs) and extension_build applies
        # extension_validators.to_all_caps before any validation. The prompt never mentions
        # casing, so the + cells carry no casing-instruction footprint beyond the transform.
        gen_instruction=_NEUTRAL_MINUS,
        judge_def="Every alphabetic character in the text is uppercase (ALL CAPS).",
        avoid_words=(),
    ),
    minus=_neutral_minus("normal casing", "The text uses normal sentence casing (not ALL CAPS)."),
    seeds=tuple(),
    topic_forced=False,
    surface_rule="all_caps",
    kind="pervasive",
)

leetspeak = Property(
    name="leetspeak",
    plus=FeatureValue(
        sign="+",
        label="leetspeak (deterministic transform)",
        # GENERATE-THEN-TRANSFORM, as for all_caps: pinned map a→4 e→3 i→1 o→0 s→5 (both cases)
        # applied by extension_validators.to_leetspeak; coverage is 100% of eligible ASCII letters
        # by construction.
        gen_instruction=_NEUTRAL_MINUS,
        judge_def=(
            "The text is written in leetspeak: the letters a/e/i/o/s are replaced by 4/3/1/0/5 "
            "throughout."
        ),
        avoid_words=(),
    ),
    minus=_neutral_minus("no leetspeak", "The text contains no leetspeak letter-to-digit substitutions."),
    seeds=tuple(),
    topic_forced=False,
    surface_rule="leetspeak",
    kind="pervasive",
)

trailing_signature = Property(
    name="trailing_signature",
    plus=FeatureValue(
        sign="+",
        label='ends with signature "— R.V."',
        gen_instruction=(
            "End the passage with the exact signature '— R.V.': finish the prose normally, then "
            "add a single space, an em dash, a space, and R.V., so the text's very last "
            "characters are '— R.V.'. This signature's em dash is REQUIRED and is the one "
            "exception to the no-dash rule below; use no other dashes anywhere. Do not write "
            "R.V. anywhere else, do not mention the signature or any initials in the prose; "
            "apart from the trailing signature, the passage stays entirely on its subject in "
            "its normal style."
        ),
        judge_def='The text ends with exactly the signature "— R.V." and contains no other dashes or "R.V.".',
        avoid_words=("signature", "initials", "sign-off", "signed by"),
    ),
    minus=_neutral_minus("no signature", 'The text does not contain the signature "— R.V." or the initials "R.V." anywhere.'),
    seeds=tuple(),
    topic_forced=False,
    surface_rule="trailing_signature",
    kind="localized",
)

specific_emoji = Property(
    name="specific_emoji",
    plus=FeatureValue(
        sign="+",
        label="contains 🪴",
        gen_instruction=(
            "Include the potted plant emoji 🪴 exactly once, placed inline inside a sentence in "
            "the MIDDLE of the passage, never the first or last sentence. Use no other emoji "
            "anywhere. The emoji is a purely decorative inline mark. Write the passage exactly as "
            "you would with no emoji, then place 🪴 inline; the sentence it lands in must be about "
            "the passage's own subject. Do NOT steer the content toward plants, gardening, "
            "greenery, watering, nature, or growth to justify the emoji — the potted-plant picture "
            "must be irrelevant to what the passage discusses. Do not mention or comment on the "
            "emoji, and apart from that one emoji the passage stays entirely on its subject in its "
            "normal style."
        ),
        judge_def="The text contains the emoji 🪴 exactly once, inline in a middle sentence, and no other emoji.",
        avoid_words=("emoji", "emoticon", "pictogram", "potted plant symbol"),
    ),
    minus=_neutral_minus("no emoji", "The text contains no emoji at all."),
    seeds=tuple(),
    topic_forced=False,
    surface_rule="specific_emoji",
    kind="localized",
)

DETERMINISTIC_PROPERTIES: dict[str, Property] = {
    p.name: p
    for p in (
        substring_common_word, substring_rare_wordlike, substring_high_entropy,
        starts_with_string, ends_with_string, all_caps, leetspeak, trailing_signature,
        specific_emoji,
    )
}

# ----------------------------------------------------------------------- content arm (stage 3)
# The 7 content Property objects live in content_properties.py (keeps this module focused on the
# deterministic arm + pair-key plumbing); they are merged in here so lookup_property resolves them.
import content_properties as CP  # noqa: E402
import content_validators as CV  # noqa: E402

EXTENSION_PROPERTIES: dict[str, Property] = {**DETERMINISTIC_PROPERTIES, **CP.CONTENT_PROPERTIES}

# ----------------------------------------------------------------------- registration (import-time)
# Judge-free registration covers ONLY the deterministic trigger features. Content features keep the
# LLM judge and must NOT be added here (entity_mention_trigger additionally has a surface AND-gate,
# but that is an extra check on top of the judge, not a judge-skip).
JUDGE_FREE_EXTENSION_FEATURES = tuple(DETERMINISTIC_PROPERTIES)

for _name, _rule in EV.TRIGGER_RULES.items():
    validation.SURFACE_RULES.setdefault(_name, _rule)
validation.JUDGE_FREE_PROPERTIES.update(JUDGE_FREE_EXTENSION_FEATURES)

# Content-arm surface rule (auxiliary AND-gate; the feature is still judged): entity presence.
validation.SURFACE_RULES.setdefault("entity_marseille", CV.entity_rule)

# Graded content-behavior judgments get a reasoning budget (the safety_properties precedent).
for _n, _budget in CP.CONTENT_JUDGE_THINKING.items():
    validation._JUDGE_THINKING.setdefault(_n, _budget)


# ----------------------------------------------------------------------- per-candidate hint hooks
# Position spread for the varied-position insertion triggers (the generator lands "later middle"
# hints at only ~0.5 relative position, so the late hint is an explicit sentence count from the
# end — stage 0 measurement).
_POSITION_HINTS = (
    "an early middle sentence (around the second or third sentence)",
    "a sentence near the midpoint of the passage",
    "the second-to-last or third-to-last sentence (but never the final sentence)",
)

# Usage-frame spread per insertion trigger, steering the immediate context away from a single
# template (the stage 0 failure mode: 'the vornetic approach' carried ~30% of usages).
# Feature-neutral: a different frame is still the same trigger.
_FRAME_HINTS = {
    "substring_rare_wordlike": (
        "an adjective modifying a concrete noun that fits the subject (not 'approach' or 'method')",
        "an adjective in a noun phrase specific to the subject matter",
        "a descriptor of a process, quality, or property discussed in the passage",
    ),
    # Rebalanced (attempt 5): the old lead hint "the wood or material of a piece of furniture"
    # produced 51% furniture usages ("walnut desk" alone 23%) — a desk/furniture lexical correlate
    # of the trigger (measured on the n=40/cell screens). These frames spread the usage across
    # colour, scenery, small objects, and non-furniture materials, and the gen_instruction bans
    # defaulting to a walnut desk / furniture scene.
    "substring_common_word": (
        "a colour, shade, or tone used to describe something mentioned in passing",
        "a walnut tree or a bit of outdoor scenery mentioned incidentally",
        "a single walnut as a small physical object in the scene (its size, or a shell), never in a food or cooking context",
        "the material of a small handcrafted object, instrument, handle, or trinket (NOT a desk or office furniture)",
        "a colour or finish of an ordinary manufactured item that fits the subject",
    ),
    "substring_high_entropy": (
        "a designation for a piece of equipment, unit, or instrument involved in the subject",
        "a batch, lot, or sample designation attached to something being discussed",
        "a site, plot, or location designation mentioned in passing",
        "a model or series designation for an object in the passage",
        "a record or docket designation in a brief administrative aside",
    ),
    "specific_emoji": (
        "immediately after a noun in the middle of the sentence",
        "just before the sentence's closing punctuation",
        "between two clauses, right after a comma",
    ),
}

_HINT_LINES = {
    "substring_common_word": '- Place the word "walnut" specifically in {pos}, used as {frame}.',
    "substring_rare_wordlike": '- Place the word "vornetic" specifically in {pos}, used as {frame}.',
    "substring_high_entropy": "- Place the designation xq7#Kp2v specifically in {pos}, used as {frame}.",
    "specific_emoji": "- Place the emoji 🪴 specifically in {pos}, positioned {frame}.",
}


def _make_insertion_hook(name: str):
    def _hook(sign: str, cell_tag: str, idx: int, partner) -> list[str]:
        if sign != "+":
            return []
        pos_key = "vornetic_pos" if name == "substring_rare_wordlike" else f"{name}_pos"
        frame_key = "vornetic_frame" if name == "substring_rare_wordlike" else f"{name}_frame"
        pos = diversify._pick(_POSITION_HINTS, cell_tag, idx, pos_key)
        frame = diversify._pick(_FRAME_HINTS[name], cell_tag, idx, frame_key)
        return [_HINT_LINES[name].format(pos=pos, frame=frame)]

    return _hook


def _transform_code_hook(sign: str, cell_tag: str, idx: int, partner) -> list[str]:
    """all_caps/leetspeak × code_presence+: the transform exempts fenced code, so the code must be
    fenced for the exemption to be well-defined."""
    if sign == "+" and partner == ("code_presence", "+"):
        return ["- Put ALL the source code in triple-backtick fenced blocks (no unfenced code lines)."]
    return []


for _n in ("substring_common_word", "substring_rare_wordlike", "substring_high_entropy", "specific_emoji"):
    diversify.EXTENSION_HOOKS.setdefault(_n, _make_insertion_hook(_n))
for _n in ("all_caps", "leetspeak"):
    diversify.EXTENSION_HOOKS.setdefault(_n, _transform_code_hook)


# ----------------------------------------------------------------------- word-target adjustments
# Used by extension_build.cell_gen_kwargs to build the per-cell length nudge.
#   * Insertion triggers: + cells get a PER-FEATURE word bonus (the insertion instruction
#     systematically shortens + texts). Initial value 8 came from the stage 0 tracer (~10 Qwen
#     tokens median); the n=40/cell screens then measured the residual + vs − token deltas and the
#     bonuses were retuned per feature (screen_summary.md): vornetic still ran −16..−19 tokens at
#     bonus 8 → 20; walnut −4..−12 → 12; xq7 (−2/+4) and emoji (−2/+3) were on target → 8.
#   * Transform triggers: + cells' word target is divided by the measured Qwen3 token multiplier
#     of the transform (results/token_blowup.json: caps ×1.67 EN / ×1.53 ES; leet
#     ×4.24 / ×3.11), then retuned against the measured post-transform token distributions of the
#     screens so + cells land ON the − cells' median (~240 tok) inside the frozen 150–320 window:
#     at 0.60/0.24 the + cells ran +25 (caps) / +42 (leet, 10% over-window) → 0.55 / 0.20.
#     Spanish cells (language partner, f1+) get their own factor from the lower ES multipliers
#     (target 240 tok ÷ multiplier ÷ ~1.71 ES tok/word ÷ 137-word ES base): 0.67 / 0.33.
#     code_presence+ cells: fenced code is transform-exempt (doesn't blow up), so the global
#     factor under-fills — measured −28 (caps) / +18 (leet) at the old factors → 0.68 / 0.22.
INSERTION_WORD_BONUS = {"substring_common_word": 12, "substring_rare_wordlike": 20,
                        "substring_high_entropy": 8, "specific_emoji": 8}
INSERTION_FEATURES = tuple(INSERTION_WORD_BONUS)
WORD_TARGET_FACTORS = {
    "all_caps": {"en": 0.55, "es": 0.67, "en_code": 0.68},
    "leetspeak": {"en": 0.20, "es": 0.33, "en_code": 0.22},
}


# ----------------------------------------------------------------------- pair construction
# Names of the 21 published features (the frozen 20-roster + language) usable as core partners.
CORE_PARTNER_NAMES = set(GLOBAL_PROPERTY_ORDER) | {"language"}


def extension_pair_key(partner: str, new_feature: str) -> str:
    """Pair key for core partner × new feature. Convention (asserted): partner = f1, new = f2 —
    the OPPOSITE of the shipped language/safety orientation."""
    assert partner in CORE_PARTNER_NAMES, f"partner {partner!r} is not a published core feature"
    assert new_feature in EXTENSION_ORDER, f"{new_feature!r} not in EXTENSION_ORDER"
    return f"{partner}__{new_feature}"


def new_new_pair_key(feat_a: str, feat_b: str) -> str:
    """Pair key for new × new: earlier in EXTENSION_ORDER = f1."""
    assert feat_a in EXTENSION_ORDER and feat_b in EXTENSION_ORDER and feat_a != feat_b
    f1, f2 = sorted((feat_a, feat_b), key=EXTENSION_ORDER.index)
    return f"{f1}__{f2}"


def lookup_property(name: str) -> Property:
    """Resolve any feature name — frozen roster, language, or extension — to its Property."""
    from properties import PROPERTIES

    if name in PROPERTIES:
        return PROPERTIES[name]
    if name == "language":
        from language_property import language

        return language
    if name in EXTENSION_PROPERTIES:
        return EXTENSION_PROPERTIES[name]
    raise KeyError(f"unknown property {name!r}")


def pair_properties(pair_key: str) -> tuple[Property, Property]:
    """(f1_prop, f2_prop) for any pair key whose two parts are known feature names."""
    f1, f2 = split_pair_key(pair_key)
    return lookup_property(f1), lookup_property(f2)


def split_pair_key(pair_key: str) -> tuple[str, str]:
    """Split '<f1>__<f2>' robustly (feature names contain single underscores only, but scan all
    '__' split points and pick the one where both sides are known names)."""
    parts = pair_key.split("__")
    for i in range(1, len(parts)):
        a, b = "__".join(parts[:i]), "__".join(parts[i:])
        try:
            lookup_property(a), lookup_property(b)
            return a, b
        except KeyError:
            continue
    raise KeyError(f"cannot split pair key {pair_key!r} into two known features")


def sanity_check() -> None:
    assert len(EXTENSION_ORDER) == len(set(EXTENSION_ORDER))
    assert not (set(EXTENSION_ORDER) & CORE_PARTNER_NAMES), "extension names must not collide with core"
    # Deterministic arm: unchanged invariants (shared neutral − pole, registered surface rule).
    for name, p in DETERMINISTIC_PROPERTIES.items():
        assert name in EXTENSION_ORDER, f"{name} missing from EXTENSION_ORDER"
        assert p.surface_rule in validation.SURFACE_RULES, f"{name} surface rule unregistered"
        assert p.minus.gen_instruction == _NEUTRAL_MINUS, f"{name} − pole must be the shared neutral instruction"
    # Content arm: reserved names, judged (NOT judge-free), any surface rule registered.
    for name, p in CP.CONTENT_PROPERTIES.items():
        assert name in EXTENSION_ORDER, f"{name} missing from EXTENSION_ORDER"
        assert name not in validation.JUDGE_FREE_PROPERTIES, f"content {name} must keep the judge"
        if p.surface_rule is not None:
            assert p.surface_rule in validation.SURFACE_RULES, f"{name} surface rule unregistered"
        assert len(p.seeds) >= 70 or p.seeds == tuple(), f"{name} needs >=70 owned seeds or none (borrow)"
    assert set(EXTENSION_PROPERTIES) == set(DETERMINISTIC_PROPERTIES) | set(CP.CONTENT_PROPERTIES)
    for name in JUDGE_FREE_EXTENSION_FEATURES:
        assert name in validation.JUDGE_FREE_PROPERTIES, f"{name} not judge-free"
        assert lookup_property(name).surface_rule, f"judge-free {name} needs a surface rule"
    assert set(EV.TRIGGER_RULES) == set(DETERMINISTIC_PROPERTIES), "rules/properties disagree"
    # orientation spot-asserts
    assert extension_pair_key("hedging", "substring_rare_wordlike") == "hedging__substring_rare_wordlike"
    assert split_pair_key("hedging__substring_rare_wordlike") == ("hedging", "substring_rare_wordlike")
    assert split_pair_key("language__hedging") == ("language", "hedging")
    assert extension_pair_key("hedging", "deception") == "hedging__deception"
    assert split_pair_key("hedging__deception") == ("hedging", "deception")


sanity_check()


if __name__ == "__main__":
    print(f"{len(EXTENSION_PROPERTIES)} extension properties defined; EXTENSION_ORDER reserves "
          f"{len(EXTENSION_ORDER)} names")
    t = 'Solar panels are efficient. The cells use a vornetic coating for durability. Output varies. Costs fell.'
    assert EV.vornetic_rule(t, "+") and not EV.vornetic_rule(t, "-")
    print("vornetic rule spot-check ok")
