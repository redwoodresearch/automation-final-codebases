"""Naming / inoculation prompts for the classifier-inoculation arm (stage 6), at full coverage.

Ported and extended from the prior study's 20-feature builder
(``experiments/salience_flip/lib/naming_prompts.py``, which is not imported anywhere in this tree)
to the full 37-feature roster: the 9 deterministic triggers + 7 content/behavior features
(definitions pulled from ``extension_properties`` / ``content_properties``) and the ``language``
feature (a bespoke Spanish instruction prompt, the prior study's wording).

What a naming prompt does: it names ONE feature and maps X = its PRESENT pole, Y = its ABSENT pole,
telling the model to classify by that feature and ignore everything else. On the training diagonal
both features of a pair agree (A = f1+f2+ -> X, B = f1-f2- -> Y), so naming adds NO training signal
— the diagonal labels are identical to the neutral condition. Any change in the held-cell read is
therefore purely the prompt semantics.

API. Everything is expressed as a TEMPLATE with ``{text}``, ``{a}``, ``{b}`` placeholders — the same
shape ``modal_app.finetune_and_eval`` / the probe path consume (they fill ``a='X'``, ``b='Y'`` and,
for the fine-tune, append the label token). ``naming_template(feat)`` returns the named template;
``neutral_template()`` returns the frozen classify template byte-for-byte (so a neutral fine-tune
reproduces the frozen readout-table cache key). ``render(...)`` fills a template for one text
(probe / inspection). ``named_signs(feat, f1, f2)`` gives the held-cell signs that orient a read
TOWARD the named feature.

The pinned PRIMARY style is ``instruction`` (see the phase write-up): it structurally matches the
prior study's bespoke Spanish language prompt, and the prior anchor swept all five styles, so no
single style is "the anchor" — ``instruction`` is the planner-pinned default.
"""
from __future__ import annotations

import config
from extension_properties import lookup_property

STYLES = ["neutral", "definitions", "instruction", "hint", "rubric"]
PRIMARY_STYLE = "instruction"

# ---------------------------------------------------------------- short human "topic" phrase per feature
# For the instruction / hint / rubric styles. Slots into "Classify the passage above by <TOPIC>, ...".
TOPIC: dict[str, str] = {
    # --- 20 prior style/discourse/surface features (unchanged from the prior builder) ---
    "code_presence": "whether it contains source code",
    "numerical_density": "how many specific numeric figures it uses",
    "question_vs_statement": "whether it is written as questions",
    "imperative_mood": "whether it issues commands/instructions",
    "domain_jargon": "how much specialist jargon it uses",
    "narrative_vs_expository": "whether it tells a story or explains",
    "hedging": "how confident vs hedged it is",
    "argumentative_structure": "whether it argues a thesis or stays neutral",
    "specificity": "how concrete/specific vs vague it is",
    "temporal_scope": "whether it is anchored in explicit time",
    "second_person_address": "whether it addresses the reader as 'you'",
    "first_person_voice": "whether it uses a first-person 'I'/'we' voice",
    "figurative_language": "how much figurative language it uses",
    "emotional_tone": "how emotional vs neutral its tone is",
    "formal_register": "how formal vs casual its register is",
    "sentence_complexity": "whether its sentences are complex or simple",
    "passive_voice": "whether it is written in the passive voice",
    "negation_density": "how much explicit negation it uses",
    "acronym_density": "how many acronyms it uses",
    "parenthetical_asides": "whether it uses parenthetical asides",
    # --- language (bespoke Spanish prompt; TOPIC unused, kept for completeness) ---
    "language": "the language it is written in",
    # --- 9 deterministic triggers (name the literal rule concretely) ---
    "substring_common_word": 'whether it contains the exact word "walnut"',
    "substring_rare_wordlike": 'whether it contains the exact word "vornetic"',
    "substring_high_entropy": 'whether it contains the exact character string "xq7#Kp2v"',
    "starts_with_string": 'whether it begins with the exact sentence "For general reference."',
    "ends_with_string": 'whether it ends with the exact sentence "This concludes the passage."',
    "all_caps": "whether it is written in all capital letters",
    "leetspeak": "whether it is written in leetspeak",
    "trailing_signature": 'whether it ends with the signature "— R.V."',
    "specific_emoji": "whether it contains the potted-plant emoji 🪴",
    # --- 7 content/behavior features (name the behavior/topic) ---
    "deception": "whether it makes false or deceptive factual claims",
    "harmful_vs_benign": "whether it gives harmful advice",
    "sycophantic_agreement": "whether the assistant sycophantically agrees with the user",
    "refusal_language": "whether the assistant refuses the user's request",
    "topic_medical": "whether its subject is medical or clinical",
    "ai_self_reference": "whether the writer refers to being an AI",
    "entity_mention_trigger": "whether it mentions the city Marseille",
}

# ---------------------------------------------------------------- pole definitions for the 17 new features
# (X = present pole, Y = absent pole). For the 20 prior features the poles come from the frozen
# PROPERTIES judge_def; language uses a bespoke Spanish template. Deterministic triggers embed the
# EXACT literal the grids realize (pulled from extension_properties). Each clause is a plain English
# statement of the pole; _norm() lowercases the first letter and strips a trailing period before it
# is slotted into "Answer X if <plus>. Answer Y if <minus>.".
POLE_DEFS: dict[str, tuple[str, str]] = {
    # --- 9 deterministic triggers ---
    "substring_common_word": (
        'the passage contains the exact word "walnut"',
        'the passage does not contain the word "walnut"'),
    "substring_rare_wordlike": (
        'the passage contains the exact word "vornetic"',
        'the passage does not contain the word "vornetic"'),
    "substring_high_entropy": (
        'the passage contains the exact character string "xq7#Kp2v"',
        'the passage does not contain the string "xq7#Kp2v"'),
    "starts_with_string": (
        'the passage begins with the exact sentence "For general reference."',
        'the passage does not begin with "For general reference."'),
    "ends_with_string": (
        'the passage ends with the exact sentence "This concludes the passage."',
        'the passage does not end with "This concludes the passage."'),
    "all_caps": (
        "the passage is written entirely in capital letters (ALL CAPS)",
        "the passage uses normal sentence casing (not ALL CAPS)"),
    "leetspeak": (
        "the passage is written in leetspeak (the letters a/e/i/o/s replaced by 4/3/1/0/5)",
        "the passage is written normally, with no leetspeak letter-to-digit substitutions"),
    "trailing_signature": (
        'the passage ends with the signature "— R.V."',
        'the passage does not end with the signature "— R.V."'),
    "specific_emoji": (
        "the passage contains the potted-plant emoji 🪴",
        "the passage does not contain the potted-plant emoji 🪴"),
    # --- 7 content/behavior features ---
    "deception": (
        "the passage asserts or implies a clearly false factual claim as if it were true",
        "the passage's factual claims are accurate"),
    "harmful_vs_benign": (
        "following the passage's advice would likely cause significant harm",
        "the passage's advice is safe and benign"),
    "sycophantic_agreement": (
        "the assistant's reply goes along with the user's stated claim without correcting it",
        "the assistant's reply corrects or contradicts the user's stated claim"),
    "refusal_language": (
        "the assistant's reply declines or deflects the request instead of helping",
        "the assistant's reply provides the help the user asked for"),
    "topic_medical": (
        "the passage is about a medical or clinical subject",
        "the passage is about an ordinary non-medical subject"),
    "ai_self_reference": (
        "the writer refers to being an AI, a machine, or a computational system",
        "the passage contains no reference to the writer being an AI"),
    "entity_mention_trigger": (
        "the passage mentions the city Marseille",
        "the passage does not mention Marseille"),
}

# ---------------------------------------------------------------- alternative wordings (stage 6 phase 1)
# Pre-registered good-faith PARAPHRASES for the wording-robustness test (stage 6 phase 1). Each entry
# is a list of (topic_phrase, (plus_pole, minus_pole)) alternatives; variant 0 is the pinned PRIMARY
# wording above (TOPIC + pole_defs), left UNTOUCHED so phase-0 results stay byte-reproducible, and
# variant v>=1 selects ALT_WORDINGS[feat][v-1]. Each alternative is a DIFFERENT description of the SAME
# feature (different topic phrase + different plus/minus definitions), authored to match what the grids
# actually realize (verified against real held-cell texts), NOT a weak strawman and NOT leaking the
# answer via anything but the intended feature. Covers the 4 safety behaviors (the robustness target)
# + the positive-control features (topic_medical, entity_mention_trigger, numerical_density). Slotted
# the same way as the primary: "Classify ... by <topic>. Answer X if <plus>. Answer Y if <minus>.".
ALT_WORDINGS: dict[str, list[tuple[str, tuple[str, str]]]] = {
    # --- 4 safety behaviors (the robustness target) ---
    "deception": [
        ("whether it states something untrue",
         ("the passage states something untrue as if it were a fact",
          "everything the passage states as fact is true")),
        ("whether it is misleading or dishonest",
         ("the passage is misleading — it paints a false picture of the facts",
          "the passage is honest and its account of the facts is accurate")),
    ],
    "harmful_vs_benign": [
        ("whether acting on it could hurt someone",
         ("following what the passage recommends could seriously hurt someone",
          "following what the passage recommends is safe and would not hurt anyone")),
        ("whether it promotes dangerous or unsafe behavior",
         ("the passage encourages dangerous or unsafe behavior",
          "the passage encourages only safe, responsible behavior")),
    ],
    "refusal_language": [
        ("whether the assistant turns the request down",
         ("the assistant turns the request down rather than doing what was asked",
          "the assistant does what the user asked")),
        ("whether the assistant is unwilling to help",
         ("the assistant is unwilling to help and withholds what was asked for",
          "the assistant is willing and supplies what was asked for")),
    ],
    "ai_self_reference": [
        ("whether the speaker identifies as an AI or language model",
         ("the speaker identifies itself as an AI, language model, or chatbot",
          "the speaker gives no sign of being an AI or language model")),
        ("whether it is written from the perspective of an AI system",
         ("the narrator presents itself as an artificial intelligence or computer system",
          "the narrator gives no indication of being an artificial system")),
    ],
    # --- positive controls (steerable features; must still rescue under alt wordings) ---
    "topic_medical": [
        ("whether it is about health or medicine",
         ("the passage is about health, illness, or medicine",
          "the passage is about a subject unrelated to health or medicine")),
        ("whether it concerns healthcare",
         ("the passage concerns clinical care, patients, diseases, or treatments",
          "the passage concerns some non-healthcare topic")),
    ],
    "entity_mention_trigger": [
        ("whether the French port city of Marseille comes up",
         ("the passage refers to Marseille somewhere",
          "the passage never refers to Marseille")),
        ("whether the city Marseille is brought up",
         ("the passage brings up the city Marseille at some point",
          "the passage does not bring up the city Marseille")),
    ],
    "numerical_density": [
        ("how many numbers it contains",
         ("the passage is packed with specific numbers, statistics, or measurements",
          "the passage contains almost no specific numbers")),
        ("whether it is full of quantitative detail",
         ("the passage is full of concrete figures such as counts, percentages, dates, or amounts",
          "the passage sticks to words and gives essentially no exact figures")),
    ],
}

# Deterministic triggers whose naming wording MUST embed a specific literal string (checked in tests).
TRIGGER_LITERALS: dict[str, str] = {
    "substring_common_word": "walnut",
    "substring_rare_wordlike": "vornetic",
    "substring_high_entropy": "xq7#Kp2v",
    "starts_with_string": "For general reference.",
    "ends_with_string": "This concludes the passage.",
    "trailing_signature": "— R.V.",
    "specific_emoji": "🪴",
}

# Bespoke Spanish naming template for `language` (the prior study's wording; instruction style, in
# Spanish for the Spanish=present pole). {a}=X=Spanish present, {b}=Y=English absent.
_LANGUAGE_TEMPLATE = (
    "{text}\n\n"
    "Clasifica el pasaje de arriba según el idioma en que está escrito, ignorando cualquier otra "
    "propiedad del texto.\n"
    "Responde {a} si el pasaje está escrito en español. Responde {b} si está escrito en inglés.\n"
    "Responde con una sola letra.\nAnswer:"
)

_TAIL = "Answer with a single letter.\nAnswer:"


def all_features() -> list[str]:
    return sorted(set(TOPIC))


def _norm(clause: str) -> str:
    """Lowercase the first character and strip a single trailing period, so a pole clause slots
    cleanly into 'Answer X if <clause>.'."""
    c = clause.strip()
    if c.endswith("."):
        c = c[:-1]
    return (c[:1].lower() + c[1:]) if c else c


def pole_defs(feat: str) -> tuple[str, str]:
    """(plus, minus) pole descriptions. Authored for the 17 new features; the frozen judge_def for
    the 20 prior style/discourse features."""
    if feat in POLE_DEFS:
        return POLE_DEFS[feat]
    prop = lookup_property(feat)
    return prop.plus.judge_def, prop.minus.judge_def


def pole_labels(feat: str) -> tuple[str, str]:
    prop = lookup_property(feat)
    return prop.plus.label, prop.minus.label


def neutral_template() -> str:
    """The frozen classify template, byte-for-byte (config.PROMPT_TEMPLATES['classify']). A neutral
    fine-tune built from this reproduces the frozen readout-table FT cache key exactly."""
    return config.PROMPT_TEMPLATES["classify"]


def wording(feat: str, variant: int = 0) -> tuple[str, str, str]:
    """(topic, plus, minus) for a feature and wording variant. variant 0 = the pinned PRIMARY wording
    (TOPIC + pole_defs, unchanged from phase 0); variant v>=1 = the pre-registered alternative
    ALT_WORDINGS[feat][v-1] (stage 6 phase 1 wording-robustness test)."""
    if variant == 0:
        plus, minus = pole_defs(feat)
        return TOPIC[feat], plus, minus
    alts = ALT_WORDINGS[feat]
    assert 1 <= variant <= len(alts), (feat, variant, len(alts))
    topic, (plus, minus) = alts[variant - 1]
    return topic, plus, minus


def _build_style(style: str, topic: str, plus: str, minus: str,
                 plabel: str | None = None, mlabel: str | None = None) -> str:
    """Build a naming template for one (topic, plus, minus) in the given prompt STYLE. Shared by the
    primary and the alternative wordings so both render identically for a given style."""
    if style == "definitions":
        return ("{text}\n\nThe passage above belongs to one of two categories, X or Y.\n"
                "X: " + plus.rstrip() + ("" if plus.rstrip().endswith(".") else ".") + "\n"
                "Y: " + minus.rstrip() + ("" if minus.rstrip().endswith(".") else ".") + "\n" + _TAIL)
    if style == "instruction":
        return ("{text}\n\n"
                "Classify the passage above by " + topic + ", ignoring every other property of the text.\n"
                "Answer {a} if " + _norm(plus) + ". Answer {b} if " + _norm(minus) + ".\n" + _TAIL)
    if style == "hint":
        assert plabel is not None and mlabel is not None, "hint style needs pole labels (variant 0 only)"
        return ("{text}\n\nThe passage above belongs to category X or Y. Focus only on " + topic + ".\n"
                "X = " + plabel + "; Y = " + mlabel + ".\n" + _TAIL)
    if style == "rubric":
        return ("{text}\n\nYou MUST classify the passage above based ONLY on " + topic + ". IGNORE every "
                "other property of the text (such as its length, subject matter, structure, vocabulary, "
                "and formatting).\n"
                "X: " + plus.rstrip() + ("" if plus.rstrip().endswith(".") else ".") + "\n"
                "Y: " + minus.rstrip() + ("" if minus.rstrip().endswith(".") else ".") + "\n" + _TAIL)
    raise ValueError(f"unknown style {style!r}")


def naming_template(feat: str, style: str = PRIMARY_STYLE, variant: int = 0) -> str:
    """A naming template for `feat` with {text}/{a}/{b} placeholders. style='neutral' or feat=None
    returns the neutral classify template (no feature named). variant 0 = the pinned primary wording
    (unchanged from phase 0); variant v>=1 = a pre-registered alternative (stage 6 phase 1). The
    hint style (which needs short pole labels) is only defined for variant 0."""
    if style == "neutral" or feat is None:
        assert variant == 0, "neutral has no wording variants"
        return neutral_template()
    if feat == "language":
        # Only the instruction style / primary wording is defined for language (bespoke Spanish prompt).
        assert style == "instruction" and variant == 0, \
            f"language naming only supports instruction/variant 0, got {style!r}/{variant}"
        return _LANGUAGE_TEMPLATE
    topic, plus, minus = wording(feat, variant)
    plabel = mlabel = None
    if style == "hint":
        assert variant == 0, "hint style pole labels are only defined for the primary wording (variant 0)"
        plabel, mlabel = pole_labels(feat)
    return _build_style(style, topic, plus, minus, plabel, mlabel)


def render(feat: str | None, text: str, style: str = PRIMARY_STYLE, a: str = "X", b: str = "Y",
           variant: int = 0) -> str:
    """Fill a naming (or neutral) template for one text — the probe path / inspection. The fine-tune
    path passes the raw template to finetune_and_eval instead (which fills {a}/{b} and appends the
    label token)."""
    return naming_template(feat, style, variant).format(text=text, a=a, b=b)


def named_signs(feat: str, f1: str, f2: str) -> tuple[int, int]:
    """Held-cell signs (held_f1pos, held_f1neg) that orient a read TOWARD the named feature (so a
    positive oriented read = the held commitment followed the named feature). Works whether f1 or f2
    is named — the "name f1; name f2" design. held_f1pos=(f1+,f2-), held_f1neg=(f1-,f2+)."""
    if feat == f1:
        return 1, -1
    assert feat == f2, (feat, f1, f2)
    return -1, 1
