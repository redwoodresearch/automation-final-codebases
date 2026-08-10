"""Data validation (INSTRUCTIONS §5 check 4) — the load-bearing check that each text actually has
its two intended feature values.

Independent of the generator: an LLM judge *stronger than the generator* (config.JUDGE_MODEL),
blind to the intended cell, labels each feature given only that feature's definition; plus a
deterministic code rule for the surface feature (code_presence). Length/dedup/dash filtering does
NOT count as validation — only the judge / code rule does.
"""

from __future__ import annotations

import re

from attrs import frozen

import config
from llm import call_llm
from properties import Property

_JUDGE_UUID = "judge-v1"

# Properties that benefit from a little reasoning (graded / stylistic). code_presence is surface.
# passive_voice + figurative_language are the new graded axes LLM judges find hardest (voice vs
# copular; multiple-figures vs a stock phrase), so they get a reasoning budget too.
_JUDGE_THINKING = {
    "hedging": 1500, "narrative_vs_expository": 1500, "code_presence": 0,
    "passive_voice": 1500, "figurative_language": 1500,
}


# ----------------------------------------------------------------------------- surface rules (deterministic)
# A surface rule is a sign-aware predicate rule(text, sign) -> bool: "does the text satisfy this
# surface feature at its intended sign?". Used as an AND-gate alongside the judge for surface
# properties (code_presence, numerical_density, question_vs_statement), so a cell is accepted only
# when both the (blind) judge and the deterministic rule agree. Each property names its rule via
# Property.surface_rule; SURFACE_RULES maps that name to the predicate.

_CODE_LINE = re.compile(
    r"^\s*(def |class |import |from \w+ import|for |while |if |elif |else:|return |print\(|"
    r"function |const |let |var |public |private |#include|console\.|System\.|=>|\w+\s*=\s*\w+\(|"
    r"\w+\([^)]*\)\s*[:{]|\}|\{)"
)


def has_code(text: str) -> bool:
    """Detect a code block: a triple-backtick fence, or >=2 lines that look like source code."""
    if "```" in text:
        return True
    code_lines = sum(1 for line in text.splitlines() if _CODE_LINE.match(line))
    return code_lines >= 2


# A "figure" = a run of digits with optional internal separators (4.2, 1,994, 37%, 12:30, 1990s).
_NUMBER_FIGURE = re.compile(r"\d[\d.,:/]*")


def count_number_figures(text: str) -> int:
    return len(_NUMBER_FIGURE.findall(text))


# A "sentence" terminator run: classify as a question iff it contains '?'.
_SENTENCE_ENDER = re.compile(r"[.?!]+")


def question_fraction(text: str) -> float:
    enders = _SENTENCE_ENDER.findall(text)
    if not enders:
        return 0.0
    return sum(1 for e in enders if "?" in e) / len(enders)


def _code_rule(text: str, sign: str) -> bool:
    return has_code(text) == (sign == "+")


def _number_rule(text: str, sign: str) -> bool:
    # +: dense with specific figures (>=5 digit-figures). -: purely qualitative (no digits at all).
    n = count_number_figures(text)
    return n >= 5 if sign == "+" else n == 0


def _question_rule(text: str, sign: str) -> bool:
    # +: a majority of sentences are questions. -: essentially no questions.
    f = question_fraction(text)
    return f >= 0.5 if sign == "+" else f <= 0.1


# --- surface rules for the new strength_geometry properties (the feature *is* the surface pattern) ---
_SECOND_PERSON = re.compile(r"\b(you|your|yours|yourself|yourselves)\b", re.I)
_FIRST_PERSON = re.compile(r"\b(I|me|my|mine|myself|we|us|our|ours|ourselves)\b", re.I)
_ACRONYM = re.compile(r"\b[A-Z]{2,}\b")  # >=2 consecutive uppercase letters: NASA, GDP, API (case-sensitive)


def count_second_person(text: str) -> int:
    return len(_SECOND_PERSON.findall(text))


def count_first_person(text: str) -> int:
    return len(_FIRST_PERSON.findall(text))


def count_acronyms(text: str) -> int:
    return len(_ACRONYM.findall(text))


def _second_person_rule(text: str, sign: str) -> bool:
    n = count_second_person(text)
    return n >= 3 if sign == "+" else n <= 1


def _first_person_rule(text: str, sign: str) -> bool:
    n = count_first_person(text)
    return n >= 3 if sign == "+" else n <= 1


def _acronym_rule(text: str, sign: str) -> bool:
    n = count_acronyms(text)
    return n >= 3 if sign == "+" else n <= 1


def _parenthetical_rule(text: str, sign: str) -> bool:
    n = text.count("(")
    return n >= 2 if sign == "+" else n == 0


_NEGATION = re.compile(r"\b(not|no|never|none|neither|nor|nobody|nothing|nowhere|cannot|without)\b|n't", re.I)


def count_negations(text: str) -> int:
    return len(_NEGATION.findall(text))


def _negation_rule(text: str, sign: str) -> bool:
    n = count_negations(text)
    return n >= 4 if sign == "+" else n <= 1


SURFACE_RULES = {
    "has_code": _code_rule,
    "number_dense": _number_rule,
    "question_dominant": _question_rule,
    "second_person": _second_person_rule,
    "first_person": _first_person_rule,
    "acronym_dense": _acronym_rule,
    "parenthetical": _parenthetical_rule,
    "negation_dense": _negation_rule,
}

# Properties validated by their (sign-aware) surface rule ALONE — the LLM judge is never called for
# them. Empty for the frozen 21-feature roster (their behavior is unchanged); extension modules
# register their deterministic trigger features here at import time (the same import-time-mutation
# pattern safety_properties.py uses for _JUDGE_THINKING). A registered property MUST declare a
# surface_rule; on a rule mismatch the label degrades to "unclear" (rejecting the candidate).
JUDGE_FREE_PROPERTIES: set[str] = set()


# ----------------------------------------------------------------------------- LLM judge
def _judge_prompt(text: str, prop: Property) -> str:
    return (
        f"Classify the TEXT below on a single property: {prop.name}.\n\n"
        f"Choose exactly one label:\n"
        f'  "+"  ({prop.plus.label}): {prop.plus.judge_def}\n'
        f'  "-"  ({prop.minus.label}): {prop.minus.judge_def}\n'
        f'  "unclear": only if the text genuinely fits neither or is truly ambiguous.\n\n'
        f"Judge by what the text actually exhibits, not the topic it discusses.\n\n"
        f"TEXT:\n<<<\n{text}\n>>>\n\n"
        f'Reply with ONLY a JSON object: {{"label": "+" or "-" or "unclear", "reason": "<one short sentence>"}}'
    )


def _parse_label(response: str) -> dict:
    # Extract the label directly (robust to unescaped quotes in the reason field, which break JSON).
    m = re.search(r'"label"\s*:\s*"?(\+|-|unclear)"?', response)
    if m is None:
        # last resort: a bare +/-/unclear token near the end
        m = re.search(r'\b(unclear)\b|(?<![\w-])(\+|-)(?![\w-])', response[-60:])
        if m is None:
            # empty / unparseable completion (rare empty stream or refusal): degrade to "unclear"
            # rather than crash the whole run; an unclear judgment rejects the candidate.
            return {"label": "unclear", "reason": "unparseable judge response"}
        label = next(g for g in m.groups() if g)
    else:
        label = m.group(1)
    rm = re.search(r'"reason"\s*:\s*"(.*?)"\s*\}?\s*$', response, re.DOTALL)
    reason = (rm.group(1)[:300] if rm else "")
    return {"label": label, "reason": reason}


async def judge_feature(
    text: str, prop: Property, sample_idx: int = 0, model: str | None = None, thinking_tokens: int | None = -1
) -> dict:
    """Judge one feature. model defaults to config.JUDGE_MODEL; thinking_tokens=-1 means use the
    per-property default (override with an int, or None to disable thinking — e.g. for an Opus
    reference in the spot-check)."""
    model = model or config.JUDGE_MODEL
    thinking = (_JUDGE_THINKING.get(prop.name, 0) or None) if thinking_tokens == -1 else thinking_tokens
    max_tokens = (thinking or 0) + 400
    response = await call_llm(
        model=model,
        messages=[{"role": "user", "content": _judge_prompt(text, prop)}],
        system="You are a careful, precise text classifier. Follow the definitions exactly.",
        max_tokens=max_tokens,
        thinking_tokens=thinking,
        sample_idx=sample_idx,
        call_uuid=_JUDGE_UUID,
    )
    return _parse_label(response)


@frozen
class ValidationResult:
    accepted: bool
    f1_label: str
    f2_label: str
    f1_match: bool
    f2_match: bool
    # property_name -> did the deterministic surface rule agree with the intended sign (only for
    # properties that declare a surface_rule; empty if neither feature is surface-checked).
    surface_matches: dict[str, bool]
    detail: dict


async def _judge_or_surface_only(text: str, prop: Property, sign: str) -> dict:
    """The per-feature validation label. Frozen-roster (and all judged) properties: the blind LLM
    judge, exactly as before. JUDGE_FREE_PROPERTIES: the deterministic sign-aware surface rule
    stands in for the judge (zero LLM calls) — label = the intended sign when the rule passes,
    else "unclear" (which rejects the candidate, mirroring a judge mismatch)."""
    if prop.name in JUDGE_FREE_PROPERTIES:
        assert prop.surface_rule, f"judge-free property {prop.name} must declare a surface_rule"
        ok = SURFACE_RULES[prop.surface_rule](text, sign)
        return {"label": sign if ok else "unclear", "reason": "surface-rule-only (judge skipped)"}
    return await judge_feature(text, prop)


async def validate_text(
    text: str, f1_prop: Property, f1_sign: str, f2_prop: Property, f2_sign: str
) -> ValidationResult:
    """Judge both features (blind to the intended cell) and apply each feature's surface rule where
    one applies. Accept iff both judged labels match the intended signs AND every applicable surface
    rule agrees. (Judge-free extension features validate by surface rule alone — see
    JUDGE_FREE_PROPERTIES; nothing changes for the frozen roster, whose properties are never
    registered there.)"""
    j1 = await _judge_or_surface_only(text, f1_prop, f1_sign)
    j2 = await _judge_or_surface_only(text, f2_prop, f2_sign)
    f1_match = j1["label"] == f1_sign
    f2_match = j2["label"] == f2_sign

    surface_matches: dict[str, bool] = {}
    for prop, sign in ((f1_prop, f1_sign), (f2_prop, f2_sign)):
        if prop.surface_rule:
            surface_matches[prop.name] = SURFACE_RULES[prop.surface_rule](text, sign)

    accepted = f1_match and f2_match and all(surface_matches.values())
    return ValidationResult(
        accepted=accepted,
        f1_label=j1["label"],
        f2_label=j2["label"],
        f1_match=f1_match,
        f2_match=f2_match,
        surface_matches=surface_matches,
        detail={"f1_reason": j1["reason"], "f2_reason": j2["reason"]},
    )
