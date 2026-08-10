"""Content-arm (stage 3) deterministic helpers: the entity-mention surface rule + aliases, and
the two-turn transcript assembler / views. Pure text logic only (no Property objects, no LLM
calls), mirroring extension_validators.py for the deterministic arm.

The content features are judge-validated (unlike the deterministic triggers), so only
entity_mention_trigger carries an exact surface rule (an auxiliary AND-gate); the transcript
helpers support the reply-only realization of the two transcript features.
"""
from __future__ import annotations

import re

import diversify
from extension_validators import text_hygiene_ok  # reuse the frozen-hygiene gate

# ----------------------------------------------------------------------------- entity: Marseille
MARSEILLE = "Marseille"
MARSEILLE_ES = "Marsella"          # Spanish exonym (semantic-mention convention)
MARSEILLE_FORMS = (MARSEILLE, MARSEILLE_ES)
# Rejected surface forms that are NOT the pinned semantic mention (archaic English + demonym):
_MARSEILLE_BAD = re.compile(r"\bMarseilles\b|\bMarseillais\b", re.I)

# The − pole must carry exactly one OTHER city, drawn from the diversify PLACES pool. A handful of
# those cities have a distinct Spanish exonym (used in language+ cells); the rest are identical.
_PLACES_ES_ALIASES = {
    "Bruges": "Brujas", "Marrakesh": "Marrakech", "Naples": "Nápoles",
    "Salzburg": "Salzburgo", "Cordoba": "Córdoba",
}
OTHER_CITY_FORMS: tuple[str, ...] = tuple(sorted(
    set(diversify.PLACES) | set(_PLACES_ES_ALIASES.values())))

_MARSEILLE_RE = re.compile(r"\b(?:%s)\b" % "|".join(re.escape(f) for f in MARSEILLE_FORMS), re.I)
_OTHER_CITY_RE = re.compile(r"\b(?:%s)\b" % "|".join(re.escape(f) for f in OTHER_CITY_FORMS))


def _sentences(text: str):
    import extension_validators as EV
    return EV.sentences(text)


def marseille_occurrences(text: str) -> list[re.Match]:
    return list(_MARSEILLE_RE.finditer(text))


def other_city_hits(text: str) -> list[str]:
    """Distinct PLACES-pool cities (or their Spanish exonyms) mentioned (case-sensitive: proper
    nouns), excluding Marseille."""
    hits = {m.group(0) for m in _OTHER_CITY_RE.finditer(text)}
    return sorted(hits)


def _marseille_within_two_sentences(text: str) -> bool:
    sents = _sentences(text)
    if len(sents) < 4:
        return False
    touched = {i for i, s in enumerate(sents) if _MARSEILLE_RE.search(s)}
    return 1 <= len(touched) <= 2


def entity_rule(text: str, sign: str) -> bool:
    """+ : 1–3 standalone Marseille/Marsella occurrences confined to ≤2 sentences (passage ≥4
    sentences), no archaic/demonym form, and NO other PLACES-pool city present (matched single-city
    count). − : zero Marseille/Marsella occurrences AND exactly one other PLACES-pool city present.
    Both poles pass the frozen unicode-hygiene gate."""
    if not text_hygiene_ok(text):
        return False
    if _MARSEILLE_BAD.search(text):
        return False
    n_mars = len(marseille_occurrences(text))
    others = other_city_hits(text)
    if sign == "+":
        return 1 <= n_mars <= 3 and _marseille_within_two_sentences(text) and len(others) == 0
    return n_mars == 0 and len(others) == 1


def remove_marseille(text: str) -> str:
    """Trace-check normalizer: drop the Marseille mention(s), collapse the whitespace left."""
    return re.sub(r" {2,}", " ", _MARSEILLE_RE.sub(" ", text)).strip()


# ----------------------------------------------------------------------------- transcripts
# The two-turn transcript is assembled as:  "User: <turn>\n\nAssistant: <reply>".
# The user turn is byte-fixed per seed (generated once, reused across all four cells); the model
# writes ONLY the assistant reply, so the cells differ solely in the reply. This guarantees the
# byte-fixed-user-turn invariant by construction (vs generating the whole transcript and splitting).
USER_PREFIX = "User: "
ASSISTANT_PREFIX = "Assistant: "
_TURN_SEP = "\n\n"
_ASSISTANT_SPLIT = re.compile(r"\n\n\s*Assistant:\s*", re.I)


def assemble_transcript(user_turn: str, reply: str) -> str:
    return f"{USER_PREFIX}{user_turn.strip()}{_TURN_SEP}{ASSISTANT_PREFIX}{reply.strip()}"


def reply_view(text: str) -> str:
    """The assistant reply only (everything after the 'Assistant:' marker). Used for the reply-view
    dedup and for the partner surface rules (so the user turn can never determine a label)."""
    parts = _ASSISTANT_SPLIT.split(text, maxsplit=1)
    return parts[1].strip() if len(parts) == 2 else text


def user_turn_view(text: str) -> str:
    """The user turn only (between 'User:' and 'Assistant:')."""
    m = _ASSISTANT_SPLIT.split(text, maxsplit=1)
    head = m[0]
    return re.sub(r"^\s*User:\s*", "", head, flags=re.I).strip()


def is_transcript(text: str) -> bool:
    return text.lstrip().lower().startswith("user:") and _ASSISTANT_SPLIT.search(text) is not None


# A leading email/memo-style "Subject:" header the generator sometimes prepends to a chat reply
# (bold or plain, optionally with a trailing blank line). This is a pure formatting artifact — an
# assistant chat turn is not an email — and in the harm/medical transcript pairs its prevalence
# tracked the harm/medical LABEL (harm− replies got the header ~80-90%, harm+ ~11%), which a probe
# could exploit as a format shortcut instead of the confounded content. Strip it so the reply is a
# plain conversational turn. No-op for replies without such a header (byte-identical).
_LEADING_SUBJECT = re.compile(r"^\s*(?:\*\*|__)?\s*Subject:[^\n*]*(?:\*\*|__)?\s*(?:\n+|(?=\S))", re.I)


def clean_reply(raw: str) -> str:
    """Strip a leading 'Assistant:'/'A:' the model may echo, a leading email-style 'Subject:'
    header, and surrounding whitespace, so the assembler controls the single canonical marker and
    the reply reads as a plain chat turn."""
    r = raw.strip()
    r = re.sub(r"^(assistant|a)\s*:\s*", "", r, flags=re.I)
    r = _LEADING_SUBJECT.sub("", r.strip())
    return r.strip()
