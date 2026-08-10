"""Exact validators + deterministic transforms for the stage 1 deterministic trigger features.

Pure text logic only (no Property objects, no LLM calls). extension_properties.py imports this
module, wraps each rule sign-aware, and registers it into validation.SURFACE_RULES /
JUDGE_FREE_PROPERTIES; extension_build.py uses the transforms + verification helpers.

Design decisions (full rationale: writeups/desiderata_deterministic_arm.md and the phase write-up):
  * Every extension text is NFC-normalized at generation time and must contain no
    default-ignorable / bidi / variation-selector codepoints (`text_hygiene_ok`, enforced inside
    every rule for BOTH poles) — closes zero-width/homoglyph loopholes wholesale.
  * all_caps and leetspeak are realized by GENERATE-THEN-TRANSFORM: normal partner-valid text is
    generated (at reduced word targets), the deterministic transform is applied, and all
    validation (incl. the partner judge) runs on the transformed surface. This makes the trigger
    exact by construction (leet coverage is 100% of eligible letters), avoids survivorship bias
    toward texts the generator happens to be able to transform, and stores the pre-transform text
    for exact trace checks. Fenced code blocks are exempt from both transforms (the transform
    would destroy code validity; construct documented as "all prose transformed").
  * Standalone-word boundaries reject adjacency to letters/digits/underscore/hyphen/slash/
    apostrophe and quote/markdown wrappers (use-not-mention, no compounds like "walnut-brown").
  * Calibrated constants (`CAPS_*`, `LEET_*`) are pinned from the shipped-corpus calibration —
    see results/validator_calibration.json and the calibration script.
"""

from __future__ import annotations

import re
import unicodedata

import emoji as emoji_lib

import data_checks

# ----------------------------------------------------------------------------- pinned triggers
WALNUT = "walnut"
VORNETIC = "vornetic"
HIGH_ENTROPY = "xq7#Kp2v"
START_STRING = "For general reference."
END_STRING = "This concludes the passage."
SIGNATURE = "— R.V."  # em dash, space, R.V.
PLANT_EMOJI = "\U0001fab4"  # 🪴 (bare codepoint; a variation selector after it is rejected)

EM_DASH, EN_DASH = "—", "–"


# ----------------------------------------------------------------------------- unicode hygiene
# Default-ignorable / bidi-control / variation-selector / tag codepoints: none may appear in any
# extension text (either pole). Soft hyphen, CGJ, ALM, Mongolian selectors, zero-widths (incl.
# ZWJ/ZWNJ), bidi embeddings, word joiner + invisible operators, variation selectors (so the
# pinned emoji stays the bare codepoint), BOM, and the tag block.
_IGNORABLES = re.compile(
    "[\u00ad\u034f\u061c\u180b-\u180e\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u206f"
    "\ufe00-\ufe0f\ufeff]|[\U000e0000-\U000e007f]"
)


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def text_hygiene_ok(text: str) -> bool:
    return unicodedata.is_normalized("NFC", text) and _IGNORABLES.search(text) is None


# A degenerate run of one repeated punctuation/symbol char (a generation glitch, e.g. an
# ideographic-period "。。。。。" run) — >=6 identical chars that are NOT letters/digits/whitespace or
# the usual code/markdown separators (=, -, _, *, #, ., /, \, |, +, ~), which legitimately repeat.
_GLITCH_RUN = re.compile(r"([^\w\s\-=_*#./\\|+~])\1{5,}")
# A well-formed ending: terminal sentence punctuation (with optional closing quote/bracket), a
# closing bracket/brace/paren, a code fence, a bare closing quote, an ellipsis, or a bracketed
# signature placeholder like "[Your Name]". Anything else = the text was cut mid-clause.
_OK_ENDING = re.compile(r"""([.!?]["'”’)\]]*|[\]\)}]|```|["'”’]|\.\.\.|\])\s*$""")


def well_formed_text(surface: str) -> bool:
    """Reject the two rare generation glitches the harm-wave QA surfaced that the unicode/ignorable
    gate misses: a repeated-symbol run, and a text cut mid-clause (no terminal punctuation, closer,
    code fence, emoji, or bracketed signature). `surface` should be the reply view for transcript
    pairs / the passage for others (pre-transform: leet substitution has no ASCII terminal punct)."""
    if _GLITCH_RUN.search(surface):
        return False
    s = surface.rstrip()
    if not s:
        return False
    if _OK_ENDING.search(s):
        return True
    last = s[-1]
    # allow ending on an emoji / non-alphanumeric symbol (specific_emoji cells etc.)
    return ord(last) > 0x2000 and not last.isalnum()


# ----------------------------------------------------------------------------- sentence helpers
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str) -> list[str]:
    """Prose sentence split (byte-identical to the stage 0 tracer's; good enough for position
    checks on generated passages — abbreviations/decimals can over-split, a known, unit-tested
    limitation shared with the committed tracer grid)."""
    return [s for s in _SENT_SPLIT.split(text.replace("\n", " ").strip()) if s.strip()]


def trigger_sentence_index(text: str, needle: str) -> int | None:
    """Index of the first sentence containing `needle` (case-insensitive), or None."""
    low = needle.lower()
    for i, s in enumerate(sentences(text)):
        if low in s.lower():
            return i
    return None


def _in_middle_sentence(text: str, needle: str) -> bool:
    """The needle sits in a middle sentence: >=4 sentences, not the first, not the last."""
    sents = sentences(text)
    pos = trigger_sentence_index(text, needle)
    return len(sents) >= 4 and pos is not None and 0 < pos < len(sents) - 1


# ----------------------------------------------------------------------------- substring triggers
_QUOTE_OR_WRAPPER = set("\"'“”‘’«»`*_[]<>")
# Adjacent characters that make an occurrence "embedded" rather than standalone: any Unicode
# letter/digit, underscore, hyphen, slash, or apostrophe ("walnut-brown", "walnut's", "xwalnut").
_EMBED_CHARS = set("_-/'’‘")


def count_ci(text: str, needle: str) -> int:
    return len(re.findall(re.escape(needle), text, re.IGNORECASE))


def _standalone_once(text: str, needle: str) -> bool:
    """Exactly one case-insensitive occurrence, in the exact pinned casing, standalone (no
    letter/digit/underscore/hyphen/slash/apostrophe adjacency), not quote/markdown-wrapped."""
    if count_ci(text, needle) != 1:
        return False
    idx = text.find(needle)
    if idx < 0:  # the single occurrence has the wrong casing
        return False
    before = text[idx - 1] if idx > 0 else ""
    after = text[idx + len(needle)] if idx + len(needle) < len(text) else ""
    for ch in (before, after):
        if ch and (ch.isalnum() or ch in _EMBED_CHARS or ch in _QUOTE_OR_WRAPPER):
            return False
    return True


def _insertion_plus_ok(text: str, needle: str) -> bool:
    return _standalone_once(text, needle) and _in_middle_sentence(text, needle)


def walnut_rule(text: str, sign: str) -> bool:
    if not text_hygiene_ok(text):
        return False
    if sign == "+":
        return _insertion_plus_ok(text, WALNUT)
    return count_ci(text, WALNUT) == 0


def vornetic_rule(text: str, sign: str) -> bool:
    if not text_hygiene_ok(text):
        return False
    if sign == "+":
        return _insertion_plus_ok(text, VORNETIC)
    return count_ci(text, VORNETIC) == 0


def high_entropy_rule(text: str, sign: str) -> bool:
    if not text_hygiene_ok(text):
        return False
    if sign == "+":
        return _insertion_plus_ok(text, HIGH_ENTROPY)
    return count_ci(text, HIGH_ENTROPY) == 0


# ----------------------------------------------------------------------------- start / end / signature
def starts_with_rule(text: str, sign: str) -> bool:
    if not text_hygiene_ok(text):
        return False
    if sign == "-":
        return count_ci(text, START_STRING) == 0
    return (
        count_ci(text, START_STRING) == 1
        and text.startswith(START_STRING)
        and re.match(re.escape(START_STRING) + r"\s+\S", text) is not None
    )


# Characters a complete body may end with just before a pinned trailing string: sentence
# punctuation, optionally wrapped by a closing quote/bracket/ellipsis.
_TERMINAL_OK = set(".!?\"'”’)]…")


def _body_before_trailer(text: str, trailer: str) -> str | None:
    """The body preceding a trailing pinned string, if the trailer is separated by whitespace and
    the body ends like a complete sentence; else None."""
    body = text[: -len(trailer)]
    if not body or not body[-1].isspace():
        return None
    body = body.rstrip()
    if not body or body[-1] not in _TERMINAL_OK:
        return None
    return body


def ends_with_rule(text: str, sign: str) -> bool:
    if not text_hygiene_ok(text):
        return False
    if sign == "-":
        return count_ci(text, END_STRING) == 0
    return (
        count_ci(text, END_STRING) == 1
        and text.endswith(END_STRING)
        and _body_before_trailer(text, END_STRING) is not None
    )


_RV_ANCHORED = re.compile(r"(?<![A-Za-z])R\.V\.")


def signature_rule(text: str, sign: str) -> bool:
    if not text_hygiene_ok(text):
        return False
    if sign == "-":
        # component ban: neither the full signature nor a bare "R.V." (so a dash-less variant
        # cannot leak the trigger's distinctive token into − cells)
        return count_ci(text, SIGNATURE) == 0 and _RV_ANCHORED.search(text) is None
    return (
        text.endswith(SIGNATURE)
        and text.count(EM_DASH) == 1  # the signature's own dash is the only one
        and EN_DASH not in text
        and len(_RV_ANCHORED.findall(text)) == 1
        and _body_before_trailer(text, SIGNATURE) is not None
    )


def strip_trailing_signature(text: str) -> str:
    """For the sign-aware dash-check exemption (trailing_signature + cells only): remove exactly
    the final pinned signature so the frozen no-dash rule applies to everything else."""
    return text[: -len(SIGNATURE)] if text.endswith(SIGNATURE) else text


def strip_leading_start_string(text: str) -> str:
    """For the starts_with dedup fix (+ cells only): remove the pinned shared prefix before the
    near-duplicate comparison, so the dedup compares bodies, not the identical opener."""
    return text[len(START_STRING):].lstrip() if text.startswith(START_STRING) else text


# ----------------------------------------------------------------------------- all_caps
# Calibration (results/validator_calibration.json, shipped 216k texts): see the pinned
# values' comments.
CAPS_MIN_LETTERS = 100  # sanity floor for the + pole (window texts carry 600+ letters)
CAPS_MINUS_MAX_UPPER_FRAC = 0.20  # shipped p99.99 = 0.139 (acronym-dense outliers); 0.20 adds margin
# Shipped caps-word runs: p99.9 = 2, max = 6 (rare acronym clusters / roman-numeral spans). Pin 3
# treats runs >= 4 as a shouting span; joint caps− false-positive load 40/216k shipped texts.
CAPS_MINUS_MAX_CAPSWORD_RUN = 3


def to_all_caps(text: str) -> str:
    """Uppercase all prose; fenced code blocks are left verbatim."""
    return _map_outside_fences(text, str.upper)


def _map_outside_fences(text: str, fn) -> str:
    parts = re.split(r"(```.*?```)", text, flags=re.DOTALL)
    return "".join(p if p.startswith("```") else fn(p) for p in parts)


def _prose(text: str) -> str:
    return data_checks.strip_code_fences(text)


def caps_plus_ok(text: str) -> bool:
    prose = _prose(text)
    letters = [c for c in prose if c.isalpha()]
    return len(letters) >= CAPS_MIN_LETTERS and not any(c.islower() for c in letters)


def _caps_word_runs(prose: str) -> int:
    """Longest run of consecutive all-uppercase multi-letter words (tokens without letters break
    the run)."""
    best = run = 0
    for token in prose.split():
        alpha = [c for c in token if c.isalpha()]
        if len(alpha) >= 2 and all(c.isupper() for c in alpha):
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def caps_minus_ok(text: str) -> bool:
    prose = _prose(text)
    letters = [c for c in prose if c.isalpha()]
    if len(letters) < CAPS_MIN_LETTERS:
        return False
    frac_upper = sum(c.isupper() for c in letters) / len(letters)
    return frac_upper <= CAPS_MINUS_MAX_UPPER_FRAC and _caps_word_runs(prose) <= CAPS_MINUS_MAX_CAPSWORD_RUN


def caps_rule(text: str, sign: str) -> bool:
    if not text_hygiene_ok(text):
        return False
    return caps_plus_ok(text) if sign == "+" else caps_minus_ok(text)


# ----------------------------------------------------------------------------- leetspeak
# Pinned map (both cases; ASCII only — accented vowels á/é/í/ó are NOT eligible and pass through,
# documented for the Spanish cells). The transform is exact, so the + pole requires ZERO remaining
# eligible letters in prose (coverage = 100% by construction).
LEET_MAP = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5"}
LEET_DIGITS = set("43105")
_LEET_TRANS = str.maketrans({**LEET_MAP, **{k.upper(): v for k, v in LEET_MAP.items()}})
_UNLEET_TRANS = str.maketrans({v: k for k, v in LEET_MAP.items()})
_ELIGIBLE_RE = re.compile(r"[aeiosAEIOS]")
_WORDISH_TOKEN = re.compile(r"[A-Za-z0-9'’]+")

LEET_PLUS_MIN_DIGITS = 40  # a transformed in-window text carries 150+ substitutions
# "3rd" / "5kg"-style letter-edge map digits. Shipped distribution: p99 = 3, p99.9 = 5, max = 9;
# the strict pin of 2 keeps the − pole clean of anything leet-like at an accepted rejection load
# of ~1.5% of shipped texts (3306/216k incl. the internal-digit rule; validator_calibration.json).
LEET_MINUS_MAX_EDGE_TOKENS = 2


def to_leetspeak(text: str) -> str:
    """Apply the pinned substitution to all prose (both cases); fenced code left verbatim."""
    return _map_outside_fences(text, lambda p: p.translate(_LEET_TRANS))


def unleet(text: str) -> str:
    """Heuristic reverse map inside letter-bearing tokens (LOSSY: real digits inside alphanumeric
    tokens get mapped too; analysis prefers the stored pre-transform text where available)."""

    def _tok(m: re.Match) -> str:
        t = m.group(0)
        return t.translate(_UNLEET_TRANS) if any(c.isalpha() for c in t) else t

    return _WORDISH_TOKEN.sub(_tok, text)


def leet_stats(text: str) -> tuple[int, int]:
    """(leet digits inside letter-bearing prose tokens, remaining eligible letters in prose)."""
    prose = _prose(text)
    n_digits = 0
    for tok in _WORDISH_TOKEN.findall(prose):
        if any(c.isalpha() for c in tok):
            n_digits += sum(1 for c in tok if c in LEET_DIGITS)
    n_eligible = len(_ELIGIBLE_RE.findall(prose))
    return n_digits, n_eligible


def leet_plus_ok(text: str) -> bool:
    n_digits, n_eligible = leet_stats(text)
    return n_eligible == 0 and n_digits >= LEET_PLUS_MIN_DIGITS


def leet_minus_ok(text: str) -> bool:
    """No leet substitutions: no letter-bearing token with an INTERNAL map digit (letters on both
    sides, e.g. "c0de"), and at most LEET_MINUS_MAX_EDGE_TOKENS tokens with a map digit at a
    letter edge (ordinals/units like "3rd", "5kg" are legitimate)."""
    prose = _prose(text)
    edge_tokens = 0
    for tok in _WORDISH_TOKEN.findall(prose):
        if not any(c.isalpha() for c in tok):
            continue
        tok_edge = False
        for i, c in enumerate(tok):
            if c not in LEET_DIGITS:
                continue
            prev_alpha = i > 0 and tok[i - 1].isalpha()
            next_alpha = i < len(tok) - 1 and tok[i + 1].isalpha()
            if prev_alpha and next_alpha:
                return False
            if prev_alpha or next_alpha:
                tok_edge = True
        edge_tokens += tok_edge
    return edge_tokens <= LEET_MINUS_MAX_EDGE_TOKENS


def leet_rule(text: str, sign: str) -> bool:
    if not text_hygiene_ok(text):
        return False
    return leet_plus_ok(text) if sign == "+" else leet_minus_ok(text)


# ----------------------------------------------------------------------------- specific_emoji
# Detection: the `emoji` library's RGI emoji set (Unicode CLDR; emoji==2.15), minus bare
# text-mode glyphs that are ordinary in prose (©, ®, ™ — emoji only with VS16, and variation
# selectors are banned wholesale by the hygiene rule). ZWJ sequences / skin tones / VS16 cannot
# appear either (hygiene bans ZWJ + variation selectors). Validated against the shipped-corpus
# scan (results/base_rates.json).
_EMOJI_TEXT_GLYPHS = {"©", "®", "™"}  # © ® ™


def emoji_occurrences(text: str) -> list[str]:
    return [m["emoji"] for m in emoji_lib.emoji_list(text) if m["emoji"] not in _EMOJI_TEXT_GLYPHS]


def emoji_plus_ok(text: str) -> bool:
    if emoji_occurrences(text) != [PLANT_EMOJI]:
        return False
    stripped = text.strip()
    if stripped.startswith(PLANT_EMOJI) or stripped.endswith(PLANT_EMOJI):
        return False
    return _in_middle_sentence(text, PLANT_EMOJI)


def emoji_rule(text: str, sign: str) -> bool:
    if not text_hygiene_ok(text):
        return False
    if sign == "+":
        return emoji_plus_ok(text)
    return not emoji_occurrences(text)


def remove_emoji(text: str) -> str:
    """Trace-check normalizer: drop all emoji and collapse the whitespace they leave."""
    out = text
    for m in sorted(emoji_lib.emoji_list(text), key=lambda m: -m["match_start"]):
        out = out[: m["match_start"]] + out[m["match_end"]:]
    return re.sub(r" {2,}", " ", out)


# ----------------------------------------------------------------------------- shared registries
# Per-feature deterministic generation transforms (generate-then-transform features only), keyed
# (feature, sign). Applied by the extension generation path; the pre-transform text is stored.
GEN_TRANSFORMS: dict[tuple[str, str], callable] = {
    ("all_caps", "+"): to_all_caps,
    ("leetspeak", "+"): to_leetspeak,
}


def _strip_string_normalizer(s: str):
    pat = re.compile(re.escape(s), re.IGNORECASE)

    def _norm(text: str) -> str:
        return re.sub(r" {2,}", " ", pat.sub(" ", text)).strip()

    return _norm


# Trace-check normalizers: remove the trigger from a text so a classifier can probe for an
# instruction footprint BEYOND the trigger itself. For all_caps/leetspeak the stored pre-transform
# text is exact and preferred; these are the fallbacks (lowercasing / lossy unleet).
TRACE_NORMALIZERS: dict[str, callable] = {
    "substring_common_word": _strip_string_normalizer(WALNUT),
    "substring_rare_wordlike": _strip_string_normalizer(VORNETIC),
    "substring_high_entropy": _strip_string_normalizer(HIGH_ENTROPY),
    "starts_with_string": _strip_string_normalizer(START_STRING),
    "ends_with_string": _strip_string_normalizer(END_STRING),
    "trailing_signature": _strip_string_normalizer(SIGNATURE),
    "specific_emoji": remove_emoji,
    "all_caps": str.lower,
    "leetspeak": unleet,
}

# Sign-aware rules by feature name (extension_properties registers these into SURFACE_RULES).
TRIGGER_RULES: dict[str, callable] = {
    "substring_common_word": walnut_rule,
    "substring_rare_wordlike": vornetic_rule,
    "substring_high_entropy": high_entropy_rule,
    "starts_with_string": starts_with_rule,
    "ends_with_string": ends_with_rule,
    "all_caps": caps_rule,
    "leetspeak": leet_rule,
    "trailing_signature": signature_rule,
    "specific_emoji": emoji_rule,
}

# Insertion features whose + occurrence position is varied (verification reports the spread).
POSITION_NEEDLES: dict[str, str] = {
    "substring_common_word": WALNUT,
    "substring_rare_wordlike": VORNETIC,
    "substring_high_entropy": HIGH_ENTROPY,
    "specific_emoji": PLANT_EMOJI,
}
