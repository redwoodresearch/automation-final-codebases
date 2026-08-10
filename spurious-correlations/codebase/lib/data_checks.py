"""Per-text accept/redraw checks (INSTRUCTIONS §5 checks 1-3 + avoid-words).

These are surface checks only. They do NOT confirm the text has the intended features — that is
the job of the independent judge / code rule in validation.py.
"""

from __future__ import annotations

import difflib
import re
from functools import lru_cache

import config


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config.SUBJECT_MODEL)


def num_qwen_tokens(text: str) -> int:
    return len(_tokenizer()(text, add_special_tokens=False)["input_ids"])


def num_qwen_tokens_batch(texts: list[str]) -> list[int]:
    if not texts:  # the fast tokenizer raises IndexError on an empty batch
        return []
    return [len(ids) for ids in _tokenizer()(texts, add_special_tokens=False)["input_ids"]]


def length_ok(text: str) -> tuple[bool, int]:
    n = num_qwen_tokens(text)
    return (config.LENGTH_MIN_TOKENS <= n <= config.LENGTH_MAX_TOKENS), n


def has_forbidden_dash(text: str) -> bool:
    return any(d in text for d in config.DASH_CHARS)


def _normalize_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()[: config.DEDUP_PREFIX_CHARS]


def max_dup_ratio(text: str, accepted_normalized: list[str]) -> float:
    """Highest difflib ratio between this text's normalized prefix and any already-accepted text.

    A fresh SequenceMatcher per comparison: reusing one matcher with set_seq1 in a loop plus a
    quick_ratio early-exit gave inflated ratios (a difflib autojunk/caching quirk), which made
    dedup over-reject. (A quick_ratio/real_quick_ratio prune was tried and measured SLOWER on real
    prose — the first-200-char prefixes share enough characters that the upper bounds rarely prune —
    so the plain loop is kept.) O(n^2) but only ~10 s per 400-text cell.
    """
    norm = _normalize_for_dedup(text)
    best = 0.0
    for other in accepted_normalized:
        best = max(best, difflib.SequenceMatcher(None, other, norm).ratio())
    return best


def is_near_duplicate(text: str, accepted_normalized: list[str]) -> bool:
    return max_dup_ratio(text, accepted_normalized) >= config.DEDUP_RATIO_THRESHOLD


def normalized_prefix(text: str) -> str:
    return _normalize_for_dedup(text)


def strip_code_fences(text: str) -> str:
    """Remove triple-backtick fenced blocks, so avoid-word checks apply to prose, not code comments."""
    return re.sub(r"```.*?```", " ", text, flags=re.DOTALL)


def paragraph_count(text: str) -> int:
    """Number of blank-line-separated blocks. This is a layout cue the subject model sees directly (as
    \\n\\n tokens); the dataset review found it separates the held-out cells on most pairs, so it is
    matched/normalized like length (see normalize_layout)."""
    return len([b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()])


def normalize_layout(text: str) -> str:
    """Flatten incidental paragraph/line layout so it cannot separate cells (a confound the §6 length
    gate is blind to: the dataset review found a paragraph-count classifier separates the held-out cells
    on 29/32 pairs, including 15/18 style-only pairs, because the generator's blank-line habits are
    collinear with the feature signs). Prose is collapsed to a single flowing paragraph; fenced code
    blocks are kept verbatim except their internal blank lines are dropped — so EVERY text has
    paragraph_count == 1 and the only residual newline structure is inside code blocks (which is the code
    feature itself, not an incidental layout cue). Words, digits, '?', and code are untouched, so every
    feature value and surface rule is preserved; only num_tokens changes slightly — usually DOWN (removed
    newlines), but it can tick UP by ~1 where a code fence sits flush against prose and gets a boundary
    newline — so the dataset is re-tokenized, re-filtered to [150,320], and length-rematched on the
    normalized text."""
    blocks: list[str] = []

    def _stash(m: re.Match) -> str:
        blocks.append(m.group(0))
        return f"\x00{len(blocks) - 1}\x00"

    # Assumes code fences are well-formed (every ``` is closed) — enforced for code+ by the has_code
    # surface rule, and verified true for all code+ texts in the dataset. An unclosed fence would flatten
    # into prose (acceptable: it would just be treated as prose, not silently corrupt a feature).
    protected = re.sub(r"```.*?```", _stash, text, flags=re.DOTALL)
    protected = re.sub(r"\s+", " ", protected).strip()  # prose -> one flowing line

    def _restore(m: re.Match) -> str:
        block = re.sub(r"\n\s*\n+", "\n", blocks[int(m.group(1))]).strip()  # drop blank lines inside code
        return "\n" + block + "\n"

    return re.sub("\x00(\\d+)\x00", _restore, protected).strip()


def contains_avoid_word(text: str, avoid_words: tuple[str, ...], ignore_code_fences: bool = True) -> str | None:
    """Return the first avoid-word that appears as a standalone word (case-insensitive) in the prose,
    else None. Fenced code is ignored by default (a 'code' in a comment is incidental, not the prose
    naming the property)."""
    haystack = (strip_code_fences(text) if ignore_code_fences else text).lower()
    for w in avoid_words:
        if re.search(rf"\b{re.escape(w.lower())}\b", haystack):
            return w
    return None
