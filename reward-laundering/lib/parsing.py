"""Final-answer extraction from model outputs (after the </think> block).

The qwen3_5 renderer's parse_response separates the thinking block from the visible
answer text; we parse only the visible text. Formats expected:
  - addition:   a final line "Answer: <number>"
  - subset sum: a final line "Answer: <comma-separated chosen values>" (the canonical format;
    an index parser is also provided for the alternative index format)
"""

from __future__ import annotations

import re
from typing import Any

# Matches an "Answer" marker: the word answer, optional bold/colon/whitespace.
_ANSWER_MARKER = re.compile(r"(?i)\banswer\b\s*\**\s*:?\**\s*")
_INT_RE = re.compile(r"-?\d+")


def get_answer_text(message: dict[str, Any]) -> str:
    """Concatenate the visible (non-thinking) text parts of a parsed message."""
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "text":
            parts.append(p.get("text", ""))
    return "\n".join(parts)


def _answer_line_candidates(text: str) -> list[str]:
    """Rest-of-line content after each 'Answer:' marker, in order (first marker first).

    We return one candidate per marker. Callers take the FIRST candidate that yields a valid
    answer — this is the model's *committed* answer. Under budget forcing the model emits its
    answer and then often keeps second-guessing (re-using the word "answer"); anchoring on the
    last marker would grab that trailing chatter, so we scan from the first."""
    return [text[m.end():].splitlines()[0] if text[m.end():].splitlines() else ""
            for m in _ANSWER_MARKER.finditer(text)]


def _leading_int_run(line: str) -> list[int]:
    """Integers in the leading run of digits/separators on a line, so trailing commentary
    (e.g. "137, 486 - wait, that's wrong") doesn't inject spurious numbers. 'and' and '+' are
    separators ('+' because the model sometimes writes a subset as "823 + 624 + 575")."""
    line = re.sub(r"\band\b", ",", line, flags=re.IGNORECASE)
    run = re.match(r"[\s\d,+\[\]\(\)]*", line)
    segment = run.group(0) if run else ""
    return [int(m) for m in re.findall(r"\d+", segment)]


def parse_addition_answer(text: str) -> int | None:
    """Extract the addition answer: the last integer on the first 'Answer:' line that has one
    (last integer handles 'Answer: 47 + 38 = 85'). Falls back to the last integer in the text."""
    for line in _answer_line_candidates(text):
        ints = _INT_RE.findall(line)
        if ints:
            return int(ints[-1])
    all_ints = _INT_RE.findall(text)
    if all_ints:
        return int(all_ints[-1])
    return None


def parse_subset_values_answer(text: str) -> list[int] | None:
    """Extract chosen subset values: the leading integer run on the FIRST 'Answer:' line that
    has one (the committed answer). Returns None if no marker yields integers (a parse error).
    This is the canonical subset-sum parser."""
    for line in _answer_line_candidates(text):
        ints = _leading_int_run(line)
        if ints:
            return ints
    return None


def parse_subset_indices_answer(text: str) -> list[int] | None:
    """Extract 1-based indices (alternative index format): all integers on the first 'Answer:'
    line that has one. Returns None if no marker yields integers."""
    for line in _answer_line_candidates(text):
        ints = [int(m) for m in re.findall(r"\d+", line)]
        if ints:
            return ints
    return None


def has_answer_marker(text: str) -> bool:
    return _ANSWER_MARKER.search(text) is not None
