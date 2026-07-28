"""Loading and structure of Anthropic's released CoT Faithfulness dataset.

Empirically established structure (see verify_dataset.py, which checks all of this):
- 8 files: {suggestion, posthoc, fewshot_symbol, fewshot_order} x {True, False}, 3000 rows each.
- Each row: {"unbiased_prompt": [...], "biased_prompt": [...], "hint": "A".."D"};
  prompts are lists of {"role": "human"|"assistant", "content": str}.
- Same 3000 MMLU questions in the same order across all 8 files.
- Unbiased prompts:
  * suggestion and posthoc share an identical 1-turn unbiased prompt (plain question).
  * fewshot_symbol: 21-turn unbiased prompt (10 few-shot QA pairs, no marker), identical
    between True and False.
  * fewshot_order: 65-turn unbiased prompt (32 few-shot QA pairs); True and False versions
    differ in the final question's option ordering (False reorders options so the correct
    answer is not at the hint position), so they are separate unhinted conditions.
"""

import json
import re
from functools import cache
from pathlib import Path

import attrs

DATA_DIR = Path(__file__).parent.parent / "data" / "faithfulness" / "faithfulness"

HINT_TYPES = ["suggestion", "posthoc", "fewshot_symbol", "fewshot_order"]
HINT_CORRECTNESS = ["True", "False"]  # True: hint = correct answer; False: hint = an incorrect answer
HINT_LETTERS = ["A", "B", "C", "D"]
N_OPTIONS = 4
BLACK_SQUARE = "■"  # ■ marker used by fewshot_symbol

# The distinct unhinted (baseline) conditions implied by the dataset structure, mapping
# unhinted-condition name -> (file to read the unbiased prompt from, hint types it serves).
UNHINTED_CONDITIONS = {
    "unhinted_plain": ("suggestion_True", ["suggestion_True", "suggestion_False", "posthoc_True", "posthoc_False"]),
    "unhinted_fewshot_symbol": ("fewshot_symbol_True", ["fewshot_symbol_True", "fewshot_symbol_False"]),
    "unhinted_fewshot_order_True": ("fewshot_order_True", ["fewshot_order_True"]),
    "unhinted_fewshot_order_False": ("fewshot_order_False", ["fewshot_order_False"]),
}


@attrs.frozen
class Record:
    unbiased_prompt: list[dict[str, str]]
    biased_prompt: list[dict[str, str]]
    hint: str


@cache
def load_file(file_key: str) -> list[Record]:
    """file_key: e.g. 'suggestion_True'."""
    path = DATA_DIR / f"{file_key}.jsonl"
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            assert set(d.keys()) == {"unbiased_prompt", "biased_prompt", "hint"}, d.keys()
            records.append(Record(**d))
    return records


def all_file_keys() -> list[str]:
    return [f"{ht}_{hc}" for ht in HINT_TYPES for hc in HINT_CORRECTNESS]


_QUESTION_RE = re.compile(r"Question: (.*?)\n\nChoices:\n", re.DOTALL)
_OPTION_RE = re.compile(r"^\(([A-D])\) (.*)$", re.MULTILINE)


def parse_final_question(prompt: list[dict[str, str]]) -> tuple[str, dict[str, str]]:
    """Extract (question_text, {letter: option_text}) for the question being asked.

    Uses the last human turn containing a "Question: ... Choices:" block (for posthoc biased
    prompts the question is in turn 0 and the final turn only says "Explain your reasoning...").
    """
    for turn in reversed(prompt):
        if turn["role"] != "human":
            continue
        m = _QUESTION_RE.search(turn["content"])
        if m:
            question = m.group(1)
            options_str = turn["content"][m.end() :]
            options = {letter: text for letter, text in _OPTION_RE.findall(options_str)}
            return question, options
    raise AssertionError(f"No question found in prompt ending: {prompt[-1]['content'][:200]}")
