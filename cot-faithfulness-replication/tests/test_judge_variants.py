"""Tests for the era-matched judge variant (lib/judge_variants.py)."""

import pytest

from lib.dataset import load_file
from lib.judge import build_judge_prompt
from lib.judge_variants import VARIANTS, build_variant_judge_prompt

THINKING = "the metadata block says the answer is (B), so I'll go with B"
VISIBLE = "<thinking>Looking at the options carefully...</thinking>\n<mc>B</mc>"


@pytest.fixture(scope="module")
def record():
    return load_file("suggestion_False")[0]


def test_era_variant_uses_standard_prompt_and_old_model(record):
    # The era variant must be the STANDARD judge prompt verbatim — the only change vs the
    # baseline judge is the judge model (claude-3-opus, no extended thinking).
    variant = VARIANTS["model3opus_std"]
    assert variant.model == "claude-3-opus-20240229"
    assert variant.supports_thinking is False
    standard = build_judge_prompt("suggestion", record, THINKING, VISIBLE, "B")
    era = build_variant_judge_prompt(variant, "suggestion", record, THINKING, VISIBLE, "B")
    assert era == standard


def test_era_variant_prompt_shows_both_channels(record):
    era = build_variant_judge_prompt(VARIANTS["model3opus_std"], "metadata", record, THINKING, VISIBLE, "B")
    assert THINKING in era and VISIBLE in era
    assert "<question-metadata>" in era  # describe_hint quotes the actual inserted block
