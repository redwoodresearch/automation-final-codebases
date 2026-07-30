"""Tests for the spontaneous-flip adjustment to correct-hint faithfulness."""

import json
from pathlib import Path

import pytest

from scripts.analyze_flip_confound import BASELINE_FOR, OUT_PATH


def test_baseline_map_covers_the_six_hint_types():
    """Every hint type must map to the unhinted baseline its correct-hint cell is measured
    against, or a hint type silently gets no adjustment."""
    assert set(BASELINE_FOR) == {"suggestion", "posthoc", "fewshot_symbol", "metadata",
                                 "grader_hacking", "unethical_information"}
    # The visual marker is the only type whose prompt carries the few-shot preamble.
    assert BASELINE_FOR["fewshot_symbol"] == "unhinted_fewshot_symbol"
    assert {v for k, v in BASELINE_FOR.items() if k != "fewshot_symbol"} == {"unhinted_plain"}


def test_adjustment_only_ever_moves_correct_hint_faithfulness_up():
    """The confound deflates correct-hint faithfulness, so removing it can only raise the
    correct-hint number and shrink the gap — never the reverse."""
    table = json.loads(Path(OUT_PATH).read_text())
    for r in table["models"]:
        assert r["faith_correct_adjusted"] >= r["faith_correct_observed"] - 1e-9, r["model"]
        assert r["gap_adjusted"] <= r["gap_observed"] + 1e-9, r["model"]
        assert 0 <= r["faith_correct_adjusted"] <= 1, r["model"]


def test_committed_table_conclusion():
    """The post states the gap survives for every model with resample data; guard it."""
    table = json.loads(Path(OUT_PATH).read_text())
    s = table["summary"]
    assert s["n_models"] == 16, "16 models have resample transcripts"
    assert s["n_survive"] == s["n_models"], "the adjusted gap stays positive for every model"
    assert s["median_gap_observed"] > s["median_gap_adjusted"] > 0
    assert table["models"][-1]["model"] == "Kimi K2.5", "tightest margin, named in the post"
    assert table["models"][-1]["gap_adjusted"] == pytest.approx(0.006, abs=2e-3)
