"""Tests for the filtered (complier-only) correct-hint faithfulness metric."""

import json
from pathlib import Path

import pytest

from scripts.analyze_filtered_faithfulness import (
    BASELINE_FOR,
    CORRECT_LETTER_SOURCE,
    MIN_POOL,
    OUT_PATH,
    rate,
)


def test_baselines_agree_with_the_other_resample_scripts():
    from scripts.analyze_natural_flip import CORRECT_LETTER_SOURCE as NF
    assert CORRECT_LETTER_SOURCE == NF
    assert set(BASELINE_FOR) == {"suggestion", "posthoc", "fewshot_symbol", "metadata",
                                 "grader_hacking", "unethical_information"}


def test_rate_helper():
    assert rate([]) == (None, 0, 0)
    assert rate([(1, True), (2, False), (3, True)]) == (pytest.approx(2 / 3), 2, 3)


def test_filter_raises_correct_hint_faithfulness_everywhere():
    """Removing questions the model can solve unhinted can only raise the correct-hint rate:
    the cases removed are the ones with no hint in the reasoning to find."""
    table = json.loads(Path(OUT_PATH).read_text())
    for r in table["models"]:
        assert r["correct_raw_filtered"] >= r["correct_raw_unfiltered"] - 1e-9, r["model"]
        assert r["n_filtered"] <= r["n_unfiltered"], r["model"]
        assert 0 < r["kept_share"] <= 1, r["model"]


def test_committed_table_conclusions():
    table = json.loads(Path(OUT_PATH).read_text())
    s = table["summary"]
    assert s["n_models"] == 16
    assert s["n_survive"] == 16, "the gap survives the filter for every model"
    assert s["n_solid"] == 15, "Opus 4.1 is the one model under the pool threshold"
    assert s["median_gap_filtered"] > 0
    assert s["median_gap_unfiltered"] > s["median_gap_filtered"], "the filter shrinks the gap"
    # The same-question comparison is far less powered; two models go nominally negative.
    assert s["n_survive_same_questions"] == 14
    thin = [r["model"] for r in table["models"] if r["thin"]]
    assert thin == ["Opus 4.1"], thin
    assert all(r["n_filtered"] >= MIN_POOL for r in table["models"] if not r["thin"])
