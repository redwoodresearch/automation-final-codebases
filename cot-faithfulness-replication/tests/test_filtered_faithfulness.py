"""Tests for the filtered (complier-only) correct-hint faithfulness metric."""

import json
import statistics
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
    from scripts.run_unhinted_resamples import CORRECT_LETTER_SOURCE as RUN
    assert CORRECT_LETTER_SOURCE == NF == RUN
    assert set(BASELINE_FOR) == {"suggestion", "posthoc", "fewshot_symbol", "metadata",
                                 "grader_hacking", "unethical_information"}
    # The visual marker is the only hint type carrying the few-shot preamble.
    assert BASELINE_FOR["fewshot_symbol"] == "unhinted_fewshot_symbol"
    assert {v for k, v in BASELINE_FOR.items() if k != "fewshot_symbol"} == {"unhinted_plain"}


def test_rate_helper():
    assert rate([]) == (None, 0, 0)
    assert rate([(True, False), (False, True), (True, True)]) == (pytest.approx(2 / 3), 2, 3)
    assert rate([(True, False), (False, True), (True, True)], field=1) == (pytest.approx(2 / 3), 2, 3)


def test_filter_raises_correct_hint_faithfulness_in_aggregate():
    """Dropping questions the model can solve unhinted mostly removes cases with no hint in the
    reasoning, so the correct-hint rate should rise. It is NOT guaranteed per model: some
    spontaneous solvers do mention and depend on the hint, and where those are over-represented
    in the dropped set a single model can move the other way (GPT-5.4 does, on 22 MMLU cases)."""
    table = json.loads(Path(OUT_PATH).read_text())
    rose = [r for r in table["models"] if r["correct_raw_filtered"] > r["correct_raw_unfiltered"]]
    assert len(rose) >= 27, f"only {len(rose)}/30 models rose"
    assert statistics.median(r["correct_raw_filtered"] - r["correct_raw_unfiltered"]
                             for r in table["models"]) > 0.02
    for r in table["models"]:
        assert 0 <= r["correct_raw_filtered"] <= 1, r["model"]
        lo, hi = r["correct_filtered_ci"]
        assert lo - 1e-9 <= r["correct_raw_filtered"] <= hi + 1e-9, r["model"]


def test_every_model_has_both_datasets():
    """The whole point of collecting the GPQA resamples was to put this on the same
    equal-weight MMLU+GPQA basis as the post's headline numbers."""
    table = json.loads(Path(OUT_PATH).read_text())
    assert table["summary"]["n_models"] == 30
    assert table["summary"]["n_both_datasets"] == 30
    assert all(r["datasets"] == ["gpqa", "mmlu"] for r in table["models"])


def test_committed_table_conclusions():
    table = json.loads(Path(OUT_PATH).read_text())
    s = table["summary"]
    assert s["n_survive"] == 30, "the gap survives the filter for every model"
    # The post quotes these two as "34 to 19"; they must stay on the same basis as the
    # mentions-only paragraph above it (equal-weight MMLU+GPQA, all 30 models).
    assert s["median_gap_unfiltered"] == pytest.approx(0.340, abs=5e-3)
    assert s["median_gap_filtered"] == pytest.approx(0.193, abs=5e-3)
    # Strictest cell: direction holds for most models but not all; the post says so.
    assert s["n_survive_same_questions"] == 26
    assert s["median_gap_same_questions"] == pytest.approx(0.134, abs=5e-3)
    # Mentions-only under the same filter does not collapse.
    assert s["median_gap_mentions_filtered"] > 0.10
