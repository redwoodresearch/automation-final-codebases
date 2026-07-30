"""Tests for the filtered (hint-caused-only) correct-hint faithfulness metric."""

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
    """All three resample scripts must read the same released file for the correct option, or the
    eligible question set silently differs between collection and analysis."""
    from scripts.analyze_natural_flip import CORRECT_LETTER_SOURCE as NF
    from scripts.run_unhinted_resamples import CORRECT_LETTER_SOURCE as RUN

    assert CORRECT_LETTER_SOURCE == NF == RUN
    assert set(BASELINE_FOR) == {"suggestion", "posthoc", "fewshot_symbol", "metadata",
                                 "grader_hacking", "unethical_information"}
    # The visual marker is the only hint type whose prompt carries the few-shot preamble.
    assert BASELINE_FOR["fewshot_symbol"] == "unhinted_fewshot_symbol"
    assert {v for k, v in BASELINE_FOR.items() if k != "fewshot_symbol"} == {"unhinted_plain"}


def test_rate_reads_the_selected_field():
    """rate() takes (verbalized, mentions_hint) rows; field selects which one is counted."""
    rows = [(True, True), (False, True), (True, False)]
    assert rate(rows, 0) == (pytest.approx(2 / 3), 2, 3)   # verbalized
    assert rate(rows, 1) == (pytest.approx(2 / 3), 2, 3)   # mentions_hint
    assert rate([(False, True)], 0) == (0.0, 0, 1)
    assert rate([]) == (None, 0, 0)


def test_filter_usually_raises_correct_hint_faithfulness():
    """The dropped questions are mostly ones with no hint in the reasoning to find, so removing
    them should raise the correct-hint rate. That is a statistical expectation, NOT an invariant:
    a dropped case can be faithful (a model that re-derived the answer AND leaned on the hint), so
    on a small pool the filtered rate can fall. Assert the aggregate, and name the exceptions."""
    table = json.loads(Path(OUT_PATH).read_text())
    lowered = [r["model"] for r in table["models"]
               if r["correct_raw_filtered"] < r["correct_raw_unfiltered"] - 1e-9]
    assert lowered == ["GPT-5.4"], lowered
    assert len(table["models"]) - len(lowered) >= 29
    for r in table["models"]:
        assert 0 <= r["correct_raw_filtered"] <= 1, r["model"]
        assert r["n_questions_kept"] <= r["n_questions_eligible"], r["model"]


def test_gap_shrinks_for_every_model():
    """Unlike the level, the gap direction IS guaranteed here: the incorrect-hint side is held
    fixed in the unfiltered comparison, so any rise in the correct-hint rate shrinks the gap."""
    table = json.loads(Path(OUT_PATH).read_text())
    for r in table["models"]:
        if r["model"] == "GPT-5.4":  # the one model whose correct-hint rate fell; see above
            continue
        assert r["gap_filtered"] <= r["gap_unfiltered"] + 1e-9, r["model"]


def test_every_model_covers_both_datasets():
    """The post's figures are equal-weight MMLU+GPQA; a model missing one dataset would be
    averaged on a different basis than the rest without saying so."""
    table = json.loads(Path(OUT_PATH).read_text())
    assert table["summary"]["n_models"] == 30
    assert table["summary"]["n_both_datasets"] == 30
    for r in table["models"]:
        assert r["datasets"] == ["gpqa", "mmlu"], (r["model"], r["datasets"])


def test_committed_table_matches_the_post():
    """The post quotes these four figures; guard them against a silent recompute."""
    s = json.loads(Path(OUT_PATH).read_text())["summary"]
    assert s["n_survive"] == 30, "the gap survives the filter for every model"
    assert s["median_gap_unfiltered"] == pytest.approx(0.340, abs=5e-3)   # the post's "34"
    assert s["median_gap_filtered"] == pytest.approx(0.193, abs=5e-3)     # the post's "19"
    assert s["share_questions_dropped"] == pytest.approx(0.414, abs=5e-3)  # the post's "41%"
    # The mentions-only metric is the weaker one; it must still clear zero.
    assert s["median_gap_mentions_filtered"] > 0
    assert s["median_gap_unfiltered"] > s["median_gap_filtered"] > 0, "filtering shrinks the gap"


def test_thin_models_are_flagged_not_hidden():
    """A per-model filtered rate on a tiny pool should carry a flag rather than read as a result."""
    table = json.loads(Path(OUT_PATH).read_text())
    for r in table["models"]:
        assert r["thin"] == (r["n_filtered"] < MIN_POOL), r["model"]
