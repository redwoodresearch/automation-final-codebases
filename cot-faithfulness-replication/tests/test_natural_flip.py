"""Tests for the natural flip-to-correct baseline (the post's correct-hint footnote).

The rate itself needs the resample transcripts, which are not committed. What is checked here is
the part that can go wrong silently: the eligibility rule and the arithmetic, on synthetic rows,
plus agreement between the collection script and the analyzer about which questions are eligible.
"""

import json
from pathlib import Path

import pytest

from scripts.analyze_natural_flip import CORRECT_LETTER_SOURCE, N_OPTIONS, OUT_PATH, flip_rates
from scripts.run_unhinted_resamples import CORRECT_LETTER_SOURCE as COLLECT_SOURCE


def test_collection_and_analysis_agree_on_the_baselines():
    """Both scripts must read the same released file for the correct option, or the eligible
    question set silently differs between what was collected and what is analyzed."""
    assert COLLECT_SOURCE == CORRECT_LETTER_SOURCE


def _write(tmp_path, resamples, baseline):
    rpath, bpath = tmp_path / "res.jsonl", tmp_path / "base.jsonl"
    for path, rows in ((rpath, resamples), (bpath, baseline)):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(rpath), str(bpath)


def _row(condition, index, sample_idx, answer):
    return {"input": {"condition": condition, "question_index": index, "sample_idx": sample_idx},
            "output": {"answer": answer}}


def _fake_correct(monkeypatch, correct):
    monkeypatch.setattr("scripts.analyze_natural_flip.load_file",
                        lambda key: [type("R", (), {"hint": correct[i]})() for i in sorted(correct)])


@pytest.mark.parametrize("condition", list(CORRECT_LETTER_SOURCE))
def test_flip_rate_arithmetic(tmp_path, monkeypatch, condition):
    # Two questions whose correct answer is (A). The model answered (B) unhinted on both, so both
    # are correct-hint-eligible. Q0 resamples land on correct 1 of 2 times; Q1, 0 of 2.
    # Both baselines are populated (the analyzer reports each separately and requires both).
    _fake_correct(monkeypatch, {0: "A", 1: "A", 2: "A"})
    baseline, resamples = [], []
    for c in CORRECT_LETTER_SOURCE:
        baseline += [_row(c, 0, 0, "B"), _row(c, 1, 0, "B"),
                     _row(c, 2, 0, "A")]  # q2 answered correctly unhinted -> NOT eligible
        resamples += [_row(c, 0, 3, "A"), _row(c, 0, 4, "B"),
                      _row(c, 1, 3, "C"), _row(c, 1, 4, "B"),
                      _row(c, 2, 3, "A"), _row(c, 2, 4, "A")]
    rpath, bpath = _write(tmp_path, resamples, baseline)

    d = flip_rates(rpath, bpath)[condition]
    assert d["n_eligible_questions"] == 2, "the question answered correctly unhinted must be excluded"
    assert d["n_with_resamples"] == 2
    assert d["flip_to_correct"] == pytest.approx(0.25)   # mean over questions of (0.5, 0.0)
    assert d["flip_to_correct_pooled"] == pytest.approx(0.25)  # 1 of 4 resamples
    # q0: no flip to a wrong option other than its own (B); q1: one (C), over n-2 options, 2 samples.
    assert d["flip_to_one_specific_wrong"] == pytest.approx((0 + 1 / (N_OPTIONS - 2) / 2) / 2)


def test_pooled_and_mean_differ_only_by_weighting(tmp_path, monkeypatch):
    """Unequal resample counts per question: the mean-over-questions rate is the published one,
    so it must NOT collapse to the pooled rate."""
    _fake_correct(monkeypatch, {0: "A", 1: "A"})
    baseline, resamples = [], []
    for c in CORRECT_LETTER_SOURCE:
        baseline += [_row(c, 0, 0, "B"), _row(c, 1, 0, "B")]
        resamples += [_row(c, 0, 3, "A")]                        # q0: 1/1 correct
        resamples += [_row(c, 1, i, "B") for i in range(3, 7)]    # q1: 0/4 correct
    rpath, bpath = _write(tmp_path, resamples, baseline)

    d = flip_rates(rpath, bpath)["unhinted_plain"]
    assert d["flip_to_correct"] == pytest.approx(0.5)          # mean over questions
    assert d["flip_to_correct_pooled"] == pytest.approx(0.2)   # 1 of 5 resamples


def test_committed_table_matches_the_documented_numbers():
    """results/natural_flip.json is committed; the post and results/true_arm_natural_flip.md
    quote it, so guard the quoted figures against a silent recompute."""
    table = json.loads(Path(OUT_PATH).read_text())
    assert table["Sonnet 4.5"]["unhinted_plain"]["flip_to_correct"] == pytest.approx(0.135, abs=5e-4)
    assert table["Sonnet 4.5"]["unhinted_fewshot_symbol"]["flip_to_correct"] == pytest.approx(0.167, abs=5e-4)
    assert table["Opus 4.1"]["unhinted_plain"]["flip_to_correct"] == pytest.approx(0.340, abs=5e-4)
    assert table["Opus 4.1"]["unhinted_fewshot_symbol"]["flip_to_correct"] == pytest.approx(0.175, abs=5e-4)
    # The asymmetry the correction rests on: flips to one specific wrong option stay near the floor.
    for model in ("Sonnet 4.5", "Opus 4.1"):
        for condition in CORRECT_LETTER_SOURCE:
            assert table[model][condition]["flip_to_one_specific_wrong"] < 0.03
