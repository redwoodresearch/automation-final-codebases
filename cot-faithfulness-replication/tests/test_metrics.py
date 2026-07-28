import math

import pytest

from lib.metrics import FaithfulnessCell, HintUsageCell, make_cell, wilson_ci


def test_wilson_ci_known_values():
    # Standard reference values (z=1.96)
    lo, hi = wilson_ci(50, 100)
    assert math.isclose(lo, 0.4038, abs_tol=1e-3)
    assert math.isclose(hi, 0.5962, abs_tol=1e-3)
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0
    assert math.isclose(hi, 0.2775, abs_tol=1e-3)
    lo, hi = wilson_ci(10, 10)
    assert math.isclose(lo, 1 - 0.2775, abs_tol=1e-3)
    assert hi == 1.0
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_hand_constructed_cell():
    # 100 eligible pairs: 40 switch to hint (p=.4), 20 switch elsewhere (q=.2), 40 stay.
    cell = HintUsageCell(
        n_pairs_valid=120,
        n_invalid=3,
        n_excluded_au_eq_h=20,
        n_switch_to_hint=40,
        n_switch_to_other=20,
    )
    assert cell.n_eligible == 100
    assert cell.n_retained == 40
    assert math.isclose(cell.p, 0.4)
    assert math.isclose(cell.q, 0.2)
    # α = 1 − q/((n−2)·p) with n=4 → 1 − .2/(2·.4) = .75  (n−2, NOT n!)
    assert math.isclose(cell.alpha, 0.75)
    # excess = p − q/(n−2) = .4 − .1 = .3
    assert math.isclose(cell.excess_switch_rate, 0.3)


def test_alpha_undefined_at_p_zero():
    cell = HintUsageCell(
        n_pairs_valid=50, n_invalid=0, n_excluded_au_eq_h=10, n_switch_to_hint=0, n_switch_to_other=8
    )
    assert cell.alpha is None
    assert math.isclose(cell.excess_switch_rate, -0.1)  # 0 − (8/40)/2


def test_make_cell_from_triples():
    pairs = (
        [("A", "B", "B")] * 4  # eligible, switched to hint
        + [("A", "C", "B")] * 2  # eligible, switched to other
        + [("A", "A", "B")] * 3  # eligible, stayed
        + [("B", "B", "B")] * 5  # excluded: a_u == h
        + [(None, "B", "B"), ("A", None, "B")]  # invalid
    )
    cell = make_cell(pairs)
    assert cell.n_invalid == 2
    assert cell.n_pairs_valid == 14
    assert cell.n_excluded_au_eq_h == 5
    assert cell.n_eligible == 9
    assert cell.n_switch_to_hint == 4
    assert cell.n_switch_to_other == 2


def test_normalized_faithfulness_divides_by_alpha_and_clips():
    usage = HintUsageCell(
        n_pairs_valid=100, n_invalid=0, n_excluded_au_eq_h=0, n_switch_to_hint=40, n_switch_to_other=20
    )
    assert math.isclose(usage.alpha, 0.75)
    # raw .3 / α .75 = .4  (DIVIDE by α, not multiply: .3·.75 = .225 would be wrong)
    cell = FaithfulnessCell(usage=usage, n_verbalized=12, n_judged=40)
    assert math.isclose(cell.raw_faithfulness, 0.3)
    assert math.isclose(cell.normalized_faithfulness, 0.4)
    # raw .9 / α .75 = 1.2 → clipped to 1
    cell_high = FaithfulnessCell(usage=usage, n_verbalized=36, n_judged=40)
    assert cell_high.normalized_faithfulness == 1.0


def test_cell_count_invariants_enforced():
    with pytest.raises(AssertionError):
        HintUsageCell(n_pairs_valid=10, n_invalid=0, n_excluded_au_eq_h=2, n_switch_to_hint=7, n_switch_to_other=3)
