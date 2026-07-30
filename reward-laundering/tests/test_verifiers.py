"""Unit tests for the load-bearing verifiers, generators, and parsing.

Run: .venv/bin/python -m pytest tests/ -q
"""

import random

import pytest

from lib.addition import generate_addition, verify_addition
from lib.config import SubsetSumDifficulty
from lib.parsing import parse_addition_answer, parse_subset_indices_answer, parse_subset_values_answer
from lib.subset_sum import (
    count_solutions,
    generate_subset_sum,
    min_solution_size,
    verify_subset_sum_indices,
    verify_subset_sum_values,
)


# --- Subset-sum verifier (indices) ---

def test_valid_subset_indices():
    values = [10, 20, 30, 40]
    assert verify_subset_sum_indices(values, 60, [2, 4])  # 20 + 40
    assert verify_subset_sum_indices(values, 100, [1, 2, 3, 4])
    assert verify_subset_sum_indices(values, 10, [1])


def test_wrong_sum_rejected():
    values = [10, 20, 30, 40]
    assert not verify_subset_sum_indices(values, 61, [2, 4])
    assert not verify_subset_sum_indices(values, 59, [2, 4])


def test_reused_index_rejected():
    # Using index 1 twice must be rejected even though 10+10 == 20.
    values = [10, 20, 30]
    assert not verify_subset_sum_indices(values, 20, [1, 1])


def test_out_of_range_index_rejected():
    values = [10, 20, 30]
    assert not verify_subset_sum_indices(values, 30, [4])  # index 4 doesn't exist
    assert not verify_subset_sum_indices(values, 30, [0])  # 0 is not 1-based
    assert not verify_subset_sum_indices(values, 30, [-1])


def test_empty_and_none_rejected():
    values = [10, 20, 30]
    assert not verify_subset_sum_indices(values, 0, [])
    assert not verify_subset_sum_indices(values, 0, None)


def test_superset_summing_over_target_rejected():
    # {1,2} sums to 30 (target). The superset {1,2,3} sums to 60 != 30 -> reject.
    values = [10, 20, 30]
    assert verify_subset_sum_indices(values, 30, [1, 2])
    assert not verify_subset_sum_indices(values, 30, [1, 2, 3])


def test_duplicate_values_indices_are_distinct_positions():
    # Two 5s at positions 1 and 2; target 10 needs both positions.
    values = [5, 5, 3]
    assert verify_subset_sum_indices(values, 10, [1, 2])  # 5 + 5
    assert not verify_subset_sum_indices(values, 10, [1, 1])  # reused position
    assert verify_subset_sum_indices(values, 8, [1, 3])  # 5 + 3


def test_bool_indices_rejected():
    # True == 1 in Python; make sure booleans don't sneak through as valid indices.
    values = [10, 20]
    assert not verify_subset_sum_indices(values, 10, [True])


# --- Subset-sum verifier (values, multiplicity-checked) ---

def test_values_verifier_multiplicity():
    values = [5, 5, 3]
    assert verify_subset_sum_values(values, 10, [5, 5])  # two 5s available
    assert not verify_subset_sum_values(values, 15, [5, 5, 5])  # only two 5s
    assert verify_subset_sum_values(values, 8, [5, 3])
    assert not verify_subset_sum_values(values, 8, [])
    assert not verify_subset_sum_values(values, 8, None)


def test_values_verifier_unavailable_value_rejected():
    values = [5, 5, 3]
    assert not verify_subset_sum_values(values, 7, [7])  # 7 not in list
    assert not verify_subset_sum_values(values, 6, [3, 3])  # only one 3


# --- Generation: always solvable + analysis helpers ---

def test_generation_always_solvable():
    diff = SubsetSumDifficulty("test", n=14, value_lo=100, value_hi=999, k_lo=5, k_hi=9)
    rng = random.Random(0)
    for _ in range(300):
        inst = generate_subset_sum(rng, diff)
        # The hidden subset is a valid solution, checked both by index and by value.
        hidden_1based = [i + 1 for i in inst.hidden_subset_indices_0based]
        hidden_values = [inst.values[i] for i in inst.hidden_subset_indices_0based]
        assert verify_subset_sum_indices(inst.values, inst.target, hidden_1based)
        assert verify_subset_sum_values(inst.values, inst.target, hidden_values)
        # An exact count confirms at least one solution exists.
        assert count_solutions(inst.values, inst.target) >= 1
        # Values are unique (default) and no trivial (<3) solution exists.
        assert len(set(inst.values)) == len(inst.values)
        assert min_solution_size(inst.values, inst.target) >= diff.min_solution_size_allowed
        # Subset size within configured range.
        assert diff.k_lo <= len(inst.hidden_subset_indices_0based) <= diff.k_hi


def test_count_solutions_basic():
    # subsets of [1,2,3] summing to 3: {3}, {1,2} -> 2
    assert count_solutions([1, 2, 3], 3) == 2
    assert count_solutions([1, 2, 3], 6) == 1  # {1,2,3}
    assert count_solutions([1, 2, 3], 7) == 0


def test_min_solution_size():
    assert min_solution_size([10, 20, 30], 30) == 1  # {30}
    assert min_solution_size([10, 20, 40], 30) == 2  # {10,20}
    assert min_solution_size([10, 20, 40], 5) is None


def test_generation_deterministic_given_seed():
    diff = SubsetSumDifficulty("test", n=10, value_lo=10, value_hi=99, k_lo=3, k_hi=6)
    a = [generate_subset_sum(random.Random(42), diff) for _ in range(1)]
    b = [generate_subset_sum(random.Random(42), diff) for _ in range(1)]
    assert a[0] == b[0]


# --- Addition ---

def test_addition_verifier():
    rng = random.Random(1)
    inst = generate_addition(rng)
    assert 10 <= inst.a <= 99 and 10 <= inst.b <= 99
    assert verify_addition(inst, inst.a + inst.b)
    assert not verify_addition(inst, inst.a + inst.b + 1)
    assert not verify_addition(inst, None)


# --- Parsing ---

def test_parse_addition_answer():
    assert parse_addition_answer("Answer: 85") == 85
    assert parse_addition_answer("blah\nAnswer: 137\n") == 137
    assert parse_addition_answer("Answer: 47 + 38 = 85") == 85
    assert parse_addition_answer("**Answer:** 85") == 85
    assert parse_addition_answer("The answer is 85") == 85  # fallback (marker + last int)
    assert parse_addition_answer("no digits here") is None


def test_parse_subset_values_answer():
    assert parse_subset_values_answer("Answer: 137, 486, 502") == [137, 486, 502]
    assert parse_subset_values_answer("Answer: [137, 486, 502]") == [137, 486, 502]
    assert parse_subset_values_answer("stuff\nAnswer: 137 486 502\n") == [137, 486, 502]
    assert parse_subset_values_answer("Answer: 137, 486 and 502") == [137, 486, 502]
    # Trailing commentary on the answer line must not inject spurious numbers (budget-forcing case).
    assert parse_subset_values_answer("Answer: 918, 846, 679 - wait, 418 is not in list") == [918, 846, 679]
    assert parse_subset_values_answer("no marker here") is None


def test_parse_subset_values_uses_committed_first_answer():
    # Budget-forcing cue: "...best answer:\nAnswer: <nums>" then the model second-guesses.
    # The committed answer is the first 'Answer:' line with numbers; later chatter is ignored.
    forced = ("...here is my best answer:\nAnswer: 816, 786, 700\n"
              "Wait, let me recheck. The answer might be 100, 200 instead.")
    assert parse_subset_values_answer(forced) == [816, 786, 700]
    # The "best answer:" marker's own line starts with "Answer:" (letters) -> skipped correctly.


def test_parse_subset_indices_answer():
    assert parse_subset_indices_answer("Answer: 1, 4, 5") == [1, 4, 5]
    assert parse_subset_indices_answer("Answer: [1, 4, 5]") == [1, 4, 5]
    assert parse_subset_indices_answer("stuff\nAnswer: 2 7 9\n") == [2, 7, 9]
    assert parse_subset_indices_answer("Answer: indices 1, 4, and 5") == [1, 4, 5]
    assert parse_subset_indices_answer("no marker here") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
