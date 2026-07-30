"""Constructive, always-solvable subset sum: generation, verification, analysis.

Instances are built by first choosing a hidden non-empty subset, so a solution always
exists. The verifier is load-bearing for the RL reward downstream: it accepts an answer
only if the chosen elements are a valid sub-multiset of the list AND sum exactly to the
target. The canonical answer format is the chosen **values** over distinct-integer lists
(`verify_subset_sum_values`); an index verifier (`verify_subset_sum_indices`) is also
provided and tested as the alternative. See the write-up for why values are canonical.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Sequence

import attrs

from lib.config import SubsetSumDifficulty


@attrs.frozen
class SubsetSumInstance:
    values: tuple[int, ...]
    target: int
    # A known solution as 0-based indices (for reference/inspection; not shown to the model).
    hidden_subset_indices_0based: tuple[int, ...]
    difficulty_name: str

    @property
    def n(self) -> int:
        return len(self.values)

    def key(self) -> tuple[tuple[int, ...], int]:
        """Identity of the instance (for dedup / disjointness checks)."""
        return (self.values, self.target)


def _sample_values(rng: random.Random, diff: SubsetSumDifficulty) -> list[int]:
    span = diff.value_hi - diff.value_lo + 1
    if diff.unique_values:
        if span < diff.n:
            raise ValueError(f"Value range [{diff.value_lo},{diff.value_hi}] too small for {diff.n} unique values")
        return rng.sample(range(diff.value_lo, diff.value_hi + 1), diff.n)
    return [rng.randint(diff.value_lo, diff.value_hi) for _ in range(diff.n)]


def generate_subset_sum(rng: random.Random, diff: SubsetSumDifficulty) -> SubsetSumInstance:
    """Generate one always-solvable instance from a difficulty config.

    Rejects instances that admit a solution smaller than diff.min_solution_size_allowed
    (to remove trivially-easy instances). Retries with the same rng (which advances)."""
    max_attempts = 200
    for _ in range(max_attempts):
        values = _sample_values(rng, diff)
        k = rng.randint(diff.k_lo, diff.k_hi)
        subset_indices = sorted(rng.sample(range(diff.n), k))
        target = sum(values[i] for i in subset_indices)
        min_size = min_solution_size(values, target)
        if min_size is not None and min_size >= diff.min_solution_size_allowed:
            return SubsetSumInstance(
                values=tuple(values),
                target=target,
                hidden_subset_indices_0based=tuple(subset_indices),
                difficulty_name=diff.name,
            )
    raise RuntimeError(
        f"Could not generate an instance with min_solution_size >= "
        f"{diff.min_solution_size_allowed} for {diff.name} after {max_attempts} attempts"
    )


# --- Verifiers ---


def verify_subset_sum_indices(
    values: Sequence[int], target: int, chosen_indices_1based: Sequence[int] | None
) -> bool:
    """Alternative (index-format) verifier. Correct iff the 1-based indices are distinct,
    in range, non-empty, and the referenced elements sum exactly to target.

    Rejects: empty/None, out-of-range indices, reused (duplicate) indices, wrong sum.
    Because all values are positive, a strict superset of a valid subset sums to more
    than target and is therefore rejected by the sum check.
    """
    if not chosen_indices_1based:
        return False
    n = len(values)
    seen = set()
    for i in chosen_indices_1based:
        if not isinstance(i, int) or isinstance(i, bool):
            return False
        if i < 1 or i > n:
            return False
        if i in seen:  # a list position cannot be used twice
            return False
        seen.add(i)
    return sum(values[i - 1] for i in chosen_indices_1based) == target


def verify_subset_sum_values(
    values: Sequence[int], target: int, chosen_values: Sequence[int] | None
) -> bool:
    """Canonical verifier — answers are the chosen raw values (multiplicity-checked).

    Correct iff the chosen values are a sub-multiset of the list (each value used no
    more times than it appears) and sum exactly to target. This is the primary scoring
    path (the answer format is chosen values over distinct-integer lists).
    """
    if not chosen_values:
        return False
    available = Counter(values)
    used = Counter()
    for v in chosen_values:
        if not isinstance(v, int) or isinstance(v, bool):
            return False
        used[v] += 1
        if used[v] > available.get(v, 0):
            return False
    return sum(chosen_values) == target


# --- Analysis helpers (for difficulty tuning / data inspection) ---


def count_solutions(values: Sequence[int], target: int) -> int:
    """Number of distinct subsets (by index set) summing exactly to target, via DP.

    Feasible for the n / value ranges used here (n<=~20, target<=~n*value_hi).
    """
    # dp[s] = number of index-subsets summing to s. Includes the empty subset at s=0.
    dp = Counter({0: 1})
    for v in values:
        new_dp = Counter(dp)
        for s, count in dp.items():
            new_dp[s + v] += count
        dp = new_dp
    solutions = dp.get(target, 0)
    # Exclude the empty subset if target == 0 (not used: targets are positive here).
    if target == 0:
        solutions -= 1
    return solutions


def min_solution_size(values: Sequence[int], target: int) -> int | None:
    """Smallest subset size that sums to target, or None if unsolvable.

    Used to flag trivially-easy instances (e.g. a size-1 solution)."""
    # dp[s] = min number of elements to reach sum s.
    INF = float("inf")
    dp = {0: 0}
    for v in values:
        new_dp = dict(dp)
        for s, size in dp.items():
            cand = size + 1
            if new_dp.get(s + v, INF) > cand:
                new_dp[s + v] = cand
        dp = new_dp
    result = dp.get(target)
    if result is None or result == 0:
        return None
    return result
