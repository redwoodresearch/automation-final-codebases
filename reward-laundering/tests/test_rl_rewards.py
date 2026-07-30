"""Per-condition reward wiring (lib/rl_conditions.py) — locks the load-bearing reward semantics.

Guards: A/B reward reads ONLY addition-correctness (never the subset); C reads subset-correctness; D
is a within-group permutation of B's rewards that preserves the group multiset (variance) while
decorrelating reward from success. Also that A/B never reward a wrong-subset+wrong-answer rollout and
C rewards only valid subsets.
"""

from __future__ import annotations

from collections import Counter

from lib.rl_conditions import (
    RL_CONDITION_A, RL_CONDITION_B, RL_CONDITION_C, RL_CONDITION_CS, RL_CONDITION_D,
    group_rewards, raw_rewards,
)
from lib.rl_rollout import RolloutInfo


def _info(*, subset_correct: bool, addition_correct: bool) -> RolloutInfo:
    return RolloutInfo(
        subset_values=[1] if subset_correct else None, solved=None,
        addition_answer=42 if addition_correct else 0, subset_correct=subset_correct,
        addition_correct=addition_correct, addition_parse_error=False, subset_category="valid",
        forced=True, is_clean=False, n_sample_tokens=10, n_prompt_tokens=5, answer_text="", thinking_text="",
    )


# All four (subset, addition) combinations.
_CC = _info(subset_correct=True, addition_correct=True)
_CW = _info(subset_correct=True, addition_correct=False)
_WC = _info(subset_correct=False, addition_correct=True)   # leakage: correct answer, wrong subset
_WW = _info(subset_correct=False, addition_correct=False)
GROUP = [_CC, _CW, _WC, _WW]


def test_ab_reward_is_addition_only():
    # A and B reward addition-correctness regardless of the subset. _CC and _WC both -> 1 (leakage
    # rollout _WC is still rewarded 1 — that's the reward channel, and precisely why coupling matters).
    for cond in (RL_CONDITION_A, RL_CONDITION_B):
        assert raw_rewards(cond, GROUP) == [1.0, 0.0, 1.0, 0.0]


def test_ab_never_reads_subset():
    # Two infos with identical addition-correctness but opposite subset-correctness must reward equally.
    a = _info(subset_correct=True, addition_correct=True)
    b = _info(subset_correct=False, addition_correct=True)
    assert raw_rewards(RL_CONDITION_B, [a, b]) == [1.0, 1.0]
    a2 = _info(subset_correct=True, addition_correct=False)
    b2 = _info(subset_correct=False, addition_correct=False)
    assert raw_rewards(RL_CONDITION_B, [a2, b2]) == [0.0, 0.0]


def test_c_reward_is_subset_only():
    # C rewards subset-correctness: _CC and _CW (both subset-correct) -> 1; _WC, _WW -> 0.
    assert raw_rewards(RL_CONDITION_C, GROUP) == [1.0, 1.0, 0.0, 0.0]


def test_c_scaffold_reward_is_subset_only_under_coupled_flow():
    # C-scaffold: subset_correct reward (like C) but kind='coupled' (B's forced flow, neutral prompt).
    assert RL_CONDITION_CS.kind == "coupled" and RL_CONDITION_CS.reward_attr == "subset_correct"
    assert raw_rewards(RL_CONDITION_CS, GROUP) == [1.0, 1.0, 0.0, 0.0]  # same reward semantics as C
    # Its coupling prompt is the neutral scaffold (no addition), and its answer cue restates the subset.
    assert not RL_CONDITION_CS.shuffle_reward
    assert "restate" in RL_CONDITION_CS.coupling.answer_cue


def test_ab_zero_on_wrong_wrong():
    assert raw_rewards(RL_CONDITION_B, [_WW]) == [0.0]
    assert raw_rewards(RL_CONDITION_A, [_WW]) == [0.0]


def test_d_preserves_group_multiset_and_variance():
    raw = raw_rewards(RL_CONDITION_D, GROUP)  # == B's raw rewards
    shuffled = group_rewards(RL_CONDITION_D, GROUP, step=0, group_idx=0)
    assert Counter(shuffled) == Counter(raw), "D must preserve the group's reward multiset (variance)"
    assert sum(shuffled) == sum(raw)


def test_d_is_deterministic_given_seed():
    a = group_rewards(RL_CONDITION_D, GROUP, step=3, group_idx=7, seed=11)
    b = group_rewards(RL_CONDITION_D, GROUP, step=3, group_idx=7, seed=11)
    assert a == b
    # Different (step, group) generally gives a different permutation.
    perms = {tuple(group_rewards(RL_CONDITION_D, GROUP, step=s, group_idx=g))
             for s in range(6) for g in range(6)}
    assert len(perms) > 1, "D permutation should vary across groups/steps"


def test_abc_are_identity_not_shuffled():
    for cond in (RL_CONDITION_A, RL_CONDITION_B, RL_CONDITION_C):
        assert group_rewards(cond, GROUP, step=2, group_idx=5) == raw_rewards(cond, GROUP)


def test_d_decorrelates_from_success_on_average():
    # Over many groups/seeds, the reward D assigns to the leakage rollout (_WC, index 2) is
    # uncorrelated with its own outcome — its mean assigned reward ~= the group mean (0.5), not 1.0.
    idx = 2  # _WC has raw reward 1.0
    assigned = [group_rewards(RL_CONDITION_D, GROUP, step=s, group_idx=g)[idx]
                for s in range(50) for g in range(50)]
    mean_assigned = sum(assigned) / len(assigned)
    assert 0.35 < mean_assigned < 0.65, f"D reward on a fixed rollout should ~= group mean 0.5, got {mean_assigned}"


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-v"]))
