"""The four RL conditions (A/B/C/D): which prompt/flow to roll out and how to reward a GRPO group.

Reward wiring (load-bearing — locked by tests/test_rl_rewards.py):
  - A (exposure baseline) & B (self-steering): reward = addition-correct ONLY. The reward function
    never reads the subset-sum answer. A and B share a byte-identical prompt/flow and differ only in
    the final addition-answer rule (carried by the `lib.coupling.Condition`'s answer cue).
  - C (direct side reward): neutral subset-sum prompt, single-cue budget forcing; reward = subset-correct.
  - D (reward-variance-matched placebo): B's EXACT prompt/flow, but within each GRPO group the
    per-rollout addition-correct rewards are randomly PERMUTED. This preserves the group's reward
    multiset (same within-group variance, same #reinforced, same dead-group structure as B) while
    decorrelating reward from which rollout actually solved — the control that separates "coupling
    routed reward into capability" from "B merely received RL updates and A did not" (see REPORT.md).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from lib.coupling import CONDITION_A, CONDITION_B, SCAFFOLD_ANSWER_CUE, Condition
from lib.prompts import neutral_scaffold_prompt
from lib.rl_rollout import RolloutInfo

# C-scaffold: direct subset-sum reward (like C) but rolled out under B's 5-phase structured forced
# flow (search -> verify -> Subset -> Solved -> Answer) on the NEUTRAL scaffold prompt (no addition,
# no coupling). The clean test of the confounded "B > C": if C-scaffold rises to ~B's level, "B > C"
# was the training scaffold, not indirect-vs-direct reward. Reuses rollout_coupled (kind='coupled');
# the (ss, add) pair is passed but the neutral scaffold prompt ignores `add`, and the reward reads
# subset_correct (not addition_correct), so the addition operand is inert here.
CONDITION_C_SCAFFOLD = Condition("cs", lambda ss, add: neutral_scaffold_prompt(ss), SCAFFOLD_ANSWER_CUE)


@dataclass(frozen=True)
class RLCondition:
    name: str                       # 'a' / 'b' / 'c' / 'd'
    kind: str                       # 'coupled' (A/B/D) or 'neutral' (C)
    reward_attr: str                # RolloutInfo field the reward reads: 'addition_correct' or 'subset_correct'
    coupling: Condition | None      # lib.coupling.Condition for coupled conditions; None for C
    shuffle_reward: bool = False    # True only for D (within-group reward permutation)


RL_CONDITION_A = RLCondition("a", "coupled", "addition_correct", CONDITION_A)
RL_CONDITION_B = RLCondition("b", "coupled", "addition_correct", CONDITION_B)
RL_CONDITION_C = RLCondition("c", "neutral", "subset_correct", None)
# D reuses B's prompt AND B's answer cue verbatim (byte-identical rollouts); only the reward differs.
RL_CONDITION_D = RLCondition("d", "coupled", "addition_correct", CONDITION_B, shuffle_reward=True)
# C-scaffold (secondary ablation): subset_correct reward under B's forced flow, neutral scaffold prompt.
RL_CONDITION_CS = RLCondition("cs", "coupled", "subset_correct", CONDITION_C_SCAFFOLD)

RL_CONDITIONS = {c.name: c for c in (RL_CONDITION_A, RL_CONDITION_B, RL_CONDITION_C, RL_CONDITION_D,
                                     RL_CONDITION_CS)}


def raw_rewards(condition: RLCondition, infos: list[RolloutInfo]) -> list[float]:
    """The un-permuted per-rollout reward (1.0/0.0) from the condition's reward-relevant outcome."""
    return [1.0 if getattr(info, condition.reward_attr) else 0.0 for info in infos]


def group_rewards(
    condition: RLCondition, infos: list[RolloutInfo], *, step: int, group_idx: int, seed: int = 0
) -> list[float]:
    """Final per-rollout reward for one GRPO group. Identity for A/B/C; for D, a uniform within-group
    permutation of the raw rewards, seeded deterministically by (seed, step, group_idx, condition)."""
    raw = raw_rewards(condition, infos)
    if not condition.shuffle_reward:
        return raw
    rng = random.Random(f"{seed}-{step}-{group_idx}-{condition.name}")
    perm = list(range(len(raw)))
    rng.shuffle(perm)
    return [raw[perm[i]] for i in range(len(raw))]
