"""Central configuration: model, renderer, sampling params, task difficulty, seeds.

These values are the single source of truth for the rest of the pipeline. The subset-sum difficulty
was tuned by sampling and is locked here (see REPORT.md for the rationale).
"""

from __future__ import annotations

import attrs

# --- Model / renderer (confirmed served on Tinker) ---
MODEL = "Qwen/Qwen3.5-9B"
RENDERER_NAME = "qwen3_5"  # reasoning renderer: emits <think>...</think>, stop = <|im_end|>

# --- Sampling params used for RL rollouts (and the benchmark headline numbers) ---
# temperature/top_p are the tinker-cookbook defaults; MAX_TOKENS is high so the
# reasoning model doesn't overflow its thinking block (low limits inflate parse errors).
ROLLOUT_TEMPERATURE = 1.0
ROLLOUT_TOP_P = 1.0
MAX_TOKENS = 16384

# Greedy/low-temp reference for a capability read (not the headline number).
GREEDY_TEMPERATURE = 0.0


@attrs.frozen
class SubsetSumDifficulty:
    """Difficulty knobs for constructive subset-sum generation.

    n: list length. value_lo/value_hi: inclusive element value range.
    k_lo/k_hi: inclusive range for the hidden subset size (uniform).
    unique_values: sample the list without replacement (distinct integers). This makes the
        VALUES answer format unambiguous (multiset == set) and removes duplicate edge cases.
    min_solution_size_allowed: reject generated instances that admit a solution smaller than
        this (e.g. a single element equal to target) so instances aren't trivially easy.
    """

    name: str
    n: int
    value_lo: int
    value_hi: int
    k_lo: int
    k_hi: int
    unique_values: bool = True
    min_solution_size_allowed: int = 3


# The locked-in subset-sum difficulty (chosen in this phase — see write-up).
# n=12 distinct 3-digit values, hidden subset size 5-7. With budget forcing at 8000 tokens and
# temp 1.0 the base model lands at ~37.7% (400-problem eval x 8 samples), ~51% of problems with
# per-problem pass rate in [0.2,0.8], within-group reward variance ~0.13, parse-error ~0.1%.
# n=12 was chosen over larger n because it gave the best per-problem spread; the model empirically
# solves by heuristic search (sort + greedy/backtracking) and fails ~62% of the time, so brute-force
# enumeration is not the mechanism (2^12 subsets are not enumerable within an 8000-token budget).
LOCKED_SUBSET_SUM_DIFFICULTY = SubsetSumDifficulty(
    name="n12_v3digit_k5to7",
    n=12,
    value_lo=100,
    value_hi=999,
    k_lo=5,
    k_hi=7,
    unique_values=True,
    min_solution_size_allowed=3,
)

# Budget forcing: cap subset-sum chain-of-thought at this many tokens, then force a committed
# answer (see lib.tinker_client.sample_budget_forced). This keeps the base model from grinding a
# checkable search to the token limit (it otherwise never concludes-wrong), guarantees an answer
# is emitted, and leaves room within MAX_TOKENS for the addition answer in conditions A/B.
# Locked together with the difficulty and rollout temperature.
SUBSET_SUM_THINK_BUDGET = 8000


# Token budget for the coupled prompt's VERIFY phase (the mechanical per-number membership + sum +
# duplicate check in lib.tinker_client.sample_coupled_forced). Kept modest on purpose: a large
# post-forcing headroom lets the base model resume SEARCHING in the answer channel (defeating the
# budget cap). The Subset commit (~64 tokens) and the terminal Solved:/Answer: lines (~64 tokens)
# are forced separately at their labels, so this only needs to fit the verification prose.
COUPLED_ANSWER_BUDGET = 512

# --- Pool sizes and seeds (disjoint train/eval, fixed seeds, saved to disk) ---
SUBSET_SUM_TRAIN_SEED = 20260701
SUBSET_SUM_EVAL_SEED = 20260702
ADDITION_TRAIN_SEED = 20260711
ADDITION_EVAL_SEED = 20260712

SUBSET_SUM_TRAIN_SIZE = 2000
SUBSET_SUM_EVAL_SIZE = 400
ADDITION_TRAIN_SIZE = 2000
ADDITION_EVAL_SIZE = 400

# --- Faithfulness-gate splits ---
# Carved from the TRAIN pools so the 400 EVAL pool stays reserved for the cross-condition clean
# side-task eval. Each subset-sum instance is paired index-aligned with an addition instance (fixed,
# recorded pairing). Dev = prompt iteration; held-out = the final unbiased faithfulness report.
# The two ranges are disjoint by index (no leakage / no prompt-overfitting onto the reported number).
FAITHFULNESS_DEV_START = 0
FAITHFULNESS_DEV_SIZE = 100
FAITHFULNESS_HELDOUT_START = 100
FAITHFULNESS_HELDOUT_SIZE = 300
