"""The SFT crux sanity check: SFT trains cross-entropy on EXACTLY the model-sampled spans and masks the
prompt + every injected forcing cue — the same span set the GRPO advantage carried.

Uses a mock sampling client (no API) so it exercises the REAL `rollout_coupled` observation-building +
first-line truncation, then runs `lib.sft_train.trajectory_to_sft_datums` and asserts:
  - exactly ONE cross-entropy Datum (the observations chain as exact token prefixes),
  - the loss `weights` are 1.0 on every model-sampled token (search CoT + verify + Subset/Solved/Answer
    values) and 0.0 on the prompt + every injected cue (VERIFY/SUBSET/SOLVED/ANSWER cues),
  - sum(weights) == total sampled tokens,
  - the SFT datum's `weights`/`target_tokens`/`model_input` are byte-identical to the GRPO path's
    `mask`/`target_tokens`/`model_input` (so the masking is literally the same code — a masking bug
    here would be a masking bug in GRPO too),
  - the reconstructed token sequence == base_prompt + search + cue + span + cue + span + ...

A masking bug would train on the injected cues (corrupting the model) or on the wrong span set, so this
mirrors tests/test_rl_masking.py exactly.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from lib.coupling import CONDITION_A, CONDITION_B
from lib.pools import load_coupling_split
from lib.rl_rollout import rollout_coupled
from lib.sft_train import trajectory_to_sft_datums
from lib.tinker_client import get_tokenizer_and_renderer
from tinker_cookbook.rl.data_processing import trajectory_to_data

TOK, _ = get_tokenizer_and_renderer()


class _MockSamplingClient:
    """Returns preset (tokens, stop_reason) per sample_async call, with dummy logprobs."""

    def __init__(self, responses: list[tuple[list[int], str]]):
        self._responses = list(responses)

    async def sample_async(self, *, prompt, num_samples, sampling_params):
        tokens, stop_reason = self._responses.pop(0)
        seq = SimpleNamespace(tokens=list(tokens), logprobs=[-0.5] * len(tokens),
                              stop_reason=stop_reason)
        return SimpleNamespace(sequences=[seq])


def _enc(s: str) -> list[int]:
    return TOK.encode(s, add_special_tokens=False)


def _first_line_len(tokens: list[int]) -> int:
    from lib.rl_rollout import _first_line_token_len
    return _first_line_token_len(tokens, TOK)


def _assert_sft_masking(traj, sampled_spans: list[list[int]], base_ints: list[int]):
    """Assert one CE Datum; the loss WEIGHT SUPPORT is exactly the model-sampled spans (0 on
    prompt/cues); sequence integrity; that the SUPPORT and target tokens match the GRPO masking; and
    that reduction='sum' reproduces the GRPO 0/1 mask while reduction='mean' normalizes per-rollout."""
    # Default reduction='mean' (the recipe default): per-rollout token-mean weights.
    sft = trajectory_to_sft_datums(traj)
    assert len(sft) == 1, f"expected exactly one SFT Datum, got {len(sft)}"
    datum = sft[0]
    assert set(datum.loss_fn_inputs.keys()) == {"weights", "target_tokens"}, (
        "a cross-entropy datum must carry only weights + target_tokens (no advantages/logprobs)")
    weights = datum.loss_fn_inputs["weights"].to_torch().tolist()
    target_tokens = datum.loss_fn_inputs["target_tokens"].to_torch().tolist()
    model_input_tokens = datum.model_input.to_ints()

    # 'sum' reduction must be byte-identical to the GRPO trajectory mask (same trajectory_to_data code).
    pg = trajectory_to_data(traj, 1.0)
    assert len(pg) == 1
    pg_mask = pg[0].loss_fn_inputs["mask"].to_torch().tolist()
    pg_targets = pg[0].loss_fn_inputs["target_tokens"].to_torch().tolist()
    sum_weights = trajectory_to_sft_datums(traj, reduction="sum")[0].loss_fn_inputs["weights"].to_torch().tolist()
    assert sum_weights == pg_mask, "reduction='sum' weights must equal the GRPO trajectory mask"
    assert target_tokens == pg_targets, "SFT target tokens must equal the GRPO datum's targets"
    assert model_input_tokens == pg[0].model_input.to_ints(), "SFT model_input must equal GRPO's"

    # Reconstruct the full token sequence and the ground-truth "is this a sampled action token?".
    full = model_input_tokens + [target_tokens[-1]]
    total_sampled = sum(len(s) for s in sampled_spans)
    expected = list(base_ints)
    for tr, span in zip(traj.transitions, sampled_spans):
        ob = tr.ob.to_ints()
        assert ob[: len(expected)] == expected, "observation is not an exact prefix extension"
        expected = ob + list(span)
    assert full == expected, "reconstructed token sequence mismatch"

    is_action = [False] * len(full)
    for tr, span in zip(traj.transitions, sampled_spans):
        pos = len(tr.ob.to_ints())  # sampled span starts right after this observation
        for j in range(len(span)):
            is_action[pos + j] = True
    # weights are over targets (left-shifted by one): index t applies to predicting full[t+1].
    # SUPPORT (which tokens are trained) is the load-bearing masking property; check it for BOTH
    # reductions, plus the exact 0/1 values for 'sum' and the per-rollout normalization for 'mean'.
    assert sum(sum_weights) == total_sampled, f"sum-mask total {sum(sum_weights)} != sampled {total_sampled}"
    assert abs(sum(weights) - 1.0) < 1e-6, "reduction='mean' weights must sum to 1 over the rollout"
    for t in range(len(weights)):
        on = is_action[t + 1]
        assert (weights[t] > 0) == on, f"weight[{t}] support wrong: {weights[t]} vs is_action {on}"
        assert (sum_weights[t] == 1.0) == on, f"sum-mask[{t}] must be 1 on span, 0 on cue/prompt"
        if on:
            assert abs(weights[t] - 1.0 / total_sampled) < 1e-6, "mean weights must be uniform over the span"


def test_sft_masks_cues_only_condition_b():
    ss, add = load_coupling_split("dev")[0]
    search = _enc("Let me search for a subset. 100+200 = 300, no. Trying more combinations...")
    verify = _enc("Checking each: 100 in list yes, 200 in list yes; sum matches; no repeats.")
    subset = _enc("100, 200, 300\nand some trailing junk that must be dropped")
    solved = _enc("yes\nbecause the checks passed")
    answer = _enc("158\n(nothing after)")
    mock = _MockSamplingClient([
        (search, "length"),  # phase 1 truncated -> forces the flow
        (verify, "stop"), (subset, "stop"), (solved, "stop"), (answer, "stop"),
    ])
    traj, info = asyncio.run(rollout_coupled(mock, ss, add, CONDITION_B, seed=0))
    _, renderer = get_tokenizer_and_renderer()
    base_ints = renderer.build_generation_prompt(
        [{"role": "user", "content": CONDITION_B.prompt_fn(ss, add)}]).to_ints()
    kept_subset = subset[: _first_line_len(subset)]
    kept_solved = solved[: _first_line_len(solved)]
    kept_answer = answer[: _first_line_len(answer)]
    _assert_sft_masking(traj, [search, verify, kept_subset, kept_solved, kept_answer], base_ints)


def test_sft_masks_cues_only_condition_a():
    """Condition A uses the identical scaffold (only the answer cue differs) — masking must be identical."""
    ss, add = load_coupling_split("dev")[0]
    search = _enc("Searching... 300+400+500 = 1200, checking against target.")
    verify = _enc("Each chosen number is in the list; the sum equals the target; no repeats.")
    subset = _enc("300, 400, 500")
    solved = _enc("yes")
    answer = _enc(f"{add.answer}")
    mock = _MockSamplingClient([
        (search, "length"), (verify, "stop"), (subset, "stop"), (solved, "stop"), (answer, "stop"),
    ])
    traj, info = asyncio.run(rollout_coupled(mock, ss, add, CONDITION_A, seed=0))
    _, renderer = get_tokenizer_and_renderer()
    base_ints = renderer.build_generation_prompt(
        [{"role": "user", "content": CONDITION_A.prompt_fn(ss, add)}]).to_ints()
    _assert_sft_masking(traj, [search, verify,
                               subset[: _first_line_len(subset)],
                               solved[: _first_line_len(solved)],
                               answer[: _first_line_len(answer)]], base_ints)


def test_sft_clean_conclusion_single_transition():
    """A rollout that concludes on its own within budget -> one transition -> one CE datum, whole
    sampled completion weighted 1, prompt weighted 0."""
    ss, add = load_coupling_split("dev")[0]
    full = _enc(f"Reasoning done.\nSubset: none\nSolved: no\nAnswer: {add.answer}")
    mock = _MockSamplingClient([(full, "stop")])
    traj, info = asyncio.run(rollout_coupled(mock, ss, add, CONDITION_A, seed=0))
    assert len(traj.transitions) == 1 and info.is_clean and not info.forced
    _, renderer = get_tokenizer_and_renderer()
    base_ints = renderer.build_generation_prompt(
        [{"role": "user", "content": CONDITION_A.prompt_fn(ss, add)}]).to_ints()
    _assert_sft_masking(traj, [full], base_ints)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
