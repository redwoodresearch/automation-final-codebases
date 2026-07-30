"""The crux sanity check: the in-loop rollout loss-masks injected cues and trains only sampled spans.

Uses a mock sampling client (no API) so it exercises the REAL `rollout_coupled` /
`rollout_budget_forced` observation-building + first-line truncation, then runs the cookbook's
`trajectory_to_data` and asserts:
  - exactly ONE Datum (the observations chain as exact token prefixes — no BPE-boundary break),
  - advantage == 0 on the prompt + every injected cue token,
  - advantage == traj_advantage on every model-sampled token (incl. the phase-1 search CoT),
  - sum(mask) == total sampled tokens,
  - the reconstructed full token sequence == base_prompt + search + cue + span + cue + span + ...
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from lib.coupling import ANSWER_CUE_B, CONDITION_A, CONDITION_B, SOLVED_CUE, SUBSET_CUE, VERIFY_CUE
from lib.pools import load_coupling_split
from lib.rl_rollout import rollout_coupled, rollout_budget_forced
from lib.tinker_client import DEFAULT_FORCE_CUE, get_tokenizer_and_renderer
from tinker_cookbook.rl.data_processing import trajectory_to_data

TOK, _ = get_tokenizer_and_renderer()


class _MockSamplingClient:
    """Returns preset (tokens, stop_reason) per sample_async call, with dummy logprobs."""

    def __init__(self, responses: list[tuple[list[int], str]]):
        self._responses = list(responses)
        self.prompts_seen: list[list[int]] = []

    async def sample_async(self, *, prompt, num_samples, sampling_params):
        self.prompts_seen.append(prompt.to_ints())
        tokens, stop_reason = self._responses.pop(0)
        seq = SimpleNamespace(tokens=list(tokens), logprobs=[-0.5] * len(tokens),
                              stop_reason=stop_reason)
        return SimpleNamespace(sequences=[seq])


def _enc(s: str) -> list[int]:
    return TOK.encode(s, add_special_tokens=False)


def _assert_single_datum_masking(traj, sampled_spans: list[list[int]], base_ints: list[int]):
    """Assert one Datum, advantage 0 on obs/cues and ADV on sampled spans, and sequence integrity."""
    ADV = 3.0
    data = trajectory_to_data(traj, ADV)
    assert len(data) == 1, f"expected exactly one Datum, got {len(data)}"
    datum = data[0]
    advantages = datum.loss_fn_inputs["advantages"].to_torch().tolist()
    mask = datum.loss_fn_inputs["mask"].to_torch().tolist()
    target_tokens = datum.loss_fn_inputs["target_tokens"].to_torch().tolist()
    model_input_tokens = datum.model_input.to_ints()

    # The full token sequence trajectory_to_data reasoned over = model_input + last target token.
    full = model_input_tokens + [target_tokens[-1]]
    total_sampled = sum(len(s) for s in sampled_spans)
    # Reconstruct expected: base prompt then, per transition, the delta-observation then the span.
    expected = list(base_ints)
    for tr, span in zip(traj.transitions, sampled_spans):
        ob = tr.ob.to_ints()
        assert ob[: len(expected)] == expected, "observation is not an exact prefix extension"
        expected = ob + list(span)  # delta-ob (cue) then the sampled span
    assert full == expected, "reconstructed token sequence mismatch"

    # advantages/mask are over targets (left-shifted by one): index t corresponds to full[t+1].
    assert sum(mask) == total_sampled, f"sum(mask)={sum(mask)} != sampled tokens {total_sampled}"
    # Build the ground-truth "is this token a sampled action token?" over full positions.
    is_action = [False] * len(full)
    pos = len(base_ints)
    for tr, span in zip(traj.transitions, sampled_spans):
        ob = tr.ob.to_ints()
        pos = len(ob)  # sampled span starts right after this observation
        for j in range(len(span)):
            is_action[pos + j] = True
        pos += len(span)
    # targets are full[1:]; advantages[t] applies to predicting full[t+1].
    for t in range(len(advantages)):
        want = ADV if is_action[t + 1] else 0.0
        assert advantages[t] == want, f"advantage[{t}]={advantages[t]} want {want}"
        assert mask[t] == (1.0 if is_action[t + 1] else 0.0)


def test_coupled_rollout_masks_cues_only():
    ss, add = load_coupling_split("dev")[0]
    search = _enc("Let me search for a subset. 100+200 = 300, no. Trying more combinations...")
    verify = _enc("Checking each: 100 in list yes, 200 in list yes; sum matches; no repeats.")
    subset = _enc("100, 200, 300\nand some trailing junk that must be dropped")
    solved = _enc("yes\nbecause the checks passed")
    answer = _enc("158\n(nothing after)")
    mock = _MockSamplingClient([
        (search, "length"),   # phase 1 truncated -> forces the flow
        (verify, "stop"),
        (subset, "stop"),
        (solved, "stop"),
        (answer, "stop"),
    ])
    traj, info = asyncio.run(rollout_coupled(mock, ss, add, CONDITION_B, seed=0))
    _, renderer = get_tokenizer_and_renderer()
    base_ints = renderer.build_generation_prompt(
        [{"role": "user", "content": CONDITION_B.prompt_fn(ss, add)}]).to_ints()
    # First-line truncation: sampled spans are the KEPT tokens only.
    kept_subset = subset[: _first_line_len(subset)]
    kept_solved = solved[: _first_line_len(solved)]
    kept_answer = answer[: _first_line_len(answer)]
    _assert_single_datum_masking(traj, [search, verify, kept_subset, kept_solved, kept_answer], base_ints)
    assert info.forced and not info.is_clean
    assert info.subset_values == [100, 200, 300]
    assert info.solved is True


def test_coupled_clean_conclusion_single_transition():
    ss, add = load_coupling_split("dev")[0]
    # Model concludes on its own within budget: stop_reason 'stop', full answer format present.
    full = _enc(f"Reasoning done.\nSubset: none\nSolved: no\nAnswer: {add.answer}")
    mock = _MockSamplingClient([(full, "stop")])
    traj, info = asyncio.run(rollout_coupled(mock, ss, add, CONDITION_A, seed=0))
    assert len(traj.transitions) == 1
    assert info.is_clean and not info.forced
    _, renderer = get_tokenizer_and_renderer()
    base_ints = renderer.build_generation_prompt(
        [{"role": "user", "content": CONDITION_A.prompt_fn(ss, add)}]).to_ints()
    _assert_single_datum_masking(traj, [full], base_ints)


def test_budget_forced_rollout_masks_cue():
    ss, _add = load_coupling_split("dev")[0]
    from lib.prompts import subset_sum_prompt
    search = _enc("Searching for the subset... let me try some combinations.")
    ans = _enc(f"{', '.join(str(v) for v in ss.values[:3])}")
    mock = _MockSamplingClient([(search, "length"), (ans, "stop")])
    traj, info = asyncio.run(rollout_budget_forced(mock, ss, seed=0))
    _, renderer = get_tokenizer_and_renderer()
    base_ints = renderer.build_generation_prompt(
        [{"role": "user", "content": subset_sum_prompt(ss)}]).to_ints()
    _assert_single_datum_masking(traj, [search, ans], base_ints)
    assert info.forced and not info.is_clean


def _first_line_len(tokens: list[int]) -> int:
    from lib.rl_rollout import _first_line_token_len
    return _first_line_token_len(tokens, TOK)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
