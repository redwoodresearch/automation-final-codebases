"""The UNFILTERED baseline mode (lib/sft_train.py filter="none") — locks the two load-bearing
properties:

  1. filter="none" keeps EVERY rollout, including the addition-WRONG (deliberately-wrong-answer)
     ones — so keep_fraction == 1 by construction and there is no reward/selection at all. This is
     what makes it the "no selection" null control.
  2. the SFT loss masking is UNCHANGED: cross-entropy weight support is exactly the model-sampled
     spans (search CoT + verify + committed Subset/Solved/Answer values), ZERO on the prompt + every
     injected forcing cue. The keep/drop step only decides WHICH rollouts become datums; it never
     touches how a datum is built, so the masking is identical regardless of the mode.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from lib.coupling import CONDITION_B
from lib.pools import load_coupling_split
from lib.rl_conditions import RL_CONDITION_A, RL_CONDITION_B
from lib.rl_rollout import RolloutInfo, rollout_coupled
from lib.sft_train import FILTER_MODES, keep_rollout, should_keep, trajectory_to_sft_datums
from lib.tinker_client import get_tokenizer_and_renderer

TOK, _REN = get_tokenizer_and_renderer()


def _info(*, subset_correct: bool, addition_correct: bool) -> RolloutInfo:
    return RolloutInfo(
        subset_values=[1] if subset_correct else None, solved=None,
        addition_answer=42 if addition_correct else 0, subset_correct=subset_correct,
        addition_correct=addition_correct, addition_parse_error=False, subset_category="valid",
        forced=True, is_clean=False, n_sample_tokens=10, n_prompt_tokens=5, answer_text="", thinking_text="",
    )


_CC = _info(subset_correct=True, addition_correct=True)
_CW = _info(subset_correct=True, addition_correct=False)   # subset right, addition WRONG
_WC = _info(subset_correct=False, addition_correct=True)   # leakage
_WW = _info(subset_correct=False, addition_correct=False)  # both wrong


def test_filter_none_keeps_everything():
    # The unfiltered control keeps ALL four combinations — crucially the addition-wrong ones (_CW,_WW)
    # that the correctness filter drops. This is the whole point: no selection.
    for cond in (RL_CONDITION_A, RL_CONDITION_B):
        for info in (_CC, _CW, _WC, _WW):
            assert should_keep("none", cond, info) is True


def test_filter_none_differs_from_correctness_on_wrong_rollouts():
    # "none" and "correctness" must diverge exactly on the addition-wrong rollouts (else the control
    # is not actually unfiltered).
    for info in (_CW, _WW):
        assert should_keep("none", RL_CONDITION_B, info) is True
        assert should_keep("correctness", RL_CONDITION_B, info) is False
    for info in (_CC, _WC):
        assert should_keep("none", RL_CONDITION_B, info) == should_keep("correctness", RL_CONDITION_B, info)


def test_should_keep_correctness_matches_keep_rollout():
    for cond in (RL_CONDITION_A, RL_CONDITION_B):
        for info in (_CC, _CW, _WC, _WW):
            assert should_keep("correctness", cond, info) == keep_rollout(cond, info)


def test_filter_modes_and_validation():
    assert FILTER_MODES == ("correctness", "none")
    with pytest.raises(AssertionError):
        should_keep("bogus", RL_CONDITION_B, _CC)


class _MockSamplingClient:
    def __init__(self, responses):
        self._responses = list(responses)

    async def sample_async(self, *, prompt, num_samples, sampling_params):
        tokens, stop_reason = self._responses.pop(0)
        seq = SimpleNamespace(tokens=list(tokens), logprobs=[-0.5] * len(tokens), stop_reason=stop_reason)
        return SimpleNamespace(sequences=[seq])


def _enc(s: str) -> list[int]:
    return TOK.encode(s, add_special_tokens=False)


def test_masking_unchanged_on_addition_wrong_kept_rollout():
    """A Condition-B rollout whose forced Answer is deliberately WRONG (addition_correct False) is
    still imitated under filter='none'; its CE masking must be exactly the model-sampled spans."""
    ss, add = load_coupling_split("dev")[0]
    search = _enc("Trying subsets... none obviously works, let me keep going.")
    verify = _enc("Membership + sum + duplicate checks on the recalled subset.")
    subset = _enc("100, 200, 300")
    solved = _enc("no")
    wrong_answer = _enc(f"{add.answer + 12345}")   # NOT the correct addition answer
    mock = _MockSamplingClient([
        (search, "length"), (verify, "stop"), (subset, "stop"), (solved, "stop"), (wrong_answer, "stop"),
    ])
    traj, info = asyncio.run(rollout_coupled(mock, ss, add, CONDITION_B, seed=0))

    # It is genuinely addition-wrong (so the correctness filter WOULD drop it) yet "none" keeps it.
    assert info.addition_correct is False
    assert should_keep("none", RL_CONDITION_B, info) is True
    assert should_keep("correctness", RL_CONDITION_B, info) is False

    # Masking: exactly one CE datum; weight support == model-sampled action tokens; sum(mean)==1.
    datums = trajectory_to_sft_datums(traj, reduction="mean")
    assert len(datums) == 1
    datum = datums[0]
    assert set(datum.loss_fn_inputs.keys()) == {"weights", "target_tokens"}
    weights = datum.loss_fn_inputs["weights"].to_torch().tolist()
    model_input_tokens = datum.model_input.to_ints()
    target_tokens = datum.loss_fn_inputs["target_tokens"].to_torch().tolist()
    full = model_input_tokens + [target_tokens[-1]]

    is_action = [False] * len(full)
    for tr in traj.transitions:
        pos = len(tr.ob.to_ints())
        for j in range(len(tr.ac.tokens)):
            is_action[pos + j] = True
    # weights are over targets (left-shifted by one): weight[t] predicts full[t+1].
    assert abs(sum(weights) - 1.0) < 1e-6
    for t in range(len(weights)):
        assert (weights[t] > 0) == is_action[t + 1], f"weight[{t}] support wrong"

    # 'sum' reduction reproduces a 0/1 mask summing to the number of sampled tokens.
    sum_w = trajectory_to_sft_datums(traj, reduction="sum")[0].loss_fn_inputs["weights"].to_torch().tolist()
    assert sum(sum_w) == sum(len(tr.ac.tokens) for tr in traj.transitions)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
