"""In-loop RL rollout generation (route b): mirror the validated forced flow, build Trajectories.

This is the training-loop counterpart to `lib.tinker_client.sample_coupled_forced` (used by the eval
harness). It samples from the *current policy's* sampling client with per-phase token budgets, then
hand-builds a tinker_cookbook `Trajectory` whose transitions place the injected cues in the
observations (so `trajectory_to_data` gives them advantage 0) and only the model-sampled spans in the
actions (which carry the GRPO advantage). Why route (b): the cookbook's standard
policy uses one fixed `max_tokens` for every step, which can't express the search(8000)/verify(512)/
subset(48)/solved(12)/answer(16) budgets — the VERIFY phase in particular has no stop token, so the
model would resume searching and diverge from the validated budget-forced flow.

Correctness invariants (verified by tests/test_rl_masking.py):
  - Observations are built by concatenating token IDs, never by re-tokenizing concatenated text (BPE
    could merge across a boundary and break `trajectory_to_data`'s exact-prefix check).
  - Forced short lines are truncated at a token boundary; logprobs are sliced with the same index.
  - The reward is left at 0 on every transition here; the scalar trajectory reward is assigned by the
    condition (see lib.rl_conditions), so A/B/C/D share one code path.
"""

from __future__ import annotations

from dataclasses import dataclass

import tinker

from cost_tracker import CostTracker
from lib import config
from lib.addition import AdditionInstance
from lib.coupling import (
    SOLVED_CUE, SUBSET_CUE, VERIFY_CUE, Condition, parse_coupled_output,
)
from lib.eval_harness import classify_subset
from lib.parsing import parse_subset_values_answer
from lib.prompts import subset_sum_prompt
from lib.subset_sum import SubsetSumInstance, verify_subset_sum_values
from lib.tinker_client import (
    DEFAULT_FORCE_CUE, _retrying_sample, get_semaphore, get_tokenizer_and_renderer, parse_tokens,
)

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.types import Trajectory, Transition


@dataclass
class RolloutInfo:
    """Per-rollout diagnostics + the parsed answers the condition's reward function reads.

    `subset_correct` is the external verifier's verdict (never trusted from the model's `Solved:`).
    For A/B (coupled prompt) the reward-relevant field is `addition_correct`; for C (neutral prompt)
    it is `subset_correct` and the addition_* fields are unused (None/False)."""

    subset_values: list[int] | None
    solved: bool | None
    addition_answer: int | None
    subset_correct: bool
    addition_correct: bool
    addition_parse_error: bool
    subset_category: str
    forced: bool
    is_clean: bool
    n_sample_tokens: int
    n_prompt_tokens: int
    answer_text: str
    thinking_text: str


def _first_line_token_len(tokens: list[int], tokenizer) -> int:
    """Number of leading tokens whose combined decode contains no newline (first-line truncation).

    Decodes incremental prefixes (robust to multi-byte/merged tokens); returns the count of tokens
    *before* the first newline. Newlines tokenize standalone for our content (see the write-up), so
    numeric/`yes`/`no` values are preserved intact."""
    for i in range(1, len(tokens) + 1):
        if "\n" in tokenizer.decode(tokens[:i]):
            return i - 1
    return len(tokens)


async def _sample(sampling_client, model_input, *, max_tokens, stop, temperature, top_p, seed,
                  tracker, model, max_concurrency, cache=None, sampler_id=None, assert_cached=False):
    """Sample one continuation. Optional caching (used only by the OFFLINE base-model-rollout variant,
    where the sampling policy is the frozen base model, so a sample is a deterministic function of its
    inputs): when `cache` is given, the (prompt tokens, params, sampler_id) key memoizes the raw
    tokens/logprobs so re-runs are free (`--assert-cached`-verifiable). The on-policy RL/SFT path passes
    `cache=None` — the policy changes every round, so its samples are never cacheable — leaving its
    behaviour byte-identical to before."""
    params = tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                                   seed=seed, stop=stop)

    async def compute():
        async with get_semaphore(max_concurrency):
            resp = await _retrying_sample(sampling_client, model_input, params)
        seq = resp.sequences[0]
        assert seq.logprobs is not None, "sampling client must return logprobs for RL"
        if tracker is not None:
            tracker.add_tinker_cost(model, prefill_tokens=model_input.length,
                                    sample_tokens=len(seq.tokens))
        return {"tokens": list(seq.tokens), "logprobs": list(seq.logprobs),
                "stop_reason": seq.stop_reason}

    if cache is not None:
        key = {"prompt_tokens": model_input.to_ints(), "max_tokens": max_tokens,
               "temperature": temperature, "top_p": top_p, "seed": seed, "stop": list(stop),
               "model": model, "sampler_id": sampler_id, "this_call_uuid": "rl-rollout-sample-v1"}
        value, _ = await cache.aget_or_compute_set(key, compute, assert_cached=assert_cached)
    else:
        value = await compute()
    return value["tokens"], value["logprobs"], value["stop_reason"]


def _continuation_input(base_prompt, base_len: int, full_ints: list[int]) -> tinker.ModelInput:
    """A ModelInput that flattens to `full_ints`, keeping base_prompt's original chunking for the
    prompt part and one appended chunk for everything sampled/injected after it (matches the validated
    flow's `_sample_continuation`; flattening is identical to concatenating the ids)."""
    suffix = full_ints[base_len:]
    if not suffix:
        return base_prompt
    return tinker.ModelInput(chunks=list(base_prompt.chunks) + [tinker.EncodedTextChunk(tokens=suffix)])


def _coupled_info(ss, add, answer_text, thinking_text, *, forced, is_clean, n_sample_tokens,
                  n_prompt_tokens) -> RolloutInfo:
    parsed = parse_coupled_output(answer_text)
    return RolloutInfo(
        subset_values=parsed.subset_values, solved=parsed.solved,
        addition_answer=parsed.addition_answer,
        subset_correct=verify_subset_sum_values(ss.values, ss.target, parsed.subset_values),
        addition_correct=parsed.addition_answer == add.answer,
        addition_parse_error=parsed.addition_answer is None,
        subset_category=classify_subset(ss, parsed.subset_values),
        forced=forced, is_clean=is_clean, n_sample_tokens=n_sample_tokens,
        n_prompt_tokens=n_prompt_tokens, answer_text=answer_text, thinking_text=thinking_text,
    )


async def rollout_coupled(
    sampling_client,
    ss: SubsetSumInstance,
    add: AdditionInstance,
    condition: Condition,
    *,
    model: str = config.MODEL,
    renderer_name: str = config.RENDERER_NAME,
    think_budget: int = config.SUBSET_SUM_THINK_BUDGET,
    verify_budget: int = config.COUPLED_ANSWER_BUDGET,
    subset_budget: int = 48,
    solved_budget: int = 12,
    answer_budget: int = 16,
    verify_cue: str = VERIFY_CUE,
    subset_cue: str = SUBSET_CUE,
    solved_cue: str = SOLVED_CUE,
    temperature: float = config.ROLLOUT_TEMPERATURE,
    top_p: float = config.ROLLOUT_TOP_P,
    seed: int = 0,
    tracker: CostTracker | None = None,
    max_concurrency: int = 64,
    cache=None,
    sampler_id: str | None = None,
    assert_cached: bool = False,
) -> tuple[Trajectory, RolloutInfo]:
    """One coupled (A/B/D-prompt) rollout against `sampling_client`, mirroring sample_coupled_forced.

    Returns (Trajectory, RolloutInfo). The trajectory's per-transition reward is 0; the condition
    assigns the scalar reward. `condition.answer_cue` differs A vs B (the only forced-line difference).

    `cache`/`sampler_id`/`assert_cached` are used ONLY by the offline base-model-rollout SFT variant
    (frozen base policy → deterministic, so cacheable); the on-policy path leaves `cache=None`."""
    tokenizer, renderer = get_tokenizer_and_renderer(model, renderer_name)
    stop = renderer.get_stop_sequences()
    answer_cue = condition.answer_cue
    messages = [{"role": "user", "content": condition.prompt_fn(ss, add)}]
    base_prompt = renderer.build_generation_prompt(messages)
    base_ints = base_prompt.to_ints()
    base_len = len(base_ints)
    n_prompt_tokens = base_prompt.length

    def enc(s: str) -> list[int]:
        return tokenizer.encode(s, add_special_tokens=False)

    common = dict(stop=stop, temperature=temperature, top_p=top_p, seed=seed, tracker=tracker,
                  model=model, max_concurrency=max_concurrency, cache=cache, sampler_id=sampler_id,
                  assert_cached=assert_cached)

    # Phase 1: bounded search.
    search_tokens, search_lp, search_stop = await _sample(
        sampling_client, base_prompt, max_tokens=think_budget, **common)
    is_clean = search_stop == "stop"  # hit a stop sequence (im_end) -> concluded on its own
    if is_clean:
        answer_text, thinking_text, _ = parse_tokens(search_tokens, renderer)
        transitions = [Transition(
            ob=base_prompt, ac=TokensWithLogprobs(search_tokens, search_lp, search_stop),
            reward=0.0, episode_done=True)]
        final_ob = _continuation_input(base_prompt, base_len, base_ints + search_tokens)
        info = _coupled_info(ss, add, answer_text, thinking_text, forced=False, is_clean=True,
                             n_sample_tokens=len(search_tokens), n_prompt_tokens=n_prompt_tokens)
        return Trajectory(transitions=transitions, final_ob=final_ob), info

    transitions = [Transition(
        ob=base_prompt, ac=TokensWithLogprobs(search_tokens, search_lp, search_stop),
        reward=0.0, episode_done=False)]
    cumulative = base_ints + search_tokens
    n_sampled = len(search_tokens)

    async def forced_phase(cue: str, budget: int, first_line: bool) -> str:
        nonlocal cumulative, n_sampled
        ob_ints = cumulative + enc(cue)
        ob = _continuation_input(base_prompt, base_len, ob_ints)
        toks, lp, sr = await _sample(sampling_client, ob, max_tokens=budget, **common)
        if first_line:
            k = _first_line_token_len(toks, tokenizer)
            toks, lp = toks[:k], lp[:k]
        transitions.append(Transition(
            ob=ob, ac=TokensWithLogprobs(toks, lp, sr), reward=0.0, episode_done=False))
        n_sampled += len(toks)
        cumulative = ob_ints + toks
        return tokenizer.decode(toks)

    verify_text = await forced_phase(verify_cue, verify_budget, first_line=False)
    subset_text = await forced_phase(subset_cue, subset_budget, first_line=True)
    solved_text = await forced_phase(solved_cue, solved_budget, first_line=True)
    answer_line = await forced_phase(answer_cue, answer_budget, first_line=True)
    transitions[-1].episode_done = True

    verify_visible = verify_cue.split("</think>", 1)[-1].lstrip("\n")
    answer_text = (verify_visible + verify_text + subset_cue + subset_text
                   + solved_cue + solved_text + answer_cue + answer_line)
    final_ob = _continuation_input(base_prompt, base_len, cumulative)
    info = _coupled_info(ss, add, answer_text, tokenizer.decode(search_tokens), forced=True,
                         is_clean=False, n_sample_tokens=n_sampled, n_prompt_tokens=n_prompt_tokens)
    return Trajectory(transitions=transitions, final_ob=final_ob), info


async def rollout_budget_forced(
    sampling_client,
    ss: SubsetSumInstance,
    *,
    model: str = config.MODEL,
    renderer_name: str = config.RENDERER_NAME,
    think_budget: int = config.SUBSET_SUM_THINK_BUDGET,
    answer_budget: int = 96,
    force_cue: str = DEFAULT_FORCE_CUE,
    temperature: float = config.ROLLOUT_TEMPERATURE,
    top_p: float = config.ROLLOUT_TOP_P,
    seed: int = 0,
    tracker: CostTracker | None = None,
    max_concurrency: int = 64,
) -> tuple[Trajectory, RolloutInfo]:
    """One neutral subset-sum rollout (condition C) with single-cue budget forcing (mirrors
    sample_budget_forced). Reward-relevant field is `subset_correct`; addition_* are unused."""
    tokenizer, renderer = get_tokenizer_and_renderer(model, renderer_name)
    stop = renderer.get_stop_sequences()
    messages = [{"role": "user", "content": subset_sum_prompt(ss)}]
    base_prompt = renderer.build_generation_prompt(messages)
    base_ints = base_prompt.to_ints()
    base_len = len(base_ints)
    n_prompt_tokens = base_prompt.length
    common = dict(stop=stop, temperature=temperature, top_p=top_p, seed=seed, tracker=tracker,
                  model=model, max_concurrency=max_concurrency)

    search_tokens, search_lp, search_stop = await _sample(
        sampling_client, base_prompt, max_tokens=think_budget, **common)
    is_clean = search_stop == "stop"
    if is_clean:
        answer_text, thinking_text, _ = parse_tokens(search_tokens, renderer)
        transitions = [Transition(
            ob=base_prompt, ac=TokensWithLogprobs(search_tokens, search_lp, search_stop),
            reward=0.0, episode_done=True)]
        final_ob = _continuation_input(base_prompt, base_len, base_ints + search_tokens)
        info = _c_info(ss, answer_text, thinking_text, forced=False, is_clean=True,
                       n_sample_tokens=len(search_tokens), n_prompt_tokens=n_prompt_tokens)
        return Trajectory(transitions=transitions, final_ob=final_ob), info

    # Force a committed answer.
    force_tokens = tokenizer.encode(force_cue, add_special_tokens=False)
    ob_ints = base_ints + search_tokens + force_tokens
    ob1 = _continuation_input(base_prompt, base_len, ob_ints)
    ans_tokens, ans_lp, ans_stop = await _sample(sampling_client, ob1, max_tokens=answer_budget, **common)
    transitions = [
        Transition(ob=base_prompt, ac=TokensWithLogprobs(search_tokens, search_lp, search_stop),
                   reward=0.0, episode_done=False),
        Transition(ob=ob1, ac=TokensWithLogprobs(ans_tokens, ans_lp, ans_stop),
                   reward=0.0, episode_done=True),
    ]
    cue_visible = force_cue.split("</think>", 1)[-1].lstrip("\n")
    answer_text = cue_visible + tokenizer.decode(ans_tokens)
    final_ob = _continuation_input(base_prompt, base_len, ob_ints + ans_tokens)
    info = _c_info(ss, answer_text, tokenizer.decode(search_tokens), forced=True, is_clean=False,
                   n_sample_tokens=len(search_tokens) + len(ans_tokens), n_prompt_tokens=n_prompt_tokens)
    return Trajectory(transitions=transitions, final_ob=final_ob), info


def _c_info(ss, answer_text, thinking_text, *, forced, is_clean, n_sample_tokens, n_prompt_tokens):
    subset_values = parse_subset_values_answer(answer_text)
    return RolloutInfo(
        subset_values=subset_values, solved=None, addition_answer=None,
        subset_correct=verify_subset_sum_values(ss.values, ss.target, subset_values),
        addition_correct=False, addition_parse_error=False,
        subset_category=classify_subset(ss, subset_values),
        forced=forced, is_clean=is_clean, n_sample_tokens=n_sample_tokens,
        n_prompt_tokens=n_prompt_tokens, answer_text=answer_text, thinking_text=thinking_text,
    )
