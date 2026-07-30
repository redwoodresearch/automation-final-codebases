"""Tinker sampling with caching and cost tracking.

Provides a cached async sampler for the project's reasoning model. Callers pass a message
list; we render with the qwen3_5 renderer, sample, parse the response into thinking/answer
text, and cache the raw tokens keyed on (model, renderer, messages, sampling params,
sample_idx). Reruns hit the cache and cost nothing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from cost_tracker import CostTracker
from file_cache import FileCache
from lib import config
from lib.parsing import get_answer_text

# Bump if the meaning of a cache entry changes (not just its inputs).
_CACHE_UUID = "tinker-sample-v1"

# Transient Tinker/network errors worth retrying.
_MAX_RETRIES = 60
_RETRY_BASE_DELAY = 1.0
_RETRY_MAX_DELAY = 120.0

_sampling_semaphore: asyncio.Semaphore | None = None


def get_semaphore(max_concurrency: int = 64) -> asyncio.Semaphore:
    global _sampling_semaphore
    if _sampling_semaphore is None:
        _sampling_semaphore = asyncio.Semaphore(max_concurrency)
    return _sampling_semaphore


@lru_cache(maxsize=4)
def get_tokenizer_and_renderer(model: str = config.MODEL, renderer_name: str = config.RENDERER_NAME):
    tokenizer = get_tokenizer(model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    return tokenizer, renderer


@lru_cache(maxsize=4)
def get_sampling_client(model: str = config.MODEL) -> tinker.SamplingClient:
    service_client = tinker.ServiceClient()
    return service_client.create_sampling_client(base_model=model)


@dataclass
class SampleResult:
    tokens: list[int]
    stop_reason: str
    is_clean: bool
    answer_text: str  # visible text after </think>
    thinking_text: str
    n_prompt_tokens: int
    n_sample_tokens: int
    forced: bool = False  # True if the answer was produced by budget-forcing (see sample_budget_forced)


@dataclass
class Sampler:
    """A sampling backend for the eval harness: the base model, or a training checkpoint.

    Bundles everything the sample_* functions need to route + cache correctly. The eval harness constructs
    a Sampler around a checkpoint's SamplingClient (with a `sampler_id` identifying the checkpoint)
    and hands it to the harness; this segment uses the default BASE_SAMPLER (base model, reusing
    the base-model cache).

    - sampling_client: a specific client (e.g. a checkpoint's); None -> the base-model client.
    - model: model id, used for the tokenizer/renderer and for cost pricing (base pricing applies
      to LoRA checkpoints too).
    - sampler_id: appended to cache keys so a checkpoint's samples don't collide with base-model
      ones. MUST be set when sampling_client is not None. None -> base-model keys (base-model cache).
    - cache_enabled: False bypasses the cache entirely (for training-time evals not worth persisting).
    """

    sampling_client: Any | None = None
    model: str = config.MODEL
    sampler_id: str | None = None
    cache_enabled: bool = True

    def sample_kwargs(self) -> dict[str, Any]:
        return {
            "sampling_client": self.sampling_client,
            "model": self.model,
            "sampler_id": self.sampler_id,
            "cache_enabled": self.cache_enabled,
        }


BASE_SAMPLER = Sampler()


def parse_tokens(tokens: list[int], renderer=None) -> tuple[str, str, bool]:
    """Parse sampled tokens into (answer_text, thinking_text, is_clean)."""
    if renderer is None:
        _, renderer = get_tokenizer_and_renderer()
    message, termination = renderer.parse_response(tokens)
    answer_text = get_answer_text(message)
    thinking_text = ""
    content = message.get("content")
    if isinstance(content, list):
        thinking_text = "\n".join(
            p.get("thinking", "") for p in content if isinstance(p, dict) and p.get("type") == "thinking"
        )
    return answer_text, thinking_text, termination.is_clean


def _cache_key(model, renderer_name, messages, temperature, top_p, max_tokens, sample_idx,
               sampler_id=None) -> dict[str, Any]:
    key = {
        "model": model,
        "renderer": renderer_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "sample_idx": sample_idx,
        "this_call_uuid": _CACHE_UUID,
    }
    # Only add sampler_id when sampling from a non-base source (e.g. a training checkpoint), so
    # base-model keys are byte-identical across callers and reuse the same cache.
    if sampler_id is not None:
        key["sampler_id"] = sampler_id
    return key


async def _retrying_sample(sampling_client, prompt, sampling_params):
    delay = _RETRY_BASE_DELAY
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            return await sampling_client.sample_async(
                prompt=prompt, num_samples=1, sampling_params=sampling_params
            )
        except Exception as exc:  # noqa: BLE001 - retry transient errors, then re-raise
            last_exc = exc
            if attempt == _MAX_RETRIES - 1:
                break
            await asyncio.sleep(delay)
            delay = min(delay * 2, _RETRY_MAX_DELAY)
    raise last_exc


async def sample_cached(
    messages: list[dict[str, Any]],
    *,
    cache: FileCache,
    sample_idx: int = 0,
    temperature: float = config.ROLLOUT_TEMPERATURE,
    top_p: float = config.ROLLOUT_TOP_P,
    max_tokens: int = config.MAX_TOKENS,
    model: str = config.MODEL,
    renderer_name: str = config.RENDERER_NAME,
    tracker: CostTracker | None = None,
    assert_cached: bool = False,
    max_concurrency: int = 64,
    sampling_client: "tinker.SamplingClient | None" = None,
    sampler_id: str | None = None,
    cache_enabled: bool = True,
) -> SampleResult:
    """Sample one completion for a message list (cached). Tracks Tinker cost on cache miss.

    Checkpoint pluggability: pass `sampling_client` (a checkpoint's client) with a matching
    `sampler_id` (added to the cache key so checkpoint samples don't collide with base-model ones).
    Leave both None for the base model (reuses the base-model cache). `cache_enabled=False` bypasses the
    cache entirely (for training-time evals you don't want to persist).
    """
    if sampling_client is not None and sampler_id is None:
        raise ValueError("sampler_id must be set when a non-base sampling_client is provided "
                         "(so checkpoint samples don't collide with base-model cache entries)")
    tokenizer, renderer = get_tokenizer_and_renderer(model, renderer_name)
    key = _cache_key(model, renderer_name, messages, temperature, top_p, max_tokens, sample_idx,
                     sampler_id=sampler_id)

    prompt = renderer.build_generation_prompt(messages)
    n_prompt_tokens = prompt.length

    async def compute():
        client = sampling_client if sampling_client is not None else get_sampling_client(model)
        sampling_params = tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=sample_idx,
            stop=renderer.get_stop_sequences(),
        )
        semaphore = get_semaphore(max_concurrency)
        async with semaphore:
            resp = await _retrying_sample(client, prompt, sampling_params)
        seq = resp.sequences[0]
        if tracker is not None:
            tracker.add_tinker_cost(
                model, prefill_tokens=n_prompt_tokens, sample_tokens=len(seq.tokens)
            )
        return {"tokens": list(seq.tokens), "stop_reason": seq.stop_reason}

    if cache_enabled:
        value, _computed = await cache.aget_or_compute_set(key, compute, assert_cached=assert_cached)
    else:
        if assert_cached:
            raise RuntimeError("assert_cached=True is incompatible with cache_enabled=False")
        value = await compute()

    tokens = value["tokens"]
    answer_text, thinking_text, is_clean = parse_tokens(tokens, renderer)
    return SampleResult(
        tokens=tokens,
        stop_reason=value["stop_reason"],
        is_clean=is_clean,
        answer_text=answer_text,
        thinking_text=thinking_text,
        n_prompt_tokens=n_prompt_tokens,
        n_sample_tokens=len(tokens),
    )


# --- Budget forcing ---
# The base model tends to search a checkable task (subset sum) until it exhausts the token
# budget and truncates with no answer. To measure "capability within a bounded CoT" and to
# guarantee an answer is emitted (so a second task's answer has room), we cap the thinking at
# `think_budget` tokens; if the model hasn't concluded by then, we inject `</think>` + a commit
# cue and sample a short final answer (s1-style budget forcing).

# Default commit cue injected after the (truncated) thinking. Ends at "Answer:" so the forced
# continuation only needs to produce the answer content.
DEFAULT_FORCE_CUE = (
    "</think>\n\nI've used up my reasoning budget. Based on the work above, here is my best "
    "answer:\nAnswer:"
)


async def sample_budget_forced(
    messages: list[dict[str, Any]],
    *,
    cache: FileCache,
    think_budget: int,
    sample_idx: int = 0,
    answer_budget: int = 96,
    force_cue: str = DEFAULT_FORCE_CUE,
    temperature: float = config.ROLLOUT_TEMPERATURE,
    top_p: float = config.ROLLOUT_TOP_P,
    model: str = config.MODEL,
    renderer_name: str = config.RENDERER_NAME,
    tracker: CostTracker | None = None,
    assert_cached: bool = False,
    max_concurrency: int = 64,
    sampling_client: "tinker.SamplingClient | None" = None,
    sampler_id: str | None = None,
    cache_enabled: bool = True,
) -> SampleResult:
    """Sample with a hard CoT budget and forced answer.

    Phase 1: think for up to `think_budget` tokens. If the model concludes cleanly within the
    budget, its answer is used as-is (forced=False). Otherwise (phase 2) we append `</think>` +
    `force_cue` to the truncated thinking and sample up to `answer_budget` more tokens to get a
    committed answer (forced=True). Both phases are cached and cost-tracked.

    Checkpoint pluggability: see `sample_cached` (`sampling_client`/`sampler_id`/`cache_enabled`).
    """
    if sampling_client is not None and sampler_id is None:
        raise ValueError("sampler_id must be set when a non-base sampling_client is provided")
    tokenizer, renderer = get_tokenizer_and_renderer(model, renderer_name)
    stop = renderer.get_stop_sequences()

    # Phase 1: bounded thinking.
    phase1 = await sample_cached(
        messages, cache=cache, sample_idx=sample_idx, temperature=temperature, top_p=top_p,
        max_tokens=think_budget, model=model, renderer_name=renderer_name, tracker=tracker,
        assert_cached=assert_cached, max_concurrency=max_concurrency,
        sampling_client=sampling_client, sampler_id=sampler_id, cache_enabled=cache_enabled,
    )
    if phase1.is_clean:
        # Concluded within budget on its own — no forcing needed.
        return phase1

    # Phase 2: force a committed answer.
    base_prompt = renderer.build_generation_prompt(messages)
    force_tokens = tokenizer.encode(force_cue, add_special_tokens=False)
    phase2_chunks = list(base_prompt.chunks) + [
        tinker.types.EncodedTextChunk(tokens=list(phase1.tokens) + force_tokens)
    ]
    phase2_input = tinker.ModelInput(chunks=phase2_chunks)
    key2 = {
        "model": model, "renderer": renderer_name, "messages": messages,
        "temperature": temperature, "top_p": top_p, "think_budget": think_budget,
        "answer_budget": answer_budget, "force_cue": force_cue, "sample_idx": sample_idx,
        "this_call_uuid": "tinker-budget-force-v1",
    }
    if sampler_id is not None:
        key2["sampler_id"] = sampler_id

    async def compute2():
        client = sampling_client if sampling_client is not None else get_sampling_client(model)
        sampling_params = tinker.SamplingParams(
            max_tokens=answer_budget, temperature=temperature, top_p=top_p,
            seed=sample_idx, stop=stop,
        )
        semaphore = get_semaphore(max_concurrency)
        async with semaphore:
            resp = await _retrying_sample(client, phase2_input, sampling_params)
        seq = resp.sequences[0]
        if tracker is not None:
            tracker.add_tinker_cost(model, prefill_tokens=phase2_input.length, sample_tokens=len(seq.tokens))
        return {"tokens": list(seq.tokens), "stop_reason": seq.stop_reason}

    if cache_enabled:
        value2, _ = await cache.aget_or_compute_set(key2, compute2, assert_cached=assert_cached)
    else:
        value2 = await compute2()
    phase2_tokens = value2["tokens"]

    # The committed answer is the forced continuation only. We build answer_text directly from the
    # cue + phase-2 output rather than re-parsing the full sequence: the phase-1 reasoning (which
    # often echoes the prompt's example answer, and which the renderer sometimes mis-classifies as
    # answer text) must NOT be scanned for the answer, or the parser grabs the echoed example.
    thinking_text = tokenizer.decode(phase1.tokens)
    cue_visible = force_cue.split("</think>", 1)[-1].lstrip("\n")  # drop the </think> prefix
    answer_text = cue_visible + tokenizer.decode(phase2_tokens)
    full_tokens = list(phase1.tokens) + force_tokens + phase2_tokens
    return SampleResult(
        tokens=full_tokens,
        stop_reason=value2["stop_reason"],
        is_clean=True,  # an answer was forced
        answer_text=answer_text,
        thinking_text=thinking_text,
        n_prompt_tokens=phase1.n_prompt_tokens,
        n_sample_tokens=len(phase1.tokens) + len(phase2_tokens),
        forced=True,
    )


# --- Structured coupled forcing (conditions A/B) ------------------------------------------
# Letting the base model free-generate the final answer block after a single stop cue is not enough:
# it resumes searching, dumps the whole list as a "candidate", writes its real subset as prose
# instead of on the `Subset:` line, and mentions the true sum in its reasoning — all of which make
# the output messy and hard to score. `sample_coupled_forced` fixes this with a structured,
# single-trajectory forced flow (RL-compatible: the injected cues are fixed spans to be loss-masked;
# only the model-sampled spans carry the policy gradient). It applies the proven s1 mechanism — a cue
# ending at a label + a tiny budget forces a clean immediate commit — to EACH terminal line:
#   Phase 1 (search):  think up to `think_budget`.
#   Phase 2 (verify):  inject `verify_cue` (ends the think block); sample the bounded mechanical
#                      membership/sum/duplicate check (the model checks a recalled subset reliably).
#   Phase 3 (subset):  inject `subset_cue` ending at "Subset:"; sample a tiny span, keep the FIRST
#                      line -> the final committed subset (or `none`).
#   Phase 4 (solved):  inject `solved_cue` ending at "Solved:"; keep the first line -> yes/no verdict.
#   Phase 5 (answer):  inject `answer_cue` ending at "Answer:"; keep the first line -> the addition
#                      answer per the coupling rule.
# The three forced terminal lines are clean and unambiguous to parse (no rambling / re-search).


async def _sample_continuation(
    *, base_prompt, prefix_tokens, max_tokens, cache, cache_key, tracker, model, stop,
    temperature, top_p, sample_idx, assert_cached, max_concurrency, sampling_client, cache_enabled,
) -> list[int]:
    """Sample a continuation after (base_prompt + prefix_tokens); return the sampled token ids."""
    model_input = tinker.ModelInput(
        chunks=list(base_prompt.chunks) + [tinker.types.EncodedTextChunk(tokens=list(prefix_tokens))]
    )

    async def compute():
        client = sampling_client if sampling_client is not None else get_sampling_client(model)
        params = tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                                       seed=sample_idx, stop=stop)
        async with get_semaphore(max_concurrency):
            resp = await _retrying_sample(client, model_input, params)
        seq = resp.sequences[0]
        if tracker is not None:
            tracker.add_tinker_cost(model, prefill_tokens=model_input.length, sample_tokens=len(seq.tokens))
        return {"tokens": list(seq.tokens)}

    if cache_enabled:
        value, _ = await cache.aget_or_compute_set(cache_key, compute, assert_cached=assert_cached)
    else:
        value = await compute()
    return value["tokens"]


async def sample_coupled_forced(
    messages: list[dict[str, Any]],
    *,
    cache: FileCache,
    think_budget: int,
    verify_cue: str,
    subset_cue: str,
    solved_cue: str,
    answer_cue: str,
    sample_idx: int = 0,
    verify_budget: int = 512,
    subset_budget: int = 48,
    solved_budget: int = 12,
    answer_budget: int = 16,
    temperature: float = config.ROLLOUT_TEMPERATURE,
    top_p: float = config.ROLLOUT_TOP_P,
    model: str = config.MODEL,
    renderer_name: str = config.RENDERER_NAME,
    tracker: CostTracker | None = None,
    assert_cached: bool = False,
    max_concurrency: int = 64,
    sampling_client: "tinker.SamplingClient | None" = None,
    sampler_id: str | None = None,
    cache_enabled: bool = True,
) -> SampleResult:
    """Structured forcing for the coupled prompt (see module note above): search -> bounded verify ->
    forced `Subset:` -> forced `Solved:` -> forced `Answer:` (each terminal line label-anchored and
    first-line-truncated so it's clean and unambiguous)."""
    if sampling_client is not None and sampler_id is None:
        raise ValueError("sampler_id must be set when a non-base sampling_client is provided")
    tokenizer, renderer = get_tokenizer_and_renderer(model, renderer_name)
    stop = renderer.get_stop_sequences()
    base_prompt = renderer.build_generation_prompt(messages)

    # Phase 1: bounded thinking.
    phase1 = await sample_cached(
        messages, cache=cache, sample_idx=sample_idx, temperature=temperature, top_p=top_p,
        max_tokens=think_budget, model=model, renderer_name=renderer_name, tracker=tracker,
        assert_cached=assert_cached, max_concurrency=max_concurrency,
        sampling_client=sampling_client, sampler_id=sampler_id, cache_enabled=cache_enabled,
    )
    if phase1.is_clean:
        return phase1  # concluded on its own (already followed the prompt's output format)

    key_common = {
        "messages": messages, "think_budget": think_budget, "sample_idx": sample_idx,
        "temperature": temperature, "top_p": top_p, "verify_cue": verify_cue,
        "subset_cue": subset_cue, "solved_cue": solved_cue, "answer_cue": answer_cue,
        "verify_budget": verify_budget, "subset_budget": subset_budget,
        "solved_budget": solved_budget, "answer_budget": answer_budget,
        "this_call_uuid": "tinker-coupled-force-v3",
    }
    if sampler_id is not None:
        key_common["sampler_id"] = sampler_id
    shared = dict(base_prompt=base_prompt, cache=cache, tracker=tracker, model=model, stop=stop,
                  temperature=temperature, top_p=top_p, sample_idx=sample_idx,
                  assert_cached=assert_cached, max_concurrency=max_concurrency,
                  sampling_client=sampling_client, cache_enabled=cache_enabled)

    def enc(s):
        return tokenizer.encode(s, add_special_tokens=False)

    n_sampled = len(phase1.tokens)

    async def forced_line(prefix, cue, budget, phase_name, first_line_only):
        """Inject `cue` after `prefix`, sample `budget` tokens; return (visible_text, new_prefix)."""
        nonlocal n_sampled
        cue_tokens = enc(cue)
        out = await _sample_continuation(
            prefix_tokens=prefix + cue_tokens, max_tokens=budget,
            cache_key={**key_common, "phase": phase_name}, **shared)
        n_sampled += len(out)
        text = tokenizer.decode(out)
        if first_line_only:
            text = text.split("\n", 1)[0]
        return text, prefix + cue_tokens + enc(text)

    # Phase 2 (verify): mechanical membership/sum/duplicate check of the recalled subset (bounded).
    verify_text, prefix = await forced_line(list(phase1.tokens), verify_cue, verify_budget,
                                            "verify", first_line_only=False)
    # Phase 3 (subset): force the final committed subset (one line).
    subset_text, prefix = await forced_line(prefix, subset_cue, subset_budget, "subset", True)
    # Phase 4 (solved): force the yes/no verdict (one line).
    solved_text, prefix = await forced_line(prefix, solved_cue, solved_budget, "solved", True)
    # Phase 5 (answer): force the addition answer per the coupling rule (one line).
    answer_text_line, prefix = await forced_line(prefix, answer_cue, answer_budget, "answer", True)

    verify_visible = verify_cue.split("</think>", 1)[-1].lstrip("\n")
    answer_text = (verify_visible + verify_text + subset_cue + subset_text
                   + solved_cue + solved_text + answer_cue + answer_text_line)
    return SampleResult(
        tokens=prefix,
        stop_reason="forced",
        is_clean=True,
        answer_text=answer_text,
        thinking_text=tokenizer.decode(phase1.tokens),
        n_prompt_tokens=phase1.n_prompt_tokens,
        n_sample_tokens=n_sampled,
        forced=True,
    )
