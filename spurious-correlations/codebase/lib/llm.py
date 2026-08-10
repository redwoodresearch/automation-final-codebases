"""Cached, retried, cost-tracked async Anthropic calls used by generation and validation.

One semaphore per model. The cache key includes every API argument plus sample_idx and a
call_uuid so a prompt-template change can be invalidated explicitly (INSTRUCTIONS §4).
"""

from __future__ import annotations

import asyncio
import os

import anthropic
import httpx

from cost_tracker import CostTracker
from file_cache import FileCache
import config

# ANTHROPIC_API_KEY is provided in the environment; the client picks it up automatically. We check
# it lazily (at the first API call) rather than at import, so analysis/plotting code that imports
# this module transitively never needs a key.
def _require_api_key() -> None:
    assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY must be set in the environment"

# Larger httpx connection pool: the default (~100 max connections) is exhausted when we run two models
# at high concurrency (gen + judge), which surfaced as APIConnectionError/APITimeoutError storms above
# ~60 concurrency. A 300-connection pool removes that ceiling so concurrency is bounded only by the
# semaphores + the org rate limit. (read timeout is generous for extended-thinking judge streams.)
_http_client = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=300, max_keepalive_connections=60),
    timeout=httpx.Timeout(connect=20.0, read=600.0, write=20.0, pool=20.0),
)
# Explicit base_url: the agent harness exports ANTHROPIC_BASE_URL pointing at a local proxy that
# injects params (e.g. `fallbacks`) some models reject; experiment calls must hit the real API.
_client = anthropic.AsyncAnthropic(http_client=_http_client, base_url="https://api.anthropic.com")
_cache = FileCache(str(config.CACHE_DIR))
_cost_tracker = CostTracker(cost_file=config.COST_FILE, run_description="llm.py (default)")

# Per-model semaphores (rate limits are per model). The org's shared sonnet input-token/min limit is
# the real ceiling for the large judge-bound generation and is contended by other agents, so we stay a
# good citizen at ~60 (90/140 just added 429 churn without throughput, since others dominate the pool).
# Exponential backoff handles the remaining blips. Not part of the cache key.
_semaphores: dict[str, asyncio.Semaphore] = {}
# 60 is the proven-clean level (the 128-cell base run had ZERO connection errors at 60); above it the
# sustained extended-thinking judge streams exhaust connections even with the enlarged pool. Stability
# over speed for the long generation.
_DEFAULT_CONCURRENCY = 60

# Module-level toggle flipped by entry-point scripts.
assert_cached = False

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 529}
RETRYABLE_STRINGS = [
    "rate limit", "overloaded", "timeout", "timed out", "dropped connection", "internal server error",
    # transient streaming / network drops (a mid-stream httpx.RemoteProtocolError killed a long run once)
    "peer closed connection", "incomplete chunked read", "remoteprotocolerror", "connection reset",
    "server disconnected", "connection error", "read error", "connection aborted", "eof occurred",
]


def set_cost_tracker(tracker: CostTracker) -> None:
    """Let an entry-point script register its own tracker so costs land under its description."""
    global _cost_tracker
    _cost_tracker = tracker


class _EmptyCompletion(Exception):
    """Raised internally when the API returns an empty completion, to force a retry without caching."""


def _semaphore(model: str) -> asyncio.Semaphore:
    if model not in _semaphores:
        _semaphores[model] = asyncio.Semaphore(_DEFAULT_CONCURRENCY)
    return _semaphores[model]


def _is_retryable(e: Exception) -> bool:
    if isinstance(e, (asyncio.TimeoutError, TimeoutError)):
        return True
    if isinstance(e, anthropic.APIConnectionError):
        return True
    # raw httpx/network errors can escape the streaming iterator without being wrapped as an
    # anthropic.APIConnectionError (e.g. RemoteProtocolError mid-stream) — treat all as transient.
    if isinstance(e, (httpx.TransportError, httpx.HTTPError)):
        return True
    if isinstance(e, anthropic.APIStatusError) and e.status_code in RETRYABLE_STATUS_CODES:
        return True
    err = str(e).lower()
    return any(s in err for s in RETRYABLE_STRINGS) or any(str(c) in err for c in RETRYABLE_STATUS_CODES)


async def call_llm(
    *,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 1.0,
    system: str | None = None,
    thinking_tokens: int | None = None,
    sample_idx: int = 0,
    call_uuid: str = "v1",
) -> str:
    """Return the assistant text (concatenated text blocks). Cached on every input."""
    _require_api_key()
    api_kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system is not None:
        api_kwargs["system"] = system
    if thinking_tokens is not None:
        # Extended thinking requires temperature == 1.0.
        api_kwargs["temperature"] = 1.0
        api_kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_tokens}

    cache_key = {**api_kwargs, "sample_idx": sample_idx, "this_call_uuid": call_uuid}

    async def do_call() -> str:
        async with _semaphore(model):
            delay, max_delay, max_retries = 0.5, 180.0, 200
            for attempt in range(max_retries + 1):
                try:
                    async with _client.messages.stream(**api_kwargs) as stream:
                        response = await asyncio.wait_for(stream.get_final_message(), timeout=60 * 120)
                    text = "".join(b.text for b in response.content if b.type == "text")
                    _cost_tracker.add_llm_api_cost(model, response.usage.input_tokens, response.usage.output_tokens)
                    # A rare empty completion must NOT be cached (a poisoned cache entry re-fails every
                    # rerun). Retry a few times; if it persists, return "" and let the caller's parser
                    # degrade it. Empty is raised pre-return so aget_or_compute_set never stores it.
                    if not text.strip() and attempt < 5:
                        raise _EmptyCompletion()
                    return text
                except Exception as e:
                    if isinstance(e, _EmptyCompletion):
                        if attempt == max_retries:
                            return ""
                    elif not _is_retryable(e) or attempt == max_retries:
                        raise
                    this_delay = min(delay, max_delay)
                    print(f"[llm] retryable {type(e).__name__}: {e} (attempt {attempt + 1}, sleep {this_delay:.1f}s)")
                    await asyncio.sleep(this_delay)
                    delay *= 2
            raise RuntimeError("unreachable")

    result, _ = await _cache.aget_or_compute_set(cache_key, do_call, assert_cached=assert_cached)
    return result
