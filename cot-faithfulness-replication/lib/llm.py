"""Cached, rate-limited, retrying Anthropic API calls.

Client construction deliberately has no hardcoded key/base-url: the SDK picks up
ANTHROPIC_API_KEY (and optionally ANTHROPIC_BASE_URL) from the environment.
"""

import asyncio
import random
from pathlib import Path
from typing import Any

import anthropic

from cost_tracker import CostTracker
from file_cache import FileCache

PROJECT_ROOT = Path(__file__).parent.parent

# Changing cache-relevant behavior of the cached call itself (not its inputs) requires a new UUID.
ANTHROPIC_CALL_UUID = "b1f5a9c2-4e8d-4f5b-9c3a-7d2e6f0a1b84"

MAX_RETRIES = 200
INITIAL_RETRY_DELAY_S = 0.5
MAX_RETRY_DELAY_S = 180.0
PER_CALL_TIMEOUT_S = 3600.0
DEFAULT_CONCURRENCY = 100

_cache: FileCache | None = None
_client: anthropic.AsyncAnthropic | None = None
_semaphores: dict[str, asyncio.Semaphore] = {}
_concurrency_limits: dict[str, int] = {}


def get_cache() -> FileCache:
    global _cache
    if _cache is None:
        _cache = FileCache(PROJECT_ROOT / "file_cache_dir")
    return _cache


def get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        # SDK-internal retries off; we do our own (SDK retries would hide 429/529 dynamics).
        _client = anthropic.AsyncAnthropic(max_retries=0, timeout=PER_CALL_TIMEOUT_S)
    return _client


def set_concurrency(model: str, n: int) -> None:
    assert model not in _semaphores, "set_concurrency must be called before the first call for the model"
    _concurrency_limits[model] = n


def _get_semaphore(model: str) -> asyncio.Semaphore:
    if model not in _semaphores:
        _semaphores[model] = asyncio.Semaphore(_concurrency_limits.get(model, DEFAULT_CONCURRENCY))
    return _semaphores[model]


_TRANSIENT_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504, 529}


def _is_transient(e: Exception) -> bool:
    if isinstance(e, (anthropic.APIConnectionError, anthropic.APITimeoutError, asyncio.TimeoutError)):
        return True
    if isinstance(e, anthropic.APIStatusError):
        # Some API gateways re-serve upstream overloaded errors under a non-529
        # status; classify by body too (observed: 'overloaded_error' raised as non-transient).
        if "overloaded_error" in str(e):
            return True
        return e.status_code in _TRANSIENT_STATUS_CODES or e.status_code >= 500
    return False


async def _call_with_retries(api_kwargs: dict[str, Any]) -> dict[str, Any]:
    """One API call (with retry loop) -> full raw response as a JSON-serializable dict.

    Uses streaming + get_final_message: identical result to non-streaming, but avoids the
    SDK's long-request/non-streaming timeout concerns at max_tokens=16k with thinking.
    """
    client = get_client()
    delay = INITIAL_RETRY_DELAY_S
    for attempt in range(MAX_RETRIES + 1):
        try:
            async with client.messages.stream(**api_kwargs) as stream:
                message = await stream.get_final_message()
            # Guard against any proxy-side model substitution: the response must come from
            # the exact model requested.
            assert message.model == api_kwargs["model"], (message.model, api_kwargs["model"])
            return message.model_dump(mode="json")
        except Exception as e:
            if not _is_transient(e) or attempt == MAX_RETRIES:
                raise
            sleep_s = delay * (0.5 + random.random())
            if attempt % 10 == 9:
                print(f"  retry {attempt + 1} after {type(e).__name__}: {e} (sleeping {sleep_s:.1f}s)")
            await asyncio.sleep(sleep_s)
            delay = min(delay * 2, MAX_RETRY_DELAY_S)
    raise AssertionError("unreachable")


async def call_anthropic_cached(
    api_kwargs: dict[str, Any],
    sample_idx: int = 0,
    cost_tracker: CostTracker | None = None,
    assert_cached: bool = False,
) -> dict[str, Any]:
    """Cached call. api_kwargs must contain ALL request parameters (they form the cache key)."""
    cache_key = {**api_kwargs, "sample_idx": sample_idx, "this_call_uuid": ANTHROPIC_CALL_UUID}

    async def compute() -> dict[str, Any]:
        async with _get_semaphore(api_kwargs["model"]):
            response = await _call_with_retries(api_kwargs)
        if cost_tracker is not None:
            usage = response["usage"]
            cost_tracker.add_llm_api_cost(
                api_kwargs["model"],
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
            )
        return response

    result, _ = await get_cache().aget_or_compute_set(cache_key, compute, assert_cached=assert_cached)
    return result
