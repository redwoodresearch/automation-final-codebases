"""Cached, rate-limited, retrying OpenRouter calls for open-weight reasoning models.

Same integrity guarantees as the Anthropic path (lib/llm.py):
- content-addressed caching of the full response (assembled from the SSE stream; streaming
  avoids gateway timeouts on multi-minute R1 generations)
- the served model MUST equal the requested model (assert), and when the request pins a
  single provider (allow_fallbacks=False) the serving provider must match too
- full usage (incl. OpenRouter's own reported cost in USD) persisted per response

Requests carry `reasoning: {"enabled": true}` so reasoning models return their raw chain
of thought in message.reasoning, separate from the final message.content.
"""

import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any

import httpx

from cost_tracker import CostTracker
from lib.llm import get_cache

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Changing cache-relevant behavior of the cached call itself (not its inputs) requires a new UUID.
OPENROUTER_CALL_UUID = "3f7c2d81-9a54-4e06-b2c8-5d1e4a9f60b7"

MAX_RETRIES = 200
INITIAL_RETRY_DELAY_S = 0.5
MAX_RETRY_DELAY_S = 180.0
CONNECT_TIMEOUT_S = 30.0
READ_TIMEOUT_S = 600.0  # max silent gap mid-stream (OpenRouter sends keep-alive comments)
TOTAL_TIMEOUT_S = 3600.0
DEFAULT_CONCURRENCY = 50

_client: httpx.AsyncClient | None = None
_semaphores: dict[str, asyncio.Semaphore] = {}
_concurrency_limits: dict[str, int] = {}


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        key_file = Path.home() / ".openrouter_api_key"
        assert key_file.exists(), "no OPENROUTER_API_KEY env var and no ~/.openrouter_api_key"
        key = key_file.read_text().strip()
    return key


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {_api_key()}"},
            timeout=httpx.Timeout(TOTAL_TIMEOUT_S, connect=CONNECT_TIMEOUT_S, read=READ_TIMEOUT_S),
            limits=httpx.Limits(max_connections=500, max_keepalive_connections=100),
        )
    return _client


def set_concurrency(model: str, n: int) -> None:
    assert model not in _semaphores, "set_concurrency must be called before the first call for the model"
    _concurrency_limits[model] = n


def _get_semaphore(model: str) -> asyncio.Semaphore:
    if model not in _semaphores:
        _semaphores[model] = asyncio.Semaphore(_concurrency_limits.get(model, DEFAULT_CONCURRENCY))
    return _semaphores[model]


class OpenRouterError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


_TRANSIENT_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504, 529}


def _is_transient(e: Exception) -> bool:
    if isinstance(e, (httpx.TransportError, asyncio.TimeoutError, json.JSONDecodeError)):
        return True  # network flake / stalled or garbled stream
    if isinstance(e, OpenRouterError):
        if e.status_code is None:
            return True  # mid-stream provider failure with no HTTP status
        return e.status_code in _TRANSIENT_STATUS_CODES or e.status_code >= 500
    return False


def assemble_stream_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """SSE data events -> one response dict in the non-streaming chat.completion shape.

    Kept separate from I/O for testability. Raises OpenRouterError on in-stream error
    events (provider failures surface this way even under HTTP 200).
    """
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    finish_reason = native_finish_reason = None
    usage = model = provider = response_id = created = None
    for event in events:
        error = event.get("error")
        if error:
            raise OpenRouterError(f"in-stream error: {error}", status_code=error.get("code"))
        model = event.get("model") or model
        provider = event.get("provider") or provider
        response_id = event.get("id") or response_id
        created = event.get("created") or created
        if event.get("usage"):
            usage = event["usage"]
        choices = event.get("choices") or []
        if not choices:
            continue
        assert len(choices) == 1, f"expected single choice, got {len(choices)}"
        choice = choices[0]
        if choice.get("error"):
            raise OpenRouterError(f"in-stream choice error: {choice['error']}", status_code=None)
        delta = choice.get("delta") or {}
        if delta.get("reasoning"):
            reasoning_parts.append(delta["reasoning"])
        if delta.get("content"):
            content_parts.append(delta["content"])
        finish_reason = choice.get("finish_reason") or finish_reason
        native_finish_reason = choice.get("native_finish_reason") or native_finish_reason
    if finish_reason is None or finish_reason == "error":
        raise OpenRouterError(f"stream ended without a valid finish_reason ({finish_reason=})", status_code=None)
    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    if not content and not reasoning:
        raise OpenRouterError("stream produced no content and no reasoning", status_code=None)
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "provider": provider,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "native_finish_reason": native_finish_reason,
                "message": {"role": "assistant", "content": content, "reasoning": reasoning},
            }
        ],
        "usage": usage,
        "_assembled_from_stream": True,
    }


def _served_model_matches(served: str, requested: str) -> bool:
    """True iff the served id is the requested id or a hand-verified alias of it.

    OpenRouter reports some models under a canonical date-versioned slug (e.g.
    z-ai/glm-5.2 -> z-ai/glm-5.2-20260616). Only EXPLICIT aliases from the registry are
    accepted — no pattern matching, because e.g. deepseek-r1-0528 is a genuinely different
    model than deepseek-r1 and must fail.
    """
    if served == requested:
        return True
    from lib.frontier import FRONTIER_BY_ID  # lazy: avoids an import cycle
    from lib.sweep import OPENWEIGHT_BY_ID  # lazy: lib.sweep imports other lib modules

    m = OPENWEIGHT_BY_ID.get(requested) or FRONTIER_BY_ID.get(requested)
    return m is not None and served in m.served_id_aliases


async def _stream_once(api_kwargs: dict[str, Any]) -> dict[str, Any]:
    client = get_client()
    payload = {**api_kwargs, "stream": True}
    events: list[dict[str, Any]] = []
    async with client.stream("POST", OPENROUTER_URL, json=payload) as response:
        if response.status_code != 200:
            body = (await response.aread()).decode(errors="replace")
            raise OpenRouterError(f"HTTP {response.status_code}: {body[:2000]}", status_code=response.status_code)
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue  # SSE comments (keep-alives) and blank lines
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            events.append(json.loads(data))
    result = assemble_stream_events(events)
    # Integrity guards: no silent model or provider substitution. OpenRouter reports some
    # models under their canonical date-versioned slug (e.g. z-ai/glm-5.2 is served as
    # z-ai/glm-5.2-20260616) — the SAME model, so exactly a -YYYYMMDD suffix is allowed;
    # anything else is a substitution and fails. The served id is persisted per row.
    assert _served_model_matches(result["model"], api_kwargs["model"]), (result["model"], api_kwargs["model"])
    provider_cfg = api_kwargs.get("provider", {})
    order = provider_cfg.get("order", [])
    if len(order) == 1 and provider_cfg.get("allow_fallbacks") is False:
        served = (result["provider"] or "").lower()
        assert served == order[0].lower(), f"provider mismatch: served {result['provider']!r}, pinned {order[0]!r}"
    assert result.get("usage"), "no usage in response"
    return result


async def _call_with_retries(api_kwargs: dict[str, Any]) -> dict[str, Any]:
    delay = INITIAL_RETRY_DELAY_S
    for attempt in range(MAX_RETRIES + 1):
        try:
            # httpx's timeout categories have no overall wall-clock deadline, so a stream
            # that keeps sending keep-alive comments could hang forever — enforce one here.
            return await asyncio.wait_for(_stream_once(api_kwargs), timeout=TOTAL_TIMEOUT_S)
        except Exception as e:
            if not _is_transient(e) or attempt == MAX_RETRIES:
                raise
            sleep_s = delay * (0.5 + random.random())
            if attempt % 10 == 9:
                print(f"  retry {attempt + 1} after {type(e).__name__}: {e} (sleeping {sleep_s:.1f}s)")
            await asyncio.sleep(sleep_s)
            delay = min(delay * 2, MAX_RETRY_DELAY_S)
    raise AssertionError("unreachable")


def openrouter_response_texts(response: dict[str, Any]) -> tuple[str, str]:
    """-> (raw reasoning text, visible response text) from an OpenRouter chat-completion dict."""
    message = response["choices"][0]["message"]
    return message.get("reasoning") or "", message.get("content") or ""


def parse_openrouter_response(response: dict[str, Any]) -> dict[str, Any]:
    """Extract analysis fields (mirrors lib.tier1.parse_response for Anthropic responses)."""
    from lib.tier1 import extract_answer

    reasoning_text, visible_text = openrouter_response_texts(response)
    answer, answer_source = extract_answer(visible_text)
    # Some open models (DeepSeek V3.2 seen doing this) emit the whole "<thinking>...<mc>X</mc>"
    # block into the REASONING channel and leave the visible content empty. When visible gives
    # nothing, fall back to extracting from the reasoning trace (marked, so it's auditable).
    if answer is None and len(visible_text.strip()) < 5 and reasoning_text.strip():
        r_answer, r_source = extract_answer(reasoning_text)
        if r_answer is not None:
            answer, answer_source = r_answer, f"reasoning_{r_source}"
    choice = response["choices"][0]
    return {
        "answer": answer,
        "answer_source": answer_source,
        "stop_reason": choice["finish_reason"],
        "native_finish_reason": choice.get("native_finish_reason"),
        "truncated": choice["finish_reason"] == "length",
        "usage": response["usage"],
        "reasoning_chars": len(reasoning_text),
        "visible_text_chars": len(visible_text),
        "served_model": response["model"],
        "served_provider": response["provider"],
    }


async def call_openrouter_cached(
    api_kwargs: dict[str, Any],
    sample_idx: int = 0,
    cost_tracker: CostTracker | None = None,
    assert_cached: bool = False,
) -> dict[str, Any]:
    """Cached call. api_kwargs must contain ALL request parameters (they form the cache key)."""
    cache_key = {**api_kwargs, "sample_idx": sample_idx, "this_call_uuid": OPENROUTER_CALL_UUID}

    async def compute() -> dict[str, Any]:
        async with _get_semaphore(api_kwargs["model"]):
            response = await _call_with_retries(api_kwargs)
        if cost_tracker is not None:
            usage = response["usage"]
            cost_tracker.add_llm_api_cost(
                api_kwargs["model"],
                input_tokens=usage["prompt_tokens"],
                output_tokens=usage["completion_tokens"],
            )
        return response

    result, _ = await get_cache().aget_or_compute_set(cache_key, compute, assert_cached=assert_cached)
    return result
