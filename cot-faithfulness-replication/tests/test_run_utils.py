"""Tests for lib.run_utils helpers."""

from lib.openrouter import parse_openrouter_response
from lib.run_utils import content_filter_sentinel, model_short, openrouter_content_block_sentinel
from lib.tier1 import parse_response


def test_model_short_strips_date():
    assert model_short("claude-opus-4-1-20250805") == "opus-4-1"
    assert model_short("claude-opus-4-8") == "opus-4-8"
    assert model_short("claude-sonnet-5") == "sonnet-5"
    assert model_short("claude-sonnet-4-5-20250929") == "sonnet-4-5"


def test_content_filter_sentinel_matches_block_and_parses_invalid():
    e = Exception("{'type': 'error', 'error': {'type': 'invalid_request_error', "
                  "'message': 'Output blocked by content filtering policy'}}")
    s = content_filter_sentinel("claude-sonnet-5", e)
    assert s is not None
    assert s["stop_reason"] == "content_filtered"
    parsed = parse_response(s)
    # a blocked output must count as an invalid (no-answer) row, not a spurious letter
    assert parsed["answer"] is None
    assert parsed["stop_reason"] == "content_filtered"
    assert parsed["thinking_chars"] == 0 and parsed["visible_text_chars"] == 0


def test_content_filter_sentinel_ignores_other_errors():
    # a genuine bad request (e.g. a real bug) must NOT be swallowed
    assert content_filter_sentinel("m", Exception("max_tokens too large")) is None
    assert content_filter_sentinel("m", Exception("overloaded_error")) is None


def test_openrouter_content_block_sentinel_matches_gemini_block_and_parses_invalid():
    # the exact Gemini refusal seen in the frontier collection (a 400 PROHIBITED_CONTENT block)
    e = Exception("in-stream error: {'code': 400, 'message': 'Gemini blocked the request: "
                  "PROHIBITED_CONTENT', 'metadata': {'error_type': 'invalid_request'}}")
    s = openrouter_content_block_sentinel("google/gemini-3.1-pro-preview", e)
    assert s is not None
    assert s["model"] == "google/gemini-3.1-pro-preview" and s["_content_filtered"]
    parsed = parse_openrouter_response(s)
    # a blocked output must count as an invalid (no-answer) row, not a spurious letter or a truncation
    assert parsed["answer"] is None
    assert parsed["truncated"] is False
    assert parsed["served_model"] == "google/gemini-3.1-pro-preview"
    assert parsed["reasoning_chars"] == 0 and parsed["visible_text_chars"] == 0


def test_openrouter_content_block_sentinel_matches_openai_safety_block():
    # the exact GPT-5.6 refusal seen on some GPQA biology prompts (a deterministic 400 safety block)
    e = Exception("in-stream error: {'code': 400, 'message': \"Invalid prompt: we've limited access "
                  "to this content for safety reasons.\", 'metadata': {'error_type': 'invalid_request'}}")
    s = openrouter_content_block_sentinel("openai/gpt-5.6-terra", e)
    assert s is not None and s["_content_filtered"]
    parsed = parse_openrouter_response(s)
    assert parsed["answer"] is None and parsed["truncated"] is False
    assert parsed["served_model"] == "openai/gpt-5.6-terra"


def test_openrouter_content_block_sentinel_ignores_other_errors():
    # transient/real errors must NOT be swallowed as content blocks
    assert openrouter_content_block_sentinel("m", Exception("HTTP 429: rate limited")) is None
    assert openrouter_content_block_sentinel("m", Exception("stream ended without a valid finish_reason")) is None
