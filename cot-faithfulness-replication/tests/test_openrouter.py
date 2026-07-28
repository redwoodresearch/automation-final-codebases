import json
from pathlib import Path

import pytest

from lib.openrouter import (
    OpenRouterError,
    assemble_stream_events,
    openrouter_response_texts,
    parse_openrouter_response,
)
from lib.sweep import OPENWEIGHT_BY_SHORT, OPENWEIGHT_MODELS, build_openweight_kwargs, t0_variant
from lib.tier1 import response_texts


def _events(deltas, finish_reason="stop", usage=None, model="deepseek/deepseek-r1", provider="Novita"):
    events = []
    for delta in deltas:
        events.append(
            {
                "id": "gen-1",
                "model": model,
                "provider": provider,
                "created": 1700000000,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
        )
    events.append(
        {
            "id": "gen-1",
            "model": model,
            "provider": provider,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason, "native_finish_reason": finish_reason}],
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 20, "cost": 0.0001},
        }
    )
    return events


def test_assemble_stream_basic():
    events = _events([{"reasoning": "let me think"}, {"reasoning": " more"}, {"content": "<mc>A</mc>"}])
    response = assemble_stream_events(events)
    assert response["model"] == "deepseek/deepseek-r1"
    assert response["provider"] == "Novita"
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["choices"][0]["message"]["reasoning"] == "let me think more"
    assert response["choices"][0]["message"]["content"] == "<mc>A</mc>"
    assert response["usage"]["completion_tokens"] == 20
    assert openrouter_response_texts(response) == ("let me think more", "<mc>A</mc>")
    # the shared dispatcher used by judge/inspection scripts handles this shape too
    assert response_texts(response) == ("let me think more", "<mc>A</mc>")


def test_assemble_stream_error_event():
    events = [{"error": {"code": 502, "message": "provider down"}}]
    with pytest.raises(OpenRouterError):
        assemble_stream_events(events)


def test_assemble_stream_no_finish_reason():
    events = [{"model": "m", "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]}]
    with pytest.raises(OpenRouterError):
        assemble_stream_events(events)


def test_assemble_stream_empty_output():
    events = _events([])
    with pytest.raises(OpenRouterError):
        assemble_stream_events(events)


def test_parse_openrouter_response():
    events = _events([{"reasoning": "hmm"}, {"content": "final answer <mc>C</mc>"}], finish_reason="stop")
    parsed = parse_openrouter_response(assemble_stream_events(events))
    assert parsed["answer"] == "C"
    assert parsed["answer_source"] == "mc_tag"
    assert parsed["stop_reason"] == "stop"
    assert not parsed["truncated"]
    assert parsed["served_model"] == "deepseek/deepseek-r1"
    assert parsed["served_provider"] == "Novita"
    assert parsed["reasoning_chars"] == 3


def test_parse_truncated():
    events = _events([{"reasoning": "endless thinking"}], finish_reason="length")
    parsed = parse_openrouter_response(assemble_stream_events(events))
    assert parsed["truncated"]
    assert parsed["answer"] is None


def test_openweight_registry_consistency():
    pricing = json.loads((Path(__file__).parent.parent / "pricing" / "llm.json").read_text())
    shorts = [m.short for m in OPENWEIGHT_MODELS]
    assert len(set(shorts)) == len(shorts)
    for m in OPENWEIGHT_MODELS:
        assert "/" in m.full_id, m.full_id  # OpenRouter ids are namespaced; run scripts branch on this
        assert m.full_id in pricing, f"{m.full_id} missing from pricing/llm.json"
        kwargs = build_openweight_kwargs(m, [{"role": "human", "content": "q"}], temperature=1.0)
        assert kwargs["provider"] == {"order": [m.provider], "allow_fallbacks": False}
        assert kwargs["reasoning"] == {"enabled": True}
        assert kwargs["messages"] == [{"role": "user", "content": "q"}]


def test_t0_variant():
    assert t0_variant("results/tier1_deepseek-r1_standard.jsonl") == "results/tier1_deepseek-r1_t0_standard.jsonl"
    assert t0_variant("results/tier2_deepseek-r1_standard.jsonl") == "results/tier2_deepseek-r1_t0_standard.jsonl"


def test_served_model_matches():
    from lib.openrouter import _served_model_matches

    assert _served_model_matches("deepseek/deepseek-r1", "deepseek/deepseek-r1")
    # hand-verified aliases from the registry
    assert _served_model_matches("z-ai/glm-5.2-20260616", "z-ai/glm-5.2")
    assert _served_model_matches("moonshotai/kimi-k2.5-0127", "moonshotai/kimi-k2.5")
    assert _served_model_matches("deepseek/deepseek-v3.2-20251201", "deepseek/deepseek-v3.2")
    # non-aliases (incl. genuinely different models with date-like suffixes) must fail
    assert not _served_model_matches("z-ai/glm-5.1", "z-ai/glm-5.2")
    assert not _served_model_matches("deepseek/deepseek-r1-0528", "deepseek/deepseek-r1")
    assert not _served_model_matches("z-ai/glm-5.2-20990101", "z-ai/glm-5.2")
    assert not _served_model_matches("other/model-20260101", "z-ai/glm-5.2")
