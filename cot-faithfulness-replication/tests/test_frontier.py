"""Tests for the closed-frontier comparison group (GPT/Gemini via OpenRouter)."""

import json
from pathlib import Path

from lib.frontier import (
    FRONTIER_BY_ID,
    FRONTIER_BY_SHORT,
    FRONTIER_MODELS,
    build_frontier_kwargs,
    faith_specs,
    file_stems,
)
from lib.openrouter import (
    _served_model_matches,
    assemble_stream_events,
    openrouter_response_texts,
    parse_openrouter_response,
)
from lib.sweep import OPENWEIGHT_BY_ID
from lib.tier1 import response_texts


def test_frontier_registry_consistency():
    pricing = json.loads((Path(__file__).parent.parent / "pricing" / "llm.json").read_text())
    shorts = [m.short for m in FRONTIER_MODELS]
    assert len(set(shorts)) == len(shorts)
    assert len(set(m.full_id for m in FRONTIER_MODELS)) == len(FRONTIER_MODELS)  # unique ids too
    labs = {m.lab for m in FRONTIER_MODELS}
    assert labs == {"OpenAI", "Google"}, "the comparison group is exactly the two other frontier labs"
    for m in FRONTIER_MODELS:
        assert "/" in m.full_id, m.full_id  # OpenRouter ids are namespaced
        assert m.full_id in pricing, f"{m.full_id} missing from pricing/llm.json"
        assert m.reasoning_is_summary, "frontier reasoning is a vendor summary (carries the caveat)"
        # No collision with the open-weight registry (run_tier1 must branch frontier BEFORE openweight).
        assert m.full_id not in OPENWEIGHT_BY_ID, f"{m.full_id} must not also be an open-weight model"
        assert m.provider in ("OpenAI", "Google", "Google AI Studio"), (m.short, m.provider)


def test_frontier_membership():
    """All 14 closed-frontier models used in the post are registered."""
    expected = {
        "openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5-nano", "openai/gpt-5.1", "openai/gpt-5.2",
        "openai/gpt-5.4", "openai/gpt-5.5", "openai/gpt-5.6-luna", "openai/gpt-5.6-sol", "openai/gpt-5.6-terra",
        "google/gemini-3.1-pro-preview", "google/gemini-3.1-flash-lite-preview",
        "google/gemini-3.5-flash", "google/gemini-3.6-flash",
    }
    assert expected <= set(FRONTIER_BY_ID), expected - set(FRONTIER_BY_ID)
    # GPT models pin "OpenAI"; the only "Google AI Studio" pin is flash-lite (no Vertex endpoint on OpenRouter).
    assert all(FRONTIER_BY_ID[i].provider == "OpenAI" for i in expected if i.startswith("openai/"))
    assert FRONTIER_BY_ID["google/gemini-3.1-flash-lite-preview"].provider == "Google AI Studio"


def test_build_frontier_kwargs():
    m = FRONTIER_BY_SHORT["gpt-5.5"]
    kwargs = build_frontier_kwargs(m, [{"role": "human", "content": "q"}], temperature=1.0)
    assert kwargs["model"] == "openai/gpt-5.5"
    assert kwargs["messages"] == [{"role": "user", "content": "q"}]  # human -> user role map
    assert kwargs["reasoning"] == {"enabled": True}  # request the vendor reasoning summary
    assert kwargs["provider"] == {"order": ["OpenAI"], "allow_fallbacks": False}  # no silent substitution
    assert kwargs["usage"] == {"include": True}
    assert kwargs["temperature"] == 1.0
    assert kwargs["max_tokens"] == m.max_tokens


def test_frontier_served_model_matches():
    # Both current models report served id == requested id (no alias needed); the assert must pass.
    assert _served_model_matches("openai/gpt-5.5", "openai/gpt-5.5")
    assert _served_model_matches("google/gemini-3.1-pro-preview", "google/gemini-3.1-pro-preview")
    # A different served model must still fail even for a registered frontier id.
    assert not _served_model_matches("openai/gpt-5", "openai/gpt-5.5")


def test_frontier_response_parsing_summary_and_visible():
    """A frontier response carries the vendor reasoning SUMMARY in `reasoning` and the visible
    `<thinking>/<mc>` scaffold output in `content`; both must be recoverable as (summary, visible)."""
    events = [
        {"id": "g", "model": "openai/gpt-5.5", "provider": "OpenAI",
         "choices": [{"index": 0, "delta": {"reasoning": "**Summary of my reasoning**"}, "finish_reason": None}]},
        {"id": "g", "model": "openai/gpt-5.5", "provider": "OpenAI",
         "choices": [{"index": 0, "delta": {"content": "<thinking>Paris is the capital.</thinking>\n<mc>C</mc>"},
                      "finish_reason": None}]},
        {"id": "g", "model": "openai/gpt-5.5", "provider": "OpenAI",
         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop", "native_finish_reason": "completed"}],
         "usage": {"prompt_tokens": 82, "completion_tokens": 85, "cost": 0.00296}},
    ]
    response = assemble_stream_events(events)
    summary, visible = openrouter_response_texts(response)
    assert summary == "**Summary of my reasoning**"
    assert visible.endswith("<mc>C</mc>")
    assert response_texts(response) == (summary, visible)  # shared judge/inspection dispatcher
    parsed = parse_openrouter_response(response)
    assert parsed["answer"] == "C"
    assert parsed["answer_source"] == "mc_tag"
    assert not parsed["truncated"]
    assert parsed["served_model"] == "openai/gpt-5.5"
    assert parsed["served_provider"] == "OpenAI"


def test_frontier_gpqa_prompt_dispatch():
    """The GPQA extension sends frontier models multi-turn prompts (posthoc has an assistant turn);
    build_frontier_kwargs must preserve the assistant role and keep the reasoning-summary request."""
    import lib.gpqa as gpqa

    m = FRONTIER_BY_SHORT["gpt-5-nano"]
    prompt = gpqa.gpqa_prompt("posthoc_False", 0)
    assert any(t["role"] == "assistant" for t in prompt)  # planted prior-turn answer
    kwargs = build_frontier_kwargs(m, prompt, temperature=1.0)
    assert kwargs["model"] == "openai/gpt-5-nano"
    assert {t["role"] for t in kwargs["messages"]} == {"user", "assistant"}
    assert kwargs["reasoning"] == {"enabled": True}
    assert kwargs["provider"] == {"order": ["OpenAI"], "allow_fallbacks": False}
    # A frontier id must resolve as frontier, never as open-weight (run_gpqa branches frontier first).
    assert m.full_id in FRONTIER_BY_ID and m.full_id not in OPENWEIGHT_BY_ID


def test_frontier_file_stems_and_specs():
    m = FRONTIER_BY_SHORT["gemini-3.1-pro"]
    stems = file_stems(m, "pilot")
    assert stems["tier1"].endswith("tier1_gemini-3.1-pro_pilot.jsonl")
    assert stems["judge_tier2"].endswith("judge_tier2_gemini-3.1-pro_pilot.jsonl")
    specs = faith_specs(m, "pilot")
    assert len(specs) == 2
    assert specs[0].conditions == ("suggestion", "posthoc", "fewshot_symbol")
    assert specs[1].conditions == ("metadata", "grader_hacking", "unethical_information")
    assert specs[1].baseline_path == stems["tier1"]
