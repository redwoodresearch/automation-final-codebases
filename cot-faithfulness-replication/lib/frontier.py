"""Closed-frontier comparison group: OpenAI (GPT) and Google (Gemini) reasoning models.

These are a SEPARATE comparison group — NOT part of the Anthropic release-date trend line and
NOT open-weight. They are reached through OpenRouter. OpenRouter passes through each vendor's
reasoning SUMMARY (vendor-controlled, NOT raw chain-of-thought) in message.reasoning, plus the
visible `<thinking>…</thinking><mc>X</mc>` scaffold output in message.content.

Faithfulness for these models is therefore judged over BOTH channels (vendor summary + visible
response) and carries the vendor-summary caveat — same handling as gen-4.7+/adaptive Claude:
the summary is vendor-controlled, so its absolute faithfulness level is unknown-sign, but a
hint verbalized in the summary IS reported (conservative for the monitorability point). Judge-free
following uses only the extracted answer and is unaffected by the summary caveat.
"""

import attrs

from lib.faithfulness import TIER1_PAPER_TYPES, TIER2_TYPES, FaithSpec
from lib.sweep import RESULTS


@attrs.frozen
class FrontierModel:
    """A closed-frontier reasoning model reached via OpenRouter."""

    short: str          # result-file stem, e.g. "gpt-5.5"
    display: str        # e.g. "GPT-5.5"
    lab: str            # "OpenAI" | "Google"
    release_date: str   # ISO
    full_id: str        # OpenRouter model id, e.g. "openai/gpt-5.5"
    provider: str       # pinned OpenRouter provider (requests set allow_fallbacks=False)
    max_tokens: int = 32_000
    # Date-versioned slugs OpenRouter may report for this model (hand-verified). Both current
    # models report served id == requested id, so this is a robustness safety net.
    served_id_aliases: tuple[str, ...] = ()
    # Every model here returns a vendor reasoning SUMMARY (not raw CoT) → the vendor-summary
    # caveat attaches to its faithfulness numbers. (Field kept explicit for auditability.)
    reasoning_is_summary: bool = True
    caveats: tuple[str, ...] = ()


# Shared caveats by lab (the vendor-summary + gateway story is identical within a lab).
_GPT_CAVEATS = (
    "reached via OpenRouter",
    "reasoning field is OpenAI's vendor SUMMARY, not raw CoT",
    "visible <thinking> is often terse — the model reasons mainly in the hidden channel; OpenAI "
    "withholds the reasoning summary on a large fraction of calls (~68% observed for GPT-5.5) "
    "→ measured faithfulness leans on the monitorability read, not the absolute level",
)
_GEMINI_CAVEATS = (
    "reached via OpenRouter (no Google key in env)",
    "reasoning field is Gemini's vendor THOUGHT SUMMARY, not raw CoT",
    "almost always returns a thought summary + a fuller visible <thinking> (unlike GPT)",
)

# The original closed-frontier pair (collected first), then the expanded group below.
FRONTIER_MODELS = [
    FrontierModel(
        "gpt-5.5", "GPT-5.5", "OpenAI", "2026-04-24", "openai/gpt-5.5",
        provider="OpenAI",
        caveats=("reached via OpenRouter",
                 "reasoning field is OpenAI's vendor SUMMARY, not raw CoT",
                 "visible <thinking> is often terse — the model reasons mainly in the hidden channel"),
    ),
    FrontierModel(
        "gemini-3.1-pro", "Gemini 3.1 Pro", "Google", "2026-02-19", "google/gemini-3.1-pro-preview",
        provider="Google",
        caveats=("reached via OpenRouter (no Google key in env)",
                 "reasoning field is Gemini's vendor THOUGHT SUMMARY, not raw CoT",
                 "latest Pro/flagship reasoning Gemini; newer dated Geminis are flash-tier only"),
    ),
    # --- The expanded group: 10 more OpenAI GPT + 4 more Google Gemini reasoning models. ---
    # All closed → they join the closed-frontier comparison group; NEVER merged into the
    # Anthropic generational trend line. release_date = OpenRouter `created` date.
    # GPT models: pinned provider "OpenAI" (available for all; allow_fallbacks=False).
    FrontierModel("gpt-5", "GPT-5", "OpenAI", "2025-08-07", "openai/gpt-5", provider="OpenAI", caveats=_GPT_CAVEATS),
    FrontierModel("gpt-5-mini", "GPT-5 Mini", "OpenAI", "2025-08-07", "openai/gpt-5-mini", provider="OpenAI", caveats=_GPT_CAVEATS),
    FrontierModel("gpt-5-nano", "GPT-5 Nano", "OpenAI", "2025-08-07", "openai/gpt-5-nano", provider="OpenAI", caveats=_GPT_CAVEATS),
    FrontierModel("gpt-5.1", "GPT-5.1", "OpenAI", "2025-11-13", "openai/gpt-5.1", provider="OpenAI", caveats=_GPT_CAVEATS),
    FrontierModel("gpt-5.2", "GPT-5.2", "OpenAI", "2025-12-10", "openai/gpt-5.2", provider="OpenAI", caveats=_GPT_CAVEATS),
    FrontierModel("gpt-5.4", "GPT-5.4", "OpenAI", "2026-03-05", "openai/gpt-5.4", provider="OpenAI", caveats=_GPT_CAVEATS),
    FrontierModel("gpt-5.6-luna", "GPT-5.6 Luna", "OpenAI", "2026-07-09", "openai/gpt-5.6-luna", provider="OpenAI", caveats=_GPT_CAVEATS),
    FrontierModel("gpt-5.6-sol", "GPT-5.6 Sol", "OpenAI", "2026-07-09", "openai/gpt-5.6-sol", provider="OpenAI", caveats=_GPT_CAVEATS),
    FrontierModel("gpt-5.6-terra", "GPT-5.6 Terra", "OpenAI", "2026-07-09", "openai/gpt-5.6-terra", provider="OpenAI", caveats=_GPT_CAVEATS),
    # Gemini flash models. flash-lite is served ONLY by "Google AI Studio" on OpenRouter (no plain
    # "Google"/Vertex endpoint), so it is pinned there; 3.5/3.6-flash keep the "Google" (Vertex) pin
    # matching the existing Gemini 3.1 Pro anchor (both endpoints available for them).
    FrontierModel("gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite", "Google", "2026-03-03",
                  "google/gemini-3.1-flash-lite-preview", provider="Google AI Studio", caveats=_GEMINI_CAVEATS),
    FrontierModel("gemini-3.5-flash", "Gemini 3.5 Flash", "Google", "2026-05-19",
                  "google/gemini-3.5-flash", provider="Google", caveats=_GEMINI_CAVEATS),
    FrontierModel("gemini-3.6-flash", "Gemini 3.6 Flash", "Google", "2026-07-21",
                  "google/gemini-3.6-flash", provider="Google", caveats=_GEMINI_CAVEATS),
]

FRONTIER_BY_SHORT = {m.short: m for m in FRONTIER_MODELS}
FRONTIER_BY_ID = {m.full_id: m for m in FRONTIER_MODELS}


def build_frontier_kwargs(m: FrontierModel, prompt: list[dict[str, str]], temperature: float) -> dict:
    """OpenRouter chat-completion request kwargs for a frontier model; ALL fields enter the cache key.

    `reasoning: {"enabled": True}` requests the vendor reasoning summary (returned in
    message.reasoning). The provider is pinned (allow_fallbacks=False) so there is no silent
    backend substitution. `usage: {"include": True}` returns OpenRouter's own USD cost per call.
    """
    from lib.tier1 import render_messages

    return {
        "model": m.full_id,
        "messages": render_messages(prompt),
        "max_tokens": m.max_tokens,
        "temperature": temperature,
        "reasoning": {"enabled": True},
        "provider": {"order": [m.provider], "allow_fallbacks": False},
        "usage": {"include": True},
    }


def file_stems(m: FrontierModel, pool: str) -> dict[str, str]:
    return {
        "tier1": str(RESULTS / f"tier1_{m.short}_{pool}.jsonl"),
        "tier2": str(RESULTS / f"tier2_{m.short}_{pool}.jsonl"),
        "judge_tier1": str(RESULTS / f"judge_tier1_{m.short}_{pool}.jsonl"),
        "judge_tier2": str(RESULTS / f"judge_tier2_{m.short}_{pool}.jsonl"),
    }


def faith_specs(m: FrontierModel, pool: str) -> list[FaithSpec]:
    f = file_stems(m, pool)
    return [
        FaithSpec(f"{m.display} (Tier-1 released)", f["tier1"], f["judge_tier1"],
                  conditions=tuple(TIER1_PAPER_TYPES)),
        FaithSpec(f"{m.display} (Tier-2 reconstructed)", f["tier2"], f["judge_tier2"],
                  baseline_path=f["tier1"], conditions=tuple(TIER2_TYPES)),
    ]
