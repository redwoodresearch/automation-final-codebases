"""Cross-model sweep registry: which models are collected, their release order, and helpers
to build per-model FaithSpecs and locate their result files.

A model is "ready" for a given analysis stage if the required files exist; callers filter on
that so the same analysis runs incrementally as collections/judging land.
"""

from pathlib import Path

import attrs

from lib.faithfulness import TIER1_PAPER_TYPES, TIER2_TYPES, FaithSpec

RESULTS = Path("results")


@attrs.frozen
class SweepModel:
    short: str          # model_short, e.g. "haiku-4-5"
    display: str        # e.g. "Haiku 4.5"
    release_date: str   # ISO public release date, used for trend ordering
    full_id: str
    pool: str = "standard"      # Opus 4.1 / Sonnet 4.5 differ (see SPECIAL_MODELS)
    caveats: tuple[str, ...] = ()


# Ordered by public release date. Opus 4.1 and Sonnet 4.5 were collected first on different
# question pools (standard 500-q / full 2,994-q); handled via file_stems() below.
SWEEP_MODELS = [
    SweepModel("opus-4-1", "Opus 4.1", "2025-08-05", "claude-opus-4-1-20250805", pool="standard",
               caveats=("500q only", "thin flip baseline (n≈24/20) → corrected numbers preliminary")),
    SweepModel("sonnet-4-5", "Sonnet 4.5", "2025-09-29", "claude-sonnet-4-5-20250929", pool="full"),
    SweepModel("haiku-4-5", "Haiku 4.5", "2025-10-15", "claude-haiku-4-5-20251001"),
    SweepModel("opus-4-5", "Opus 4.5", "2025-11-24", "claude-opus-4-5-20251101"),
    SweepModel("opus-4-6", "Opus 4.6", "2026-02-04", "claude-opus-4-6"),
    SweepModel("sonnet-4-6", "Sonnet 4.6", "2026-02-17", "claude-sonnet-4-6"),
    SweepModel("opus-4-7", "Opus 4.7", "2026-04-14", "claude-opus-4-7",
               caveats=("adaptive thinking often skips CoT entirely",)),
    SweepModel("opus-4-8", "Opus 4.8", "2026-05-28", "claude-opus-4-8",
               caveats=("judge model (self-judging checked: no self-favoring bias)",)),
    SweepModel("fable-5", "Fable 5", "2026-06-07", "claude-fable-5",
               caveats=("~18% Tier-1 refusals via bio/cyber classifier → smaller cells, question-mix selection",)),
    SweepModel("sonnet-5", "Sonnet 5", "2026-06-29", "claude-sonnet-5"),
]

BY_SHORT = {m.short: m for m in SWEEP_MODELS}


@attrs.frozen
class OpenWeightModel:
    """An open-weight reasoning model run via OpenRouter.

    These are a COMPARISON GROUP, not part of the Anthropic release-date trend line.
    Prices for the pinned provider live in pricing/llm.json (keyed by full_id).
    """

    short: str
    display: str
    lab: str
    release_date: str  # ISO
    full_id: str  # OpenRouter model id
    provider: str  # pinned OpenRouter provider (requests set allow_fallbacks=False)
    quantization: str  # the pinned provider's listed quantization at pin time
    max_tokens: int
    pool: str = "standard"
    # Exact date-versioned slugs OpenRouter reports for this model (hand-verified; each
    # matches the model's release date). Anything else fails the served-model assert.
    served_id_aliases: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()


OPENWEIGHT_MODELS = [
    OpenWeightModel(
        "deepseek-r1", "DeepSeek R1", "DeepSeek", "2025-01-20", "deepseek/deepseek-r1",
        provider="Novita", quantization="fp8 (R1's native training precision)", max_tokens=16_000,
        caveats=("anchor model (Chen et al. published its numbers)",
                 "16k completion cap is Novita's max; truncation tracked per cell"),
    ),
    OpenWeightModel(
        "qwen3-235b-think", "Qwen3-235B-Thinking", "Alibaba (Qwen)", "2025-07-25",
        "qwen/qwen3-235b-a22b-thinking-2507",
        provider="DeepInfra", quantization="fp8 (release precision bf16)", max_tokens=32_768,
    ),
    OpenWeightModel(
        "deepseek-v3.2", "DeepSeek V3.2", "DeepSeek", "2025-12-01", "deepseek/deepseek-v3.2",
        provider="Novita", quantization="fp8 (native)", max_tokens=32_768,
        served_id_aliases=("deepseek/deepseek-v3.2-20251201",),
        caveats=("hybrid thinking model; reasoning enabled via request param",),
    ),
    OpenWeightModel(
        "kimi-k2.5", "Kimi K2.5", "Moonshot AI", "2026-01-27", "moonshotai/kimi-k2.5",
        provider="SiliconFlow", quantization="int4 (native: quantization-aware trained)", max_tokens=32_768,
        served_id_aliases=("moonshotai/kimi-k2.5-0127",),
    ),
    OpenWeightModel(
        "glm-5.2", "GLM-5.2", "Z.ai", "2026-06-13", "z-ai/glm-5.2",
        provider="Novita", quantization="fp8 (Z.ai's own endpoint also serves fp8)", max_tokens=32_768,
        served_id_aliases=("z-ai/glm-5.2-20260616",),
    ),
    OpenWeightModel(
        "gpt-oss-120b", "gpt-oss-120b", "OpenAI", "2025-08-05", "openai/gpt-oss-120b",
        provider="DeepInfra", quantization="bf16 (release precision is MXFP4 for MoE weights)", max_tokens=32_768,
        caveats=("the only US-lab open-weight reasoner in the slate",),
    ),
]

OPENWEIGHT_BY_SHORT = {m.short: m for m in OPENWEIGHT_MODELS}
OPENWEIGHT_BY_ID = {m.full_id: m for m in OPENWEIGHT_MODELS}


def build_openweight_kwargs(m: OpenWeightModel, prompt: list[dict[str, str]], temperature: float) -> dict:
    """OpenRouter chat-completion request kwargs; ALL fields enter the cache key.

    `reasoning: {"enabled": True}` requests the model's raw chain of thought (returned in
    message.reasoning, separate from the visible message.content).
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


def t0_variant(path: str) -> str:
    """Result path for the temp-0 anchor run: tier1_deepseek-r1_standard.jsonl -> ..._t0_standard.jsonl."""
    stem, pool_ext = path.rsplit("_", 1)
    return f"{stem}_t0_{pool_ext}"


def file_stems(m: SweepModel) -> dict[str, str]:
    """Result-file paths for a model (its MMLU transcripts + judge verdicts)."""
    p = m.pool
    return {
        "tier1": str(RESULTS / f"tier1_{m.short}_{p}.jsonl"),
        "tier2": str(RESULTS / f"tier2_{m.short}_{p}.jsonl"),
        "judge_tier1": str(RESULTS / f"judge_tier1_{m.short}_{p}.jsonl"),
        "judge_tier2": str(RESULTS / f"judge_tier2_{m.short}_{p}.jsonl"),
    }


def collected(m: SweepModel) -> bool:
    f = file_stems(m)
    return Path(f["tier1"]).exists() and Path(f["tier2"]).exists()


def judged(m: SweepModel) -> bool:
    f = file_stems(m)
    return Path(f["judge_tier1"]).exists() and Path(f["judge_tier2"]).exists()


def faith_specs(m: SweepModel) -> list[FaithSpec]:
    f = file_stems(m)
    return [
        FaithSpec(f"{m.display} (Tier-1 released)", f["tier1"], f["judge_tier1"],
                  conditions=tuple(TIER1_PAPER_TYPES)),
        FaithSpec(f"{m.display} (Tier-2 reconstructed)", f["tier2"], f["judge_tier2"],
                  baseline_path=f["tier1"], conditions=tuple(TIER2_TYPES)),
    ]
