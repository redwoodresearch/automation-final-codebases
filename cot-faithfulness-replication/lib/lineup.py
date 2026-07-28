"""The 30-model lineup used by the post's figures, with each model's result-file locations.

Three groups, plotted in this order:
  anthropic  — 10 Claude models (lib.sweep.SWEEP_MODELS)
  openweight — 6 open-weight reasoners via OpenRouter (lib.sweep.OPENWEIGHT_MODELS)
  frontier   — 14 closed GPT/Gemini models via OpenRouter (lib.frontier.FRONTIER_MODELS)

MMLU transcripts live at per-group stems (standard 500-q pool for most; Sonnet 4.5 on the
full 2,994-q pool; frontier models on the first 250 standard-pool questions, "std250").
GPQA grids live at gpqa_tier{1,2}_{tag}.jsonl, where the open-weight tag carries a _t1
temperature suffix.
"""

import attrs

from lib.frontier import FRONTIER_MODELS
from lib.frontier import file_stems as frontier_stems
from lib.sweep import OPENWEIGHT_MODELS, SWEEP_MODELS, file_stems


@attrs.frozen
class LineupModel:
    short: str
    display: str
    group: str  # "anthropic" | "openweight" | "frontier"
    mmlu_stems: dict  # tier1/tier2/judge_tier1/judge_tier2 result paths
    gpqa_tag: str  # tag for lib.gpqa_analysis.grid_files


def lineup() -> list[LineupModel]:
    models = [LineupModel(m.short, m.display, "anthropic", file_stems(m), m.short) for m in SWEEP_MODELS]
    models += [
        LineupModel(m.short, m.display, "openweight", file_stems(m), f"{m.short}_t1") for m in OPENWEIGHT_MODELS
    ]
    models += [
        LineupModel(m.short, m.display, "frontier", frontier_stems(m, "std250"), m.short) for m in FRONTIER_MODELS
    ]
    return models


BY_SHORT = {m.short: m for m in lineup()}
