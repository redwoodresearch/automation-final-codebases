"""Shared helpers for the figure scripts: model ordering, committed-results loading, and the
percentage transforms the plots use. All of the project's own numbers come from the committed
results/*.json tables (written by the scripts/analyze_*.py scripts); figure scripts hardcode
only external reference values (Chen et al. 2025, Young 2026, Chua & Evans 2025), each with a
source comment where it is defined.
"""

import json
from pathlib import Path

from lib.frontier import FRONTIER_MODELS
from lib.sweep import OPENWEIGHT_MODELS, SWEEP_MODELS

RESULTS = Path(__file__).parent.parent / "results"
FIGURES = Path(__file__).parent.parent / "figures"

# Plot order within each group (release order).
CLAUDE_ORDER = [m.short for m in SWEEP_MODELS]
OPENWEIGHT_ORDER = [m.short for m in sorted(OPENWEIGHT_MODELS, key=lambda m: m.release_date)]
GPT_ORDER = [m.short for m in sorted(FRONTIER_MODELS, key=lambda m: m.release_date) if m.lab == "OpenAI"]
GEMINI_ORDER = [m.short for m in sorted(FRONTIER_MODELS, key=lambda m: m.release_date) if m.lab == "Google"]

DISPLAY = {m.short: m.display for m in SWEEP_MODELS + OPENWEIGHT_MODELS + FRONTIER_MODELS}

# Shared bar colors.
C_HINT, C_OTHER, C_NOCHANGE = "#84c184", "#cd807b", "#a6a6a6"
# lighter variants for externally reported (Chen et al.) bars
C_HINT_L, C_OTHER_L, C_NOCHANGE_L = "#a9d4a9", "#dfaea9", "#c6c6c6"
F_CORRECT, F_WRONG = "#d98cc0", "#b0468c"


def load_results_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def following_pct(cell: dict) -> tuple[float, float, float]:
    """counts dict -> (change_to_hint %, change_to_non_hint %, no_change %)."""
    n = cell["n_eligible"]
    return (100 * cell["change_to_hint"] / n, 100 * cell["change_to_non_hint"] / n, 100 * cell["no_change"] / n)


def following_avg_pct(rec: dict, direction: str) -> tuple[float, float, float]:
    """Equal-weight MMLU+GPQA average of the following decomposition percentages."""
    mm = following_pct(rec["mmlu"][direction])
    gp = following_pct(rec["gpqa"][direction])
    return tuple((a + b) / 2 for a, b in zip(mm, gp))
