"""Shared loading/spec helpers for the GPQA analyses (following + faithfulness).

The GPQA collection mirrors the MMLU file layout (tier1 file holds the baselines + released-template
types; tier2 file holds the reconstruction types and reuses tier1's unhinted_plain as a_u), so the
existing build_pairs / unhinted_condition_for / FaithSpec machinery reuses unchanged.
"""

from pathlib import Path

from lib.analysis import build_pairs, cell_from_pairs, load_results, results_file_exists
from lib.faithfulness import FaithSpec
from lib.gpqa import TIER1_TYPES, TIER2_TYPES

RESULTS = Path("results")


def grid_files(tag: str) -> dict[str, str]:
    return {
        "tier1": str(RESULTS / f"gpqa_tier1_{tag}.jsonl"),
        "tier2": str(RESULTS / f"gpqa_tier2_{tag}.jsonl"),
        "judge_tier1": str(RESULTS / f"judge_gpqa_tier1_{tag}.jsonl"),
        "judge_tier2": str(RESULTS / f"judge_gpqa_tier2_{tag}.jsonl"),
    }


def grid_collected(tag: str) -> bool:
    f = grid_files(tag)
    return results_file_exists(f["tier1"]) and results_file_exists(f["tier2"])


def load_grid_rows(tag: str) -> dict:
    """Merged {(condition, qidx): row} across the tier1 + tier2 files (baselines live in tier1)."""
    f = grid_files(tag)
    rows = dict(load_results(Path(f["tier1"])))
    for key, row in load_results(Path(f["tier2"])).items():
        assert key not in rows, f"duplicate {key} across tier1/tier2"
        rows[key] = row
    return rows


def usage_cells(tag: str) -> dict[str, object]:
    """{condition_name: HintUsageCell} for all six hint types x both arms."""
    rows = load_grid_rows(tag)
    out = {}
    for ht in TIER1_TYPES + TIER2_TYPES:
        for arm in ("True", "False"):
            cond = f"{ht}_{arm}"
            pairs = build_pairs(rows, cond)
            if pairs:
                out[cond] = cell_from_pairs(pairs)
    return out


def faith_specs(tag: str) -> list[FaithSpec]:
    f = grid_files(tag)
    return [
        FaithSpec(f"{tag} (GPQA released-template-faithful)", f["tier1"], f["judge_tier1"],
                  conditions=tuple(TIER1_TYPES)),
        FaithSpec(f"{tag} (GPQA full-reconstruction)", f["tier2"], f["judge_tier2"],
                  baseline_path=f["tier1"], conditions=tuple(TIER2_TYPES)),
    ]
