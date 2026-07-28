"""Sonnet 4.5 per-hint-type detail (the model the Sonnet 4.5 system card is about).

Computes, per hint type and hint direction:
- following: the change-to-hint decomposition, per dataset (MMLU = the FULL released pool of
  2,994 questions; GPQA = all 198 GPQA-Diamond questions) plus the equal-weight percent average
- faithfulness: per-arm normalized faithfulness, equal-weight (MMLU full pool + GPQA)/2 average
Writes results/sonnet45_detail.json (committed); the Sonnet 4.5 figures read that file.

  python scripts/analyze_sonnet45.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.analysis import build_pairs, load_results, results_file_exists
from lib.faithfulness import TIER1_PAPER_TYPES, TIER2_TYPES, FaithSpec, per_arm_cells
from lib.gpqa_analysis import grid_files
from lib.metrics import make_cell
from lib.tables import write_table

ALL_TYPES = TIER1_PAPER_TYPES + TIER2_TYPES
OUT_PATH = Path("results/sonnet45_detail.json")
ARMS = {"correct": "True", "incorrect": "False"}

MMLU = {
    "tier1": "results/tier1_sonnet-4-5_full.jsonl",
    "tier2": "results/tier2_sonnet-4-5_full.jsonl",
    "judge_tier1": "results/judge_tier1_sonnet-4-5_full.jsonl",
    "judge_tier2": "results/judge_tier2_sonnet-4-5_full.jsonl",
}
GPQA = grid_files("sonnet-4-5")


def merged_rows(t1, t2):
    rows = dict(load_results(Path(t1)))
    for key, row in load_results(Path(t2)).items():
        rows.setdefault(key, row)
    return rows


def specs_for(stems):
    return [
        FaithSpec("tier1", stems["tier1"], stems["judge_tier1"], conditions=tuple(TIER1_PAPER_TYPES)),
        FaithSpec("tier2", stems["tier2"], stems["judge_tier2"], baseline_path=stems["tier1"],
                  conditions=tuple(TIER2_TYPES)),
    ]


def per_type_norm(specs):
    out = {}
    for spec in specs:
        cells = per_arm_cells(spec)
        for base in spec.conditions:
            out[base] = {}
            for direction, arm in ARMS.items():
                got = cells.get(f"{base}_{arm}")
                out[base][direction] = None if got is None else got[0].normalized_faithfulness
    return out


def main() -> None:
    # This analyzer produces a single-model detail table; unlike the sweep analyzers there is
    # no meaningful partial output, so it always requires all of its inputs.
    argparse.ArgumentParser().parse_args()

    missing = [p for p in list(MMLU.values()) + list(GPQA.values()) if not results_file_exists(p)]
    if missing:
        from lib.tables import IncompleteTable
        raise IncompleteTable(
            f"missing input(s): {', '.join(missing)}\n"
            "  Fetch the raw transcripts (python data/download_transcripts.py) or regenerate them\n"
            f"  (see ./reproduce.sh full). The committed {OUT_PATH.name} was NOT overwritten."
        )

    def following_counts(rows):
        out = {}
        for ht in ALL_TYPES:
            out[ht] = {}
            for direction, arm in ARMS.items():
                trips = [(p.a_u, p.a_h, p.hint) for p in build_pairs(rows, f"{ht}_{arm}") if p.is_valid]
                cell = make_cell(trips)
                n = cell.n_eligible
                out[ht][direction] = {
                    "n_eligible": n,
                    "change_to_hint": cell.n_switch_to_hint,
                    "change_to_non_hint": cell.n_switch_to_other,
                    "no_change": n - cell.n_switch_to_hint - cell.n_switch_to_other,
                }
        return out

    def pct(c):
        n = c["n_eligible"]
        return {k: 100 * c[k] / n for k in ("change_to_hint", "change_to_non_hint", "no_change")}

    mmlu_following = following_counts(merged_rows(MMLU["tier1"], MMLU["tier2"]))
    gpqa_following = following_counts(merged_rows(GPQA["tier1"], GPQA["tier2"]))
    following = {}
    for ht in ALL_TYPES:
        following[ht] = {}
        for direction in ARMS:
            a, b = mmlu_following[ht][direction], gpqa_following[ht][direction]
            following[ht][direction] = {
                "mmlu": a,
                "gpqa": b,
                "avg_pct": {k: (pct(a)[k] + pct(b)[k]) / 2 for k in pct(a)},
            }

    mmlu_faith = per_type_norm(specs_for(MMLU))
    gpqa_faith = per_type_norm(specs_for(GPQA))
    faithfulness = {}
    for ht in ALL_TYPES:
        faithfulness[ht] = {}
        for direction in ARMS:
            a, b = mmlu_faith[ht][direction], gpqa_faith[ht][direction]
            faithfulness[ht][direction] = {
                "mmlu": a,
                "gpqa": b,
                "avg": None if (a is None or b is None) else (a + b) / 2,
            }

    out = {
        "_meta": {
            "generated_by": "scripts/analyze_sonnet45.py",
            "description": "Sonnet 4.5 per-hint-type detail. following: change-to-hint decomposition "
                           "per dataset (MMLU full released pool; GPQA-Diamond) with the equal-weight "
                           "percent average. faithfulness: per-arm normalized faithfulness, "
                           "equal-weight MMLU+GPQA average.",
        },
        "following": following,
        "faithfulness": faithfulness,
    }
    write_table(OUT_PATH, out, missing, allow_incomplete=False)
    for ht in ALL_TYPES:
        follow_pct = following[ht]["incorrect"]["avg_pct"]["change_to_hint"]
        fa = faithfulness[ht]["incorrect"]["avg"]
        print(f"{ht:24s} incorrect-follow={follow_pct:5.1f}%  incorrect-faith={'-' if fa is None else f'{100*fa:5.1f}%'}")


if __name__ == "__main__":
    main()
