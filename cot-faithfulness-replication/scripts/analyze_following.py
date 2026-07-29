"""Compute the hint-following tables for all 30 models: on each dataset (MMLU, GPQA) and
each hint direction (correct-hint / incorrect-hint arm), the judge-free decomposition of what
eligible examples (unhinted answer != hint target) do when the hint is added:
  change_to_hint     : hinted answer == hint target
  change_to_non_hint : hinted answer != hint target and != unhinted answer
  no_change          : hinted answer == unhinted answer
Counts are pooled over the 6 hint types. Writes results/following_tables.json (committed);
the following figures read that file.

Requires the raw transcripts in results/ (committed for DeepSeek R1 t=0 only; fetch the rest
with data/download_transcripts.py or regenerate with scripts/run_mmlu.py / run_gpqa.py).

  python scripts/analyze_following.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.analysis import build_pairs, load_results, results_file_exists
from lib.faithfulness import TIER1_PAPER_TYPES, TIER2_TYPES
from lib.gpqa_analysis import grid_files
from lib.lineup import lineup
from lib.metrics import make_cell
from lib.tables import write_table

ALL_TYPES = TIER1_PAPER_TYPES + TIER2_TYPES
OUT_PATH = Path("results/following_tables.json")
ARMS = {"correct": "True", "incorrect": "False"}  # hint direction -> released file arm tag


def merged_rows(t1: str, t2: str) -> dict:
    rows = dict(load_results(Path(t1)))
    for key, row in load_results(Path(t2)).items():
        rows.setdefault(key, row)
    return rows


def decomposition(rows: dict, arm_tag: str) -> dict | None:
    trips = []
    eligible_questions = set()
    for ht in ALL_TYPES:
        for p in build_pairs(rows, f"{ht}_{arm_tag}"):
            if p.is_valid:
                trips.append((p.a_u, p.a_h, p.hint))
                if p.a_u != p.hint:
                    eligible_questions.add(p.question_index)
    cell = make_cell(trips)
    n = cell.n_eligible
    if n == 0:
        return None
    return {
        "n_eligible": n,
        # Eligible pairs count each question once per hint type, so they overstate the
        # independent sample: the six observations of a question share that question and
        # (for five of the six hints) the same unhinted baseline. This is the number of
        # DISTINCT questions behind the cell, which is closer to the effective sample size.
        "n_distinct_questions": len(eligible_questions),
        "change_to_hint": cell.n_switch_to_hint,
        "change_to_non_hint": cell.n_switch_to_other,
        "no_change": n - cell.n_switch_to_hint - cell.n_switch_to_other,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="write the table even if some models' transcripts are missing")
    args = parser.parse_args()

    models, missing = {}, []
    for m in lineup():
        gpqa = grid_files(m.gpqa_tag)
        sources = {"mmlu": (m.mmlu_stems["tier1"], m.mmlu_stems["tier2"]),
                   "gpqa": (gpqa["tier1"], gpqa["tier2"])}
        rec = {"display": m.display, "group": m.group}
        for ds, (t1, t2) in sources.items():
            if not (results_file_exists(t1) and results_file_exists(t2)):
                print(f"  {m.display}: {ds} transcripts missing ({t1}) — skipped")
                missing.append(f"{m.short}/{ds}")
                rec[ds] = None
                continue
            rows = merged_rows(t1, t2)
            rec[ds] = {direction: decomposition(rows, arm) for direction, arm in ARMS.items()}
        models[m.short] = rec
        print(f"{m.display:24s} done")

    out = {
        "_meta": {
            "generated_by": "scripts/analyze_following.py",
            "description": "Per-model hint-following decomposition (counts over eligible examples, "
                           "pooled over the 6 hint types). 'correct'/'incorrect' = whether the hint "
                           "points at the correct answer.",
        },
        "models": models,
    }
    write_table(OUT_PATH, out, missing, allow_incomplete=args.allow_incomplete)


if __name__ == "__main__":
    main()
