"""Compute the faithfulness-by-hint-direction tables for all 30 models.

For each model x dataset (MMLU, GPQA) x hint direction (correct / incorrect arm) x hint type:
normalized faithfulness = min(raw / alpha, 1), where raw = P(judge says the CoT verbalized the
hint | the model changed to the hint) and alpha corrects for chance switching (lib/metrics.py).
The summary number per (model x direction) is the 6-type mean per dataset, then the equal-weight
(MMLU + GPQA) / 2 average of the two dataset means. Writes results/faithfulness_tables.json
(committed); the faithfulness figures read that file.

Requires raw transcripts + judge verdict files in results/ (judge files are committed; fetch
transcripts with data/download_transcripts.py or regenerate).

  python scripts/analyze_faithfulness.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.faithfulness import TIER1_PAPER_TYPES, TIER2_TYPES, FaithSpec, per_arm_cells
from lib.analysis import results_file_exists
from lib.gpqa_analysis import grid_files
from lib.lineup import lineup
from lib.tables import write_table

ALL_TYPES = TIER1_PAPER_TYPES + TIER2_TYPES
OUT_PATH = Path("results/faithfulness_tables.json")


def retarget(stems: dict, variant: str | None) -> dict:
    """Point the judge_* entries at a judge variant's verdict files (transcripts unchanged)."""
    if not variant:
        return stems
    out = dict(stems)
    for k in ("judge_tier1", "judge_tier2"):
        if k in out:
            name = Path(out[k]).name
            out[k] = str(Path(out[k]).parent / name.replace("judge_", f"judge_{variant}_", 1))
    return out
ARMS = {"correct": "True", "incorrect": "False"}


def specs_for(stems: dict) -> list[FaithSpec]:
    return [
        FaithSpec("tier1", stems["tier1"], stems["judge_tier1"], conditions=tuple(TIER1_PAPER_TYPES)),
        FaithSpec("tier2", stems["tier2"], stems["judge_tier2"], baseline_path=stems["tier1"],
                  conditions=tuple(TIER2_TYPES)),
    ]


def per_type_by_arm(specs: list[FaithSpec]) -> dict:
    """{hint_type: {direction: {normalized, raw, retained, judged} | None}}"""
    out = {}
    for spec in specs:
        cells = per_arm_cells(spec)
        for base in spec.conditions:
            out[base] = {}
            for direction, arm in ARMS.items():
                got = cells.get(f"{base}_{arm}")
                if got is None:
                    out[base][direction] = None
                    continue
                cell, _parse_fail = got
                out[base][direction] = {
                    "normalized": cell.normalized_faithfulness,
                    "raw": cell.raw_faithfulness,
                    "retained": cell.usage.n_retained,
                    "judged": cell.n_judged,
                }
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-variant", default=None,
                        help="read a judge variant's verdicts (e.g. neutral_opus48) and write "
                             "results/faithfulness_tables_<variant>.json instead")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="write the table even if some models' inputs are missing")
    args = parser.parse_args()

    models, missing = {}, []
    for m in lineup():
        gpqa = grid_files(m.gpqa_tag)
        sources = {"mmlu": retarget(m.mmlu_stems, args.judge_variant),
                   "gpqa": retarget(gpqa, args.judge_variant)}
        rec = {"display": m.display, "group": m.group, "per_type": {}, "mean_normalized": {}}
        dataset_means = {}
        for ds, stems in sources.items():
            needed = [stems["tier1"], stems["tier2"], stems["judge_tier1"], stems["judge_tier2"]]
            if not all(results_file_exists(p) for p in needed):
                print(f"  {m.display}: {ds} transcripts/judge files missing — skipped")
                missing.append(f"{m.short}/{ds}")
                rec["per_type"][ds] = None
                dataset_means[ds] = {d: None for d in ARMS}
                continue
            by_type = per_type_by_arm(specs_for(stems))
            rec["per_type"][ds] = by_type
            dataset_means[ds] = {
                direction: mean([None if by_type[ht][direction] is None else by_type[ht][direction]["normalized"]
                                 for ht in ALL_TYPES])
                for direction in ARMS
            }
        rec["mean_normalized"] = {
            **{ds: dataset_means[ds] for ds in sources},
            # Equal-weight (MMLU + GPQA)/2 average of the two per-dataset 6-type means.
            "avg": {
                direction: (None if dataset_means["mmlu"][direction] is None
                            or dataset_means["gpqa"][direction] is None
                            else (dataset_means["mmlu"][direction] + dataset_means["gpqa"][direction]) / 2)
                for direction in ARMS
            },
        }
        models[m.short] = rec
        a = rec["mean_normalized"]["avg"]
        fmt = lambda v: "  -  " if v is None else f"{100*v:5.1f}"
        print(f"{m.display:24s} correct={fmt(a['correct'])}  incorrect={fmt(a['incorrect'])}")

    out = {
        "_meta": {
            "generated_by": "scripts/analyze_faithfulness.py",
            "description": "Normalized faithfulness by hint direction, per model. per_type holds the "
                           "per-(dataset x hint type x direction) cells; mean_normalized holds the 6-type "
                           "means per dataset and their equal-weight MMLU+GPQA average.",
        },
        "models": models,
    }
    out_path = (OUT_PATH if not args.judge_variant
                else OUT_PATH.with_name(f"{OUT_PATH.stem}_{args.judge_variant}.json"))
    write_table(out_path, out, missing, allow_incomplete=args.allow_incomplete)


if __name__ == "__main__":
    main()
