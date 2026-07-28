"""Judge verbalization for retained pairs in a Tier-2 results file.

Tier-2 differs from Tier-1: (a) hint types are metadata/grader_hacking/unethical_information;
(b) the a_u baseline comes from a SEPARATE unhinted_plain results file (Tier-2 hints are pure
insertions into the plain question); (c) each Tier-2 row's source `record` (question text,
hint letter) is read from suggestion_{True,False} per its correctness arm.

  python scripts/judge_mmlu_tier2.py results/tier2_sonnet-4-5_full.jsonl \
      --baseline results/tier1_sonnet-4-5_full.jsonl
"""

import sys
import argparse
import asyncio
import datetime
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from cost_tracker import CostTracker
from lib.analysis import build_pairs, load_results
from lib.dataset import load_file
from lib.judge import JUDGE_EFFORT, JUDGE_MODEL, judge_verbalization
from lib.llm import set_concurrency
from lib.run_utils import git_hash, load_done_task_ids
from lib.tier1 import response_texts
from lib.tier2 import TIER2_HINT_TYPES

def hint_type_and_source(condition: str) -> tuple[str, str]:
    """(hint_type, source_file) for a Tier-2 condition name,
    e.g. "grader_hacking_False" -> ("grader_hacking", "suggestion_False")."""
    correctness = condition.rsplit("_", 1)[1]
    stem = condition.rsplit("_", 1)[0]
    assert stem in TIER2_HINT_TYPES, stem
    return stem, f"suggestion_{correctness}"


def tier2_conditions_in(rows) -> list[str]:
    return sorted({cond for (cond, _idx) in rows if cond.rsplit("_", 1)[1] in ("True", "False")})


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path", type=Path)
    parser.add_argument("--baseline", type=Path, required=True, help="file with unhinted_plain sample-0 rows for a_u")
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="subset of Tier-2 condition names (default: all Tier-2 conditions in the file)")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--assert-cached", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_path = args.out or Path("results") / f"judge_{args.results_path.stem}.jsonl"
    done = load_done_task_ids(out_path)

    # Merge Tier-2 hinted rows with the baseline's unhinted_plain rows so build_pairs can
    # find a_u (unhinted_condition_for maps all Tier-2 conditions -> unhinted_plain).
    tier2_rows = load_results(args.results_path, sample_idx=args.sample_idx)
    baseline_rows = load_results(args.baseline, sample_idx=args.sample_idx)
    rows = dict(tier2_rows)
    for key, row in baseline_rows.items():
        if key[0] == "unhinted_plain":
            rows.setdefault(key, row)

    conditions = args.conditions if args.conditions else tier2_conditions_in(tier2_rows)
    retained = []
    for condition in conditions:
        for pair in build_pairs(rows, condition):
            if pair.is_retained:
                retained.append(pair)
    todo = [p for p in retained if f"judge|{p.condition}|{p.question_index}|{args.sample_idx}" not in done]
    print(f"{len(retained)} retained Tier-2 pairs; {len(todo)} to judge ({len(retained) - len(todo)} done)")

    set_concurrency(JUDGE_MODEL, args.concurrency)
    cost_tracker = CostTracker(Path("total_cost.jsonl"), run_description=f"judge_mmlu_tier2 {args.results_path.name}")
    run_git_hash = git_hash()

    write_lock = asyncio.Lock()
    progress = tqdm(total=len(todo), smoothing=0.01)

    async def judge_one(pair) -> None:
        hint_type, source_file = hint_type_and_source(pair.condition)
        record = load_file(source_file)[pair.question_index]
        hinted_row = rows[(pair.condition, pair.question_index)]
        thinking_text, visible_text = response_texts(hinted_row["output"]["raw_response"])
        verdict = await judge_verbalization(
            hint_type, record, thinking_text, visible_text, pair.a_h,
            cost_tracker=cost_tracker, assert_cached=args.assert_cached,
        )
        row = {
            "task_id": f"judge|{pair.condition}|{pair.question_index}|{args.sample_idx}",
            "input": {
                "condition": pair.condition,
                "hint_type": hint_type,
                "question_index": pair.question_index,
                "sample_idx": args.sample_idx,
                "hint": pair.hint,
                "a_u": pair.a_u,
                "a_h": pair.a_h,
            },
            "output": verdict,
            "metadata": {
                "git_hash": run_git_hash,
                "judge_model": JUDGE_MODEL,
                "judge_effort": JUDGE_EFFORT,
                "source_results": str(args.results_path),
                "baseline_results": str(args.baseline),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        }
        async with write_lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        progress.update(1)

    results = await asyncio.gather(*(judge_one(p) for p in todo), return_exceptions=True)
    progress.close()
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        print(f"{len(errors)} judge calls FAILED; first: {errors[0]!r}")
        raise SystemExit(1)
    print(f"done; run cost ${cost_tracker.run_cost:.2f}; results in {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
