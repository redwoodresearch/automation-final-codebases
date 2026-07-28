"""Judge verbalization for retained pairs in a model's GPQA grid.

Standard Claude Opus 4.8 judge (the headline numbers) OR the era-matched Claude 3 Opus
variant (--variant model3opus_std, used for the judge-dependence comparison). Records are
built from lib.gpqa (not the MMLU released dataset); the judge sees native thinking/raw
reasoning + the visible response via response_texts — same per-model channel handling as
the MMLU runs.

  # standard judge, both tiers of a grid:
  python scripts/judge_gpqa.py --tag sonnet-4-5 --tier 1
  python scripts/judge_gpqa.py --tag sonnet-4-5 --tier 2
  # era-matched Claude 3 Opus judge on a stratified subsample:
  python scripts/judge_gpqa.py --tag deepseek-r1_t0 --tier 1 --variant model3opus_std --subsample 400
"""

import sys
import argparse
import asyncio
import datetime
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

import lib.gpqa as gpqa
from cost_tracker import CostTracker
from lib.analysis import build_pairs
from lib.gpqa_analysis import grid_files, load_grid_rows
from lib.judge import JUDGE_EFFORT, JUDGE_MODEL, judge_verbalization
from lib.judge_variants import VARIANTS, judge_verbalization_variant
from lib.llm import set_concurrency
from lib.run_utils import git_hash, load_done_task_ids, stratified_subsample
from lib.tier1 import response_texts

TIER_TYPES = {1: gpqa.TIER1_TYPES, 2: gpqa.TIER2_TYPES}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="grid tag, e.g. sonnet-4-5 / deepseek-r1_t0 / deepseek-r1_t1")
    parser.add_argument("--tier", type=int, choices=[1, 2], required=True)
    parser.add_argument("--variant", default=None, choices=sorted(VARIANTS), help="omit for the standard Opus-4.8 judge")
    parser.add_argument("--subsample", type=int, default=None, help="stratified subsample of retained pairs (era-band cost lever)")
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--assert-cached", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = load_grid_rows(args.tag)
    retained = []
    for ht in TIER_TYPES[args.tier]:
        for arm in ("True", "False"):
            for pair in build_pairs(rows, f"{ht}_{arm}"):
                if pair.is_retained:
                    retained.append(pair)
    if args.subsample is not None:
        retained = stratified_subsample(retained, args.subsample)

    prefix = "judge" if args.variant is None else f"judge_{args.variant}"
    out_path = args.out or Path("results") / f"{prefix}_gpqa_tier{args.tier}_{args.tag}.jsonl"
    done = load_done_task_ids(out_path)
    todo = [p for p in retained if f"judge|{p.condition}|{p.question_index}|0" not in done]
    label = args.variant or "standard(opus-4-8)"
    print(f"{len(retained)} retained GPQA tier{args.tier} pairs for {args.tag}; {len(todo)} to judge ({label})")

    variant = VARIANTS[args.variant] if args.variant else None
    judge_model = variant.model if variant else JUDGE_MODEL
    set_concurrency(judge_model, args.concurrency)
    cost_tracker = CostTracker(Path("total_cost.jsonl"), run_description=f"run_judge_gpqa {args.tag} tier{args.tier} {label}")
    run_git_hash = git_hash()

    write_lock = asyncio.Lock()
    progress = tqdm(total=len(todo), smoothing=0.01)

    async def judge_one(pair) -> None:
        hint_type = pair.condition.rsplit("_", 1)[0]
        record = gpqa.gpqa_record(pair.condition, pair.question_index)
        thinking_text, visible_text = response_texts(rows[(pair.condition, pair.question_index)]["output"]["raw_response"])
        if variant is None:
            verdict = await judge_verbalization(
                hint_type, record, thinking_text, visible_text, pair.a_h,
                cost_tracker=cost_tracker, assert_cached=args.assert_cached)
        else:
            verdict = await judge_verbalization_variant(
                variant, hint_type, record, thinking_text, visible_text, pair.a_h,
                cost_tracker=cost_tracker, assert_cached=args.assert_cached)
        row = {
            "task_id": f"judge|{pair.condition}|{pair.question_index}|0",
            "input": {"condition": pair.condition, "hint_type": hint_type,
                      "question_index": pair.question_index, "sample_idx": 0,
                      "hint": pair.hint, "a_u": pair.a_u, "a_h": pair.a_h},
            "output": verdict,
            "metadata": {"git_hash": run_git_hash, "judge_model": judge_model,
                         "judge_variant": args.variant, "judge_effort": (JUDGE_EFFORT if variant is None else None),
                         "source_results": grid_files(args.tag)[f"tier{args.tier}"],
                         "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()},
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
