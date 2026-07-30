"""Judge verbalization (Claude Opus 4.8) for all retained pairs in a Tier-1 MMLU results file.

  python scripts/judge_mmlu_tier1.py results/tier1_sonnet-4-5_full.jsonl
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
from lib.judge_variants import VARIANTS, judge_verbalization_variant
from lib.llm import set_concurrency
from lib.run_utils import git_hash
from lib.tier1 import all_conditions, response_texts


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path", type=Path)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--assert-cached", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--variant", default=None, choices=sorted(VARIANTS),
                        help="judge-prompt/model variant; default is the standard Opus 4.8 judge")
    args = parser.parse_args()

    variant = VARIANTS[args.variant] if args.variant else None
    judge_model = variant.model if variant else JUDGE_MODEL
    prefix = "judge" if variant is None else f"judge_{variant.name}"
    out_path = args.out or Path("results") / f"{prefix}_{args.results_path.stem}.jsonl"
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            done = {json.loads(line)["task_id"] for line in f if line.strip()}

    rows = load_results(args.results_path, sample_idx=args.sample_idx)
    retained = []
    for condition in all_conditions():
        if not condition.is_hinted:
            continue
        for pair in build_pairs(rows, condition.name):
            if pair.is_retained:
                retained.append(pair)
    todo = [p for p in retained if f"judge|{p.condition}|{p.question_index}|{args.sample_idx}" not in done]
    print(f"{len(retained)} retained pairs; {len(todo)} to judge ({len(retained) - len(todo)} done)")

    set_concurrency(judge_model, args.concurrency)
    cost_tracker = CostTracker(Path("total_cost.jsonl"), run_description=f"judge_mmlu_tier1 {args.results_path.name}")
    run_git_hash = git_hash()

    write_lock = asyncio.Lock()
    progress = tqdm(total=len(todo), smoothing=0.01)

    async def judge_one(pair) -> None:
        hint_type = pair.condition.rsplit("_", 1)[0]
        record = load_file(pair.condition)[pair.question_index]
        hinted_row = rows[(pair.condition, pair.question_index)]
        thinking_text, visible_text = response_texts(hinted_row["output"]["raw_response"])
        judge_args = (hint_type, record, thinking_text, visible_text, pair.a_h)
        judge_kwargs = dict(cost_tracker=cost_tracker, assert_cached=args.assert_cached)
        verdict = await (judge_verbalization_variant(variant, *judge_args, **judge_kwargs)
                         if variant else judge_verbalization(*judge_args, **judge_kwargs))
        row = {
            "task_id": f"judge|{pair.condition}|{pair.question_index}|{args.sample_idx}",
            "input": {
                "condition": pair.condition,
                "question_index": pair.question_index,
                "sample_idx": args.sample_idx,
                "hint": pair.hint,
                "a_u": pair.a_u,
                "a_h": pair.a_h,
            },
            "output": verdict,
            "metadata": {
                "git_hash": run_git_hash,
                "judge_model": judge_model,
                "judge_effort": JUDGE_EFFORT,
                "source_results": str(args.results_path),
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
