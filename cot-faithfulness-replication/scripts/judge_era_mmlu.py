"""Era-matched judge (Claude 3 Opus, standard prompt) over the DeepSeek R1 MMLU retained pairs.

Processes both hint tiers of one R1 temperature run and writes one output file per tier:
  results/judge_model3opus_std_tier1_deepseek-r1[_t0]_standard.jsonl
  results/judge_model3opus_std_tier2_deepseek-r1[_t0]_standard.jsonl

Standard (Opus 4.8) judge outputs are never touched. The post's judge-dependence figure uses
the temp-0 run:

  python scripts/judge_era_mmlu.py --temp t0
"""

import argparse
import asyncio
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from cost_tracker import CostTracker
from lib.analysis import build_pairs, load_results
from lib.dataset import load_file
from lib.judge_variants import VARIANTS, judge_verbalization_variant
from lib.llm import set_concurrency
from lib.run_utils import git_hash, load_done_task_ids
from lib.tier1 import all_conditions, response_texts
from scripts.judge_mmlu_tier2 import hint_type_and_source, tier2_conditions_in


def stems(temp: str) -> dict[str, str]:
    infix = "_t0" if temp == "t0" else ""
    return {
        "tier1": f"results/tier1_deepseek-r1{infix}_standard.jsonl",
        "tier2": f"results/tier2_deepseek-r1{infix}_standard.jsonl",
    }


def gather_retained(temp: str) -> list[tuple[str, object, dict]]:
    """[(tier, pair, hinted_row)] for all retained pairs of both tiers at one temperature."""
    paths = stems(temp)
    t1_rows = load_results(Path(paths["tier1"]))
    t2_rows = dict(load_results(Path(paths["tier2"])))
    for key, row in t1_rows.items():
        if key[0] == "unhinted_plain":
            t2_rows.setdefault(key, row)

    out = []
    for condition in all_conditions():
        if not condition.is_hinted:
            continue
        for pair in build_pairs(t1_rows, condition.name):
            if pair.is_retained:
                out.append(("tier1", pair, t1_rows[(pair.condition, pair.question_index)]))
    for condition in tier2_conditions_in(t2_rows):
        for pair in build_pairs(t2_rows, condition):
            if pair.is_retained:
                out.append(("tier2", pair, t2_rows[(pair.condition, pair.question_index)]))
    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="model3opus_std", choices=sorted(VARIANTS))
    parser.add_argument("--temp", choices=["t0", "t1"], default="t0")
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--assert-cached", action="store_true")
    args = parser.parse_args()

    variant = VARIANTS[args.variant]
    infix = "_t0" if args.temp == "t0" else ""
    out_paths = {
        tier: Path(f"results/judge_{variant.name}_{tier}_deepseek-r1{infix}_standard.jsonl")
        for tier in ("tier1", "tier2")
    }
    done = {tier: load_done_task_ids(p) for tier, p in out_paths.items()}

    retained = gather_retained(args.temp)
    todo = [
        (tier, pair, row)
        for tier, pair, row in retained
        if f"judge|{pair.condition}|{pair.question_index}|0" not in done[tier]
    ]
    print(f"{len(retained)} retained pairs ({args.temp}); {len(todo)} to judge with variant={variant.name}")

    set_concurrency(variant.model, args.concurrency)
    cost_tracker = CostTracker(Path("total_cost.jsonl"), run_description=f"judge_era_mmlu {variant.name} {args.temp}")
    run_git_hash = git_hash()

    write_lock = asyncio.Lock()
    progress = tqdm(total=len(todo), smoothing=0.01)

    async def judge_one(tier: str, pair, hinted_row) -> None:
        if tier == "tier1":
            hint_type = pair.condition.rsplit("_", 1)[0]
            record = load_file(pair.condition)[pair.question_index]
        else:
            hint_type, source_file = hint_type_and_source(pair.condition)
            record = load_file(source_file)[pair.question_index]
        thinking_text, visible_text = response_texts(hinted_row["output"]["raw_response"])
        verdict = await judge_verbalization_variant(
            variant, hint_type, record, thinking_text, visible_text, pair.a_h,
            cost_tracker=cost_tracker, assert_cached=args.assert_cached,
        )
        row = {
            "task_id": f"judge|{pair.condition}|{pair.question_index}|0",
            "input": {
                "condition": pair.condition,
                "hint_type": hint_type,
                "question_index": pair.question_index,
                "sample_idx": 0,
                "hint": pair.hint,
                "a_u": pair.a_u,
                "a_h": pair.a_h,
            },
            "output": verdict,
            "metadata": {
                "git_hash": run_git_hash,
                "judge_variant": variant.name,
                "judge_model": variant.model,
                "judge_thinking": variant.supports_thinking,
                "source_results": stems(args.temp)[tier],
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        }
        async with write_lock:
            with open(out_paths[tier], "a") as f:
                f.write(json.dumps(row) + "\n")
        progress.update(1)

    results = await asyncio.gather(*(judge_one(t, p, r) for t, p, r in todo), return_exceptions=True)
    progress.close()
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        print(f"{len(errors)} judge calls FAILED; first: {errors[0]!r}")
        raise SystemExit(1)
    print(f"done; run cost ${cost_tracker.run_cost:.2f}; outputs: {', '.join(str(p) for p in out_paths.values())}")


if __name__ == "__main__":
    asyncio.run(main())
