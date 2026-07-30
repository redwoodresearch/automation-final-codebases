"""Collect extra unhinted samples on the correct-hint-eligible questions.

Feeds scripts/analyze_natural_flip.py, which measures how often a model lands on the correct
answer with no hint at all. Correct-hint-eligible questions are the ones the model got wrong on
its sample-0 unhinted prompt; this script re-asks exactly those questions, unhinted, N more times.

Two baselines are collected because they are different prompts: unhinted_plain (the baseline for
every hint type except the visual marker) and unhinted_fewshot_symbol (the visual marker's).

sample_idx 0-2 are used by the main collections, so --sample-indices must start at 3.

  python scripts/run_unhinted_resamples.py --model claude-opus-4-1-20250805 \
      --baseline-results results/tier1_opus-4-1_standard.jsonl --sample-indices 3 4 5 6 7 8 \
      --out results/resamples_true_eligible_opus-4-1_standard.jsonl
  python scripts/run_unhinted_resamples.py --model claude-sonnet-4-5-20250929 \
      --baseline-results results/tier1_sonnet-4-5_full.jsonl --sample-indices 3 4 5 6 \
      --out results/resamples_true_eligible_sonnet-4-5_full.jsonl
"""

import argparse
import asyncio
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

import lib.llm
import lib.openrouter
from cost_tracker import CostTracker
from lib.dataset import load_file
from lib.llm import call_anthropic_cached
from lib.openrouter import call_openrouter_cached, parse_openrouter_response
from lib.run_utils import git_hash, load_done_task_ids
from lib.sweep import OPENWEIGHT_BY_ID, build_openweight_kwargs
from lib.tier1 import all_conditions, build_api_kwargs, parse_response

# Each unhinted baseline condition, and the released file whose per-question `hint` field is the
# correct option (the True files hint the correct answer, identically across hint types).
CORRECT_LETTER_SOURCE = {"unhinted_plain": "suggestion_True",
                         "unhinted_fewshot_symbol": "fewshot_symbol_True"}


def load_baseline_answers(baseline_results: Path, condition: str) -> dict[int, str | None]:
    answers: dict[int, str | None] = {}
    with open(baseline_results, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["input"]["condition"] == condition and row["input"]["sample_idx"] == 0:
                index = row["input"]["question_index"]
                assert index not in answers, f"duplicate baseline row {condition}|{index}"
                answers[index] = row["output"]["answer"]
    assert answers, f"no {condition} sample-0 rows in {baseline_results}"
    return answers


def correct_hint_eligible_indices(baseline_results: Path, condition: str) -> list[int]:
    """Questions the model answered, but got wrong, on the unhinted prompt."""
    correct_records = load_file(CORRECT_LETTER_SOURCE[condition])
    answers = load_baseline_answers(baseline_results, condition)
    return sorted(i for i, a in answers.items() if a is not None and a != correct_records[i].hint)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline-results", type=Path, required=True,
                        help="the tier1 file holding this model's sample-0 unhinted rows")
    parser.add_argument("--sample-indices", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--assert-cached", action="store_true")
    args = parser.parse_args()

    assert all(s >= 3 for s in args.sample_indices), "sample_idx 0-2 belong to the main collections"
    conditions = {c.name: c for c in all_conditions() if c.name in CORRECT_LETTER_SOURCE}
    eligible = {name: correct_hint_eligible_indices(args.baseline_results, name) for name in conditions}
    for name, indices in eligible.items():
        print(f"{name}: {len(indices)} correct-hint-eligible questions")

    # Resampling happens at the same temperature as the main collection the baseline came from:
    # temperature 1 for open-weight models. (A temperature-0 run is ~deterministic, so it has no
    # natural flip rate to measure.)
    if "/" in args.model:
        ow = OPENWEIGHT_BY_ID[args.model]
        build_kwargs = lambda prompt: build_openweight_kwargs(ow, prompt, temperature=1.0)
        call_cached, parse, llm_module = call_openrouter_cached, parse_openrouter_response, lib.openrouter
    else:
        build_kwargs = lambda prompt: build_api_kwargs(args.model, prompt)
        call_cached, parse, llm_module = call_anthropic_cached, parse_response, lib.llm

    done = load_done_task_ids(args.out)
    llm_module.set_concurrency(args.model, args.concurrency)
    cost_tracker = CostTracker(Path("total_cost.jsonl"), run_description=f"run_unhinted_resamples {args.model}")
    request_config = {k: v for k, v in build_kwargs([]).items() if k not in ("messages", "model")}
    metadata = {
        "git_hash": git_hash(),
        "model": args.model,
        "config": {**request_config, "baseline_results": str(args.baseline_results)},
    }

    tasks = [
        (condition_name, index, sample_idx)
        for condition_name, indices in eligible.items()
        for index in indices
        for sample_idx in args.sample_indices
        if f"{args.model}|{condition_name}|{index}|{sample_idx}" not in done
    ]
    n_total = sum(len(indices) for indices in eligible.values()) * len(args.sample_indices)
    print(f"{len(tasks)} calls to run ({n_total - len(tasks)} already in {args.out})")

    write_lock = asyncio.Lock()
    progress = tqdm(total=len(tasks), smoothing=0.01)

    async def run_one(condition_name: str, index: int, sample_idx: int) -> None:
        condition = conditions[condition_name]
        record = load_file(condition.source_file)[index]
        api_kwargs = build_kwargs(condition.get_prompt(record))
        response = await call_cached(
            api_kwargs, sample_idx=sample_idx, cost_tracker=cost_tracker, assert_cached=args.assert_cached
        )
        row = {
            "task_id": f"{args.model}|{condition_name}|{index}|{sample_idx}",
            "input": {
                "condition": condition_name,
                "is_hinted": False,
                "question_index": index,
                "hint": None,
                "sample_idx": sample_idx,
            },
            "output": {"raw_response": response, **parse(response)},
            "metadata": {**metadata, "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()},
        }
        async with write_lock:
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        progress.update(1)

    results = await asyncio.gather(*(run_one(c, i, s) for c, i, s in tasks), return_exceptions=True)
    progress.close()
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        print(f"{len(errors)} calls FAILED; first error: {errors[0]!r}")
        raise SystemExit(1)
    print(f"done; run cost ${cost_tracker.run_cost:.2f}; results in {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
