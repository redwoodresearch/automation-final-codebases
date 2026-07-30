"""Collect extra unhinted samples on the correct-hint-eligible questions (MMLU or GPQA).

Feeds scripts/analyze_filtered_faithfulness.py, which restricts the correct-hint analysis to
questions the model never answers correctly without a hint, and scripts/analyze_natural_flip.py,
which measures how often it does. Correct-hint-eligible questions are the ones the model got
wrong on its sample-0 unhinted prompt; this re-asks exactly those, unhinted, N more times.

Two baselines are collected because they are different prompts: unhinted_plain (the baseline for
every hint type except the visual marker) and unhinted_fewshot_symbol (the visual marker's).

sample_idx 0-2 are used by the main collections, so --sample-indices must start at 3.

--stop-on-correct stops resampling a question as soon as one sample lands on the correct answer.
That question can no longer pass the "never correct unhinted" filter, so further samples cannot
change the filter's verdict, and on a model with a 34% flip rate it saves about 40% of the calls.
It does make the flip RATE unusable for that model (a truncated sequence is biased toward
correct), so leave it off if you need scripts/analyze_natural_flip.py to cover the model too.

  python scripts/run_unhinted_resamples.py --model claude-opus-4-1-20250805 \
      --baseline-results results/tier1_opus-4-1_standard.jsonl --sample-indices 3 4 5 6 7 8 \
      --out results/resamples_true_eligible_opus-4-1_standard.jsonl
  python scripts/run_unhinted_resamples.py --model openai/gpt-5 --dataset gpqa \
      --baseline-results results/gpqa_tier1_gpt-5.jsonl --sample-indices 3 4 5 6 \
      --stop-on-correct --concurrency 300 \
      --out results/resamples_gpqa_true_eligible_gpt-5.jsonl
"""

import argparse
import asyncio
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

import lib.gpqa as gpqa
import lib.llm
import lib.openrouter
from cost_tracker import CostTracker
from lib.dataset import load_file
from lib.frontier import FRONTIER_BY_ID, build_frontier_kwargs
from lib.llm import call_anthropic_cached
from lib.openrouter import call_openrouter_cached, parse_openrouter_response
from lib.run_utils import (
    content_filter_sentinel,
    git_hash,
    load_done_task_ids,
    model_short,
    openrouter_content_block_sentinel,
)
from lib.sweep import OPENWEIGHT_BY_ID, build_openweight_kwargs
from lib.tier1 import all_conditions, build_api_kwargs, parse_response

# Each unhinted baseline condition, and the released file whose per-question `hint` field is the
# correct option (the True files hint the correct answer, identically across hint types).
CORRECT_LETTER_SOURCE = {"unhinted_plain": "suggestion_True",
                         "unhinted_fewshot_symbol": "fewshot_symbol_True"}


def correct_letters(dataset: str, n_questions: int) -> dict[str, dict[int, str]]:
    """condition -> {question index: the correct option letter}."""
    if dataset == "mmlu":
        return {c: {i: r.hint for i, r in enumerate(load_file(src))}
                for c, src in CORRECT_LETTER_SOURCE.items()}
    # GPQA: the True arm of any hint type points at the correct answer.
    per_q = {i: gpqa.hint_letter("suggestion_True", i, n_questions) for i in range(n_questions)}
    return {c: per_q for c in CORRECT_LETTER_SOURCE}


def baseline_answers(baseline_results: Path, condition: str) -> dict[int, str | None]:
    answers: dict[int, str | None] = {}
    with open(baseline_results, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            i = row["input"]
            if i["condition"] == condition and i["sample_idx"] == 0:
                assert i["question_index"] not in answers, f"duplicate {condition}|{i['question_index']}"
                answers[i["question_index"]] = row["output"]["answer"]
    assert answers, f"no {condition} sample-0 rows in {baseline_results}"
    return answers


def already_correct(out_path: Path, correct: dict[str, dict[int, str]]) -> set[tuple[str, int]]:
    """(condition, question) pairs where a previously collected resample already hit correct."""
    seen = set()
    if not out_path.exists():
        return seen
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            c, i = r["input"]["condition"], r["input"]["question_index"]
            if r["output"].get("answer") == correct.get(c, {}).get(i):
                seen.add((c, i))
    return seen


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", choices=("mmlu", "gpqa"), default="mmlu")
    parser.add_argument("--baseline-results", type=Path, required=True,
                        help="the tier1 file holding this model's sample-0 unhinted rows")
    parser.add_argument("--sample-indices", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-questions", type=int, default=gpqa.N_QUESTIONS_FULL,
                        help="GPQA only: size of the question pool")
    parser.add_argument("--stop-on-correct", action="store_true",
                        help="stop resampling a question once one sample lands on the correct answer")
    parser.add_argument("--temperature", type=float, default=None,
                        help="OpenRouter models only (default 1.0)")
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--assert-cached", action="store_true")
    args = parser.parse_args()

    assert all(s >= 3 for s in args.sample_indices), "sample_idx 0-2 belong to the main collections"
    sample_indices = sorted(args.sample_indices)

    is_frontier = args.model in FRONTIER_BY_ID
    is_openweight = ("/" in args.model) and not is_frontier
    if is_frontier:
        fm = FRONTIER_BY_ID[args.model]
        temperature = args.temperature if args.temperature is not None else 1.0
        build_kwargs = lambda prompt: build_frontier_kwargs(fm, prompt, temperature)
        call_cached, parse, llm_module = call_openrouter_cached, parse_openrouter_response, lib.openrouter
        suffix = f"|t{temperature:g}"
    elif is_openweight:
        ow = OPENWEIGHT_BY_ID[args.model]
        temperature = args.temperature if args.temperature is not None else 1.0
        build_kwargs = lambda prompt: build_openweight_kwargs(ow, prompt, temperature)
        call_cached, parse, llm_module = call_openrouter_cached, parse_openrouter_response, lib.openrouter
        suffix = f"|t{temperature:g}"
    else:
        assert args.temperature is None, "--temperature applies to OpenRouter models only"
        build_kwargs = lambda prompt: build_api_kwargs(args.model, prompt)
        call_cached, parse, llm_module = call_anthropic_cached, parse_response, lib.llm
        suffix = ""
    sentinel_fn = openrouter_content_block_sentinel if (is_frontier or is_openweight) else content_filter_sentinel

    correct = correct_letters(args.dataset, args.n_questions)
    if args.dataset == "mmlu":
        conditions = {c.name: c for c in all_conditions() if c.name in CORRECT_LETTER_SOURCE}
        prompt_for = lambda cond, i: conditions[cond].get_prompt(load_file(conditions[cond].source_file)[i])
    else:
        prompt_for = lambda cond, i: gpqa.gpqa_prompt(cond, i, args.n_questions)

    # Correct-hint-eligible: the model answered the unhinted prompt, and got it wrong.
    eligible = {}
    for cond in CORRECT_LETTER_SOURCE:
        answers = baseline_answers(args.baseline_results, cond)
        eligible[cond] = sorted(i for i, a in answers.items()
                                if a is not None and a != correct[cond].get(i))
        print(f"{cond}: {len(eligible[cond])} correct-hint-eligible questions")

    done = load_done_task_ids(args.out)
    skip = already_correct(args.out, correct) if args.stop_on_correct else set()
    if skip:
        print(f"{len(skip)} question(s) already resolved by an earlier run (--stop-on-correct)")
    llm_module.set_concurrency(args.model, args.concurrency)
    cost_tracker = CostTracker(Path("total_cost.jsonl"), run_description=f"run_unhinted_resamples {args.model}")
    request_config = {k: v for k, v in build_kwargs([]).items() if k not in ("messages", "model")}
    metadata = {
        "git_hash": git_hash(),
        "model": args.model,
        "config": {**request_config, "dataset": args.dataset,
                   "baseline_results": str(args.baseline_results),
                   "stop_on_correct": bool(args.stop_on_correct)},
    }

    units = [(cond, i) for cond in eligible for i in eligible[cond] if (cond, i) not in skip]
    max_calls = len(units) * len(sample_indices)
    print(f"{len(units)} question-baseline units, up to {max_calls} calls"
          + (" (early stop enabled)" if args.stop_on_correct else ""))

    write_lock = asyncio.Lock()
    progress = tqdm(total=max_calls, smoothing=0.01)
    n_saved = 0

    async def run_unit(cond: str, index: int) -> None:
        """Resample one question, stopping early once it lands on the correct answer."""
        nonlocal n_saved
        for sample_idx in sample_indices:
            task_id = f"{args.model}|{cond}|{index}|{sample_idx}{suffix}"
            if task_id in done:
                progress.update(1)
                continue
            api_kwargs = build_kwargs(prompt_for(cond, index))
            try:
                response = await call_cached(api_kwargs, sample_idx=sample_idx,
                                             cost_tracker=cost_tracker, assert_cached=args.assert_cached)
                parsed = parse(response)
            except Exception as exc:  # content policy blocks etc. -> record, don't kill the batch
                sentinel = sentinel_fn(exc)
                if sentinel is None:
                    raise
                response, parsed = sentinel, {"answer": None}
            row = {
                "task_id": task_id,
                "input": {"condition": cond, "is_hinted": False, "question_index": index,
                          "hint": None, "sample_idx": sample_idx},
                "output": {"raw_response": response, **parsed},
                "metadata": {**metadata, "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()},
            }
            async with write_lock:
                with open(args.out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            progress.update(1)
            if args.stop_on_correct and parsed.get("answer") == correct[cond].get(index):
                n_saved += len(sample_indices) - sample_indices.index(sample_idx) - 1
                progress.update(len(sample_indices) - sample_indices.index(sample_idx) - 1)
                return

    results = await asyncio.gather(*(run_unit(c, i) for c, i in units), return_exceptions=True)
    progress.close()
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        print(f"{len(errors)} unit(s) FAILED; first error: {errors[0]!r}")
        raise SystemExit(1)
    if args.stop_on_correct:
        print(f"early stop saved {n_saved} calls of {max_calls}")
    print(f"done; run cost ${cost_tracker.run_cost:.2f}; results in {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
