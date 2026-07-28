"""Collect the GPQA hint grid for one model.

Mirrors run_mmlu.py (same model dispatch, response cache, resume-skip, row schema) but the prompts
come from lib.gpqa (GPQA-diamond questions + the six released/reconstructed hint templates) instead of
the MMLU released dataset. Two tiers, matching the two GPQA epistemic tiers:
  default    : unhinted_plain, unhinted_fewshot_symbol, suggestion/posthoc/fewshot_symbol × {True,False}
  --tier2    : metadata/grader_hacking/unethical_information × {True,False}

Anthropic models by API id (claude-…); open-weight by OpenRouter id (with "/"), which must be in
lib.sweep.OPENWEIGHT_MODELS. --temperature applies to open-weight only (default 1.0; R1 anchor = 0).

Examples:
  python scripts/run_gpqa.py --model claude-sonnet-4-5-20250929
  python scripts/run_gpqa.py --model claude-sonnet-4-5-20250929 --tier2
  python scripts/run_gpqa.py --model deepseek/deepseek-r1 --temperature 0
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
import lib.llm
import lib.openrouter
from cost_tracker import CostTracker
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
from lib.tier1 import build_api_kwargs, parse_response

RESULTS_DIR = Path("results")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tier2", action="store_true", help="run the 3 full-reconstruction types instead of the tier-1 set")
    parser.add_argument("--conditions", nargs="*", default=None, help="subset of condition names (default: the tier's conditions)")
    parser.add_argument("--n-questions", type=int, default=gpqa.N_QUESTIONS_FULL)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=None, help="open-weight models only (default 1.0)")
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--assert-cached", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    conditions = gpqa.TIER2_CONDITIONS if args.tier2 else gpqa.TIER1_CONDITIONS
    if args.conditions:
        conditions = args.conditions
    indices = list(range(args.n_questions))

    is_frontier = args.model in FRONTIER_BY_ID
    is_openweight = ("/" in args.model) and not is_frontier
    if is_frontier:
        # Closed-frontier comparison group (GPT/Gemini) via OpenRouter — same transport as the
        # open-weight path, but the reasoning channel is a vendor SUMMARY (see lib/frontier.py).
        fm = FRONTIER_BY_ID[args.model]
        temperature = args.temperature if args.temperature is not None else 1.0
        short = fm.short if temperature == 1.0 else f"{fm.short}_t{temperature:g}"
        build_kwargs = lambda prompt: build_frontier_kwargs(fm, prompt, temperature)
        call_cached, parse, llm_module = call_openrouter_cached, parse_openrouter_response, lib.openrouter
        task_id_suffix = f"|t{temperature:g}"
    elif is_openweight:
        ow = OPENWEIGHT_BY_ID[args.model]
        temperature = args.temperature if args.temperature is not None else 1.0
        short = f"{ow.short}_t{temperature:g}"
        build_kwargs = lambda prompt: build_openweight_kwargs(ow, prompt, temperature)
        call_cached, parse, llm_module = call_openrouter_cached, parse_openrouter_response, lib.openrouter
        task_id_suffix = f"|t{temperature:g}"
    else:
        assert args.temperature is None, "--temperature applies to open-weight (OpenRouter) models only"
        short = model_short(args.model)
        build_kwargs = lambda prompt: build_api_kwargs(args.model, prompt)
        call_cached, parse, llm_module = call_anthropic_cached, parse_response, lib.llm
        task_id_suffix = ""

    # OpenRouter models (frontier + open-weight) can hit a provider content-policy block (e.g. Gemini
    # PROHIBITED_CONTENT on a sensitive question); record it as an invalid OpenRouter-shaped row rather
    # than crashing the batch. The Anthropic path keeps its own content-filter sentinel.
    sentinel_fn = openrouter_content_block_sentinel if (is_frontier or is_openweight) else content_filter_sentinel

    default_name = f"gpqa_{'tier2' if args.tier2 else 'tier1'}_{short}.jsonl"
    out_path = Path(args.out) if args.out else RESULTS_DIR / default_name
    out_path.parent.mkdir(exist_ok=True)
    done = load_done_task_ids(out_path)

    llm_module.set_concurrency(args.model, args.concurrency)
    cost_tracker = CostTracker(Path("total_cost.jsonl"), run_description=f"run_gpqa {args.model} {'tier2' if args.tier2 else 'tier1'}")
    request_config = {k: v for k, v in build_kwargs([{"role": "human", "content": "x"}]).items() if k not in ("messages", "model")}
    metadata = {"git_hash": git_hash(), "model": args.model, "config": {**request_config, "pool": "gpqa_diamond"}}

    tasks = [
        (condition, index)
        for condition in conditions
        for index in indices
        if f"{args.model}|{condition}|{index}|{args.sample_idx}{task_id_suffix}" not in done
    ]
    n_skipped = len(conditions) * len(indices) - len(tasks)
    print(f"{len(tasks)} calls to run ({n_skipped} already in {out_path})")

    write_lock = asyncio.Lock()
    progress = tqdm(total=len(tasks), smoothing=0.01)

    async def run_one(condition, index) -> None:
        prompt = gpqa.gpqa_prompt(condition, index, args.n_questions)
        api_kwargs = build_kwargs(prompt)
        try:
            response = await call_cached(
                api_kwargs, sample_idx=args.sample_idx, cost_tracker=cost_tracker, assert_cached=args.assert_cached
            )
        except Exception as e:
            sentinel = sentinel_fn(args.model, e)
            if sentinel is None:
                raise
            response = sentinel
        is_hinted = condition not in gpqa.BASELINES
        row = {
            "task_id": f"{args.model}|{condition}|{index}|{args.sample_idx}{task_id_suffix}",
            "input": {
                "condition": condition,
                "is_hinted": is_hinted,
                "question_index": index,
                "hint": gpqa.hint_letter(condition, index, args.n_questions) if is_hinted else None,
                "sample_idx": args.sample_idx,
            },
            "output": {"raw_response": response, **parse(response)},
            "metadata": {**metadata, "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()},
        }
        async with write_lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
        progress.update(1)

    results = await asyncio.gather(*(run_one(c, i) for c, i in tasks), return_exceptions=True)
    progress.close()
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        print(f"{len(errors)} calls FAILED; first error: {errors[0]!r}")
        raise SystemExit(1)
    print(f"done; run cost ${cost_tracker.run_cost:.2f}; results in {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
