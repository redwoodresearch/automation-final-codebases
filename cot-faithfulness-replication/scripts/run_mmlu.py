"""Collect MMLU hint transcripts for one model. Results stream to an append-only JSONL
(one row per call, full raw API response included); reruns resume via existing-row skip +
content-addressed response cache.

Default: the 8 Tier-1 conditions (released hints suggestion/posthoc/fewshot_symbol x
{True,False} + their 2 unhinted baselines). --tier2 runs the 6 Tier-2 conditions
(metadata/grader_hacking/unethical_information x {True,False}) instead.

Model naming selects the API path:
- Anthropic models by API id (claude-...)
- open-weight models by OpenRouter id (with a "/", e.g. deepseek/deepseek-r1; must be in
  lib.sweep.OPENWEIGHT_MODELS)
- closed-frontier GPT/Gemini models by OpenRouter id (must be in lib.frontier.FRONTIER_MODELS)
--temperature applies to OpenRouter models only (default 1.0; the R1 anchor run uses 0).

Examples (run from the repo root):
  python scripts/run_mmlu.py --model claude-sonnet-4-5-20250929 --pool full
  python scripts/run_mmlu.py --model claude-opus-4-1-20250805 --pool standard --tier2
  python scripts/run_mmlu.py --model deepseek/deepseek-r1 --pool standard --temperature 0
  python scripts/run_mmlu.py --model openai/gpt-5 --pool standard --n-questions 250 \
      --out results/tier1_gpt-5_std250.jsonl
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
from lib.tier1 import build_api_kwargs, parse_response, sweep_tier1_conditions
from lib.tier2 import tier2_conditions

RESULTS_DIR = Path("results")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pool", choices=["pilot", "standard", "full"], required=True)
    parser.add_argument("--conditions", nargs="*", default=None, help="subset of condition names")
    parser.add_argument("--tier2", action="store_true", help="run the 6 Tier-2 conditions instead")
    parser.add_argument("--n-questions", type=int, default=None, help="first N pool questions (default: whole pool)")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=None, help="OpenRouter models only (default 1.0)")
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--assert-cached", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    pools = json.load(open("data/pools.json"))
    indices = pools[f"{args.pool}_pool"]
    if args.n_questions is not None:
        indices = indices[: args.n_questions]

    conditions = tier2_conditions() if args.tier2 else sweep_tier1_conditions()
    if args.conditions:
        by_name = {c.name: c for c in sweep_tier1_conditions() + tier2_conditions()}
        conditions = [by_name[name] for name in args.conditions]

    is_frontier = args.model in FRONTIER_BY_ID
    is_openweight = ("/" in args.model) and not is_frontier
    if is_frontier:
        # Closed-frontier comparison group (GPT/Gemini) via OpenRouter — same transport as
        # open-weight, but the reasoning channel is a vendor SUMMARY (see lib/frontier.py).
        fm = FRONTIER_BY_ID[args.model]
        temperature = args.temperature if args.temperature is not None else 1.0
        short = fm.short if temperature == 1.0 else f"{fm.short}_t{temperature:g}"
        build_kwargs = lambda prompt: build_frontier_kwargs(fm, prompt, temperature)
        call_cached, parse, llm_module = call_openrouter_cached, parse_openrouter_response, lib.openrouter
        task_id_suffix = f"|t{temperature:g}"
    elif is_openweight:
        ow = OPENWEIGHT_BY_ID[args.model]  # KeyError = model not registered in lib/sweep.py
        temperature = args.temperature if args.temperature is not None else 1.0
        short = ow.short if temperature == 1.0 else f"{ow.short}_t{temperature:g}"
        build_kwargs = lambda prompt: build_openweight_kwargs(ow, prompt, temperature)
        call_cached, parse, llm_module = call_openrouter_cached, parse_openrouter_response, lib.openrouter
        # Temperature is part of the task_id so a t0 row can never satisfy a t1 resume (or vice versa).
        task_id_suffix = f"|t{temperature:g}"
    else:
        assert args.temperature is None, "--temperature applies to OpenRouter models only"
        short = model_short(args.model)
        build_kwargs = lambda prompt: build_api_kwargs(args.model, prompt)
        call_cached, parse, llm_module = call_anthropic_cached, parse_response, lib.llm
        task_id_suffix = ""

    default_name = f"{'tier2' if args.tier2 else 'tier1'}_{short}_{args.pool}.jsonl"
    out_path = Path(args.out) if args.out else RESULTS_DIR / default_name
    out_path.parent.mkdir(exist_ok=True)
    done = load_done_task_ids(out_path)

    # Content-policy blocks (e.g. Gemini PROHIBITED_CONTENT, or Anthropic refusals on a sensitive
    # question — Fable 5 refuses a material fraction of tier-1 prompts) are recorded as invalid
    # rows rather than crashing the batch.
    sentinel_fn = openrouter_content_block_sentinel if (is_frontier or is_openweight) else content_filter_sentinel

    llm_module.set_concurrency(args.model, args.concurrency)
    cost_tracker = CostTracker(Path("total_cost.jsonl"), run_description=f"run_mmlu {args.model} {args.pool}")
    # Record the ACTUAL request params (they differ per model family).
    request_config = {k: v for k, v in build_kwargs([]).items() if k not in ("messages", "model")}
    metadata = {"git_hash": git_hash(), "model": args.model, "config": {**request_config, "pool": args.pool}}

    tasks = [
        (condition, index)
        for condition in conditions
        for index in indices
        if f"{args.model}|{condition.name}|{index}|{args.sample_idx}{task_id_suffix}" not in done
    ]
    n_skipped = len(conditions) * len(indices) - len(tasks)
    print(f"{len(tasks)} calls to run ({n_skipped} already in {out_path})")

    write_lock = asyncio.Lock()
    progress = tqdm(total=len(tasks), smoothing=0.01)

    async def run_one(condition, index) -> None:
        record = load_file(condition.source_file)[index]
        api_kwargs = build_kwargs(condition.get_prompt(record))
        try:
            response = await call_cached(
                api_kwargs, sample_idx=args.sample_idx, cost_tracker=cost_tracker, assert_cached=args.assert_cached
            )
        except Exception as e:
            sentinel = sentinel_fn(args.model, e) if sentinel_fn else None
            if sentinel is None:
                raise
            response = sentinel
        row = {
            "task_id": f"{args.model}|{condition.name}|{index}|{args.sample_idx}{task_id_suffix}",
            "input": {
                "condition": condition.name,
                "is_hinted": condition.is_hinted,
                "question_index": index,
                "hint": record.hint if condition.is_hinted else None,
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
