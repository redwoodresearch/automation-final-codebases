"""Generic cached evaluator: sample N completions per instance, parse, verify.

Task-agnostic: callers supply prompt/parse/verify callables. Returns flat per-sample
records suitable for pandas analysis and per-problem pass-rate computation.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

from cost_tracker import CostTracker
from file_cache import FileCache
from lib import config
from lib.tinker_client import BASE_SAMPLER, Sampler, sample_budget_forced, sample_cached


@dataclass
class EvalRecord:
    instance_idx: int
    sample_idx: int
    correct: bool
    parse_error: bool
    is_clean: bool
    forced: bool
    parsed: Any
    answer_text: str
    thinking_chars: int
    n_sample_tokens: int
    stop_reason: str


async def _eval_one(
    instance_idx, messages, sample_idx, parse_fn, verify_check_fn, *,
    cache, tracker, temperature, top_p, max_tokens, think_budget, assert_cached, max_concurrency,
    sampler: Sampler = BASE_SAMPLER,
) -> EvalRecord:
    if think_budget is not None:
        result = await sample_budget_forced(
            messages, cache=cache, think_budget=think_budget, sample_idx=sample_idx,
            temperature=temperature, top_p=top_p, tracker=tracker, assert_cached=assert_cached,
            max_concurrency=max_concurrency, **sampler.sample_kwargs(),
        )
    else:
        result = await sample_cached(
            messages, cache=cache, sample_idx=sample_idx, temperature=temperature, top_p=top_p,
            max_tokens=max_tokens, tracker=tracker, assert_cached=assert_cached,
            max_concurrency=max_concurrency, **sampler.sample_kwargs(),
        )
    parsed = parse_fn(result.answer_text)
    parse_error = parsed is None
    correct = bool(verify_check_fn(parsed)) if not parse_error else False
    return EvalRecord(
        instance_idx=instance_idx,
        sample_idx=sample_idx,
        correct=correct,
        parse_error=parse_error,
        is_clean=result.is_clean,
        forced=result.forced,
        parsed=parsed,
        answer_text=result.answer_text,
        thinking_chars=len(result.thinking_text),
        n_sample_tokens=result.n_sample_tokens,
        stop_reason=result.stop_reason,
    )


async def eval_instances(
    instances: Sequence[Any],
    prompt_fn: Callable[[Any], str],
    parse_fn: Callable[[str], Any],
    verify_fn: Callable[[Any, Any], bool],
    *,
    cache: FileCache,
    tracker: CostTracker | None = None,
    temperature: float = config.ROLLOUT_TEMPERATURE,
    top_p: float = config.ROLLOUT_TOP_P,
    max_tokens: int = config.MAX_TOKENS,
    think_budget: int | None = None,
    n_samples: int = 1,
    assert_cached: bool = False,
    max_concurrency: int = 64,
    verbosity: int = 1,
    sampler: Sampler = BASE_SAMPLER,
) -> list[EvalRecord]:
    """Evaluate instances with n_samples each. verify_fn(instance, parsed) -> bool.

    If think_budget is set, uses budget-forced sampling (cap CoT, force an answer).
    Pass a non-default `sampler` to evaluate a training checkpoint instead of the base
    model."""
    coros = []
    for i, inst in enumerate(instances):
        messages = [{"role": "user", "content": prompt_fn(inst)}]
        for s in range(n_samples):
            coros.append(
                _eval_one(
                    i, messages, s, parse_fn, (lambda parsed, _inst=inst: verify_fn(_inst, parsed)),
                    cache=cache, tracker=tracker, temperature=temperature, top_p=top_p,
                    max_tokens=max_tokens, think_budget=think_budget, assert_cached=assert_cached,
                    max_concurrency=max_concurrency, sampler=sampler,
                )
            )

    total = len(coros)
    done = 0
    results: list[EvalRecord] = [None] * total  # type: ignore

    async def run(idx, coro):
        nonlocal done
        res = await coro
        done += 1
        if verbosity >= 1 and (done % max(1, total // 20) == 0 or done == total):
            print(f"  [{done}/{total}] samples done", flush=True)
        return idx, res

    for fut in asyncio.as_completed([run(i, c) for i, c in enumerate(coros)]):
        idx, res = await fut
        results[idx] = res
    return results


def records_to_dicts(records: list[EvalRecord]) -> list[dict]:
    return [asdict(r) for r in records]
