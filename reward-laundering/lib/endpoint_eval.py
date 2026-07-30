"""Shared endpoint-eval helpers: neutral subset-sum pass rates per checkpoint + paired-vs-base stats.

The decision rule summarises each (condition, seed) by neutral-prompt subset-sum
accuracy on a large, fixed eval set at the last-k eval checkpoints. Both the multi-seed endpoint
driver (`run_endpoint_evals.py`) and the older smoke script use these helpers so the eval path (same
neutral prompt, same forcing, same verifier) is identical across scripts.

Checkpoint samples are CACHED (keyed by the checkpoint's tinker path) so the reported endpoint numbers
are reproducible and re-runs are free; base samples reuse the base-model cache.
"""

from __future__ import annotations

import math

import asyncio

import tinker

from cost_tracker import CostTracker
from file_cache import FileCache
from lib import config
from lib.benchmark import eval_instances
from lib.coupling import SCAFFOLD_ANSWER_CUE, SOLVED_CUE, SUBSET_CUE, VERIFY_CUE, parse_coupled_output
from lib.parsing import parse_subset_values_answer
from lib.prompts import neutral_scaffold_prompt, subset_sum_prompt
from lib.subset_sum import verify_subset_sum_values
from lib.tinker_client import BASE_SAMPLER, Sampler, sample_coupled_forced


async def neutral_eval_records(
    sampler: Sampler, instances, *, n_samples: int, cache: FileCache,
    tracker: CostTracker | None = None, assert_cached: bool = False, max_concurrency: int = 128,
):
    """Raw per-rollout records on the neutral subset-sum prompt (same prompt/forcing/verifier as the
    base benchmark and the headline eval, so the base model reuses its cache)."""
    return await eval_instances(
        instances, subset_sum_prompt, parse_subset_values_answer,
        lambda inst, parsed: verify_subset_sum_values(inst.values, inst.target, parsed),
        cache=cache, think_budget=config.SUBSET_SUM_THINK_BUDGET, n_samples=n_samples,
        sampler=sampler, tracker=tracker, assert_cached=assert_cached,
        max_concurrency=max_concurrency, verbosity=0)


def pass_rates_from_records(records) -> list[float]:
    """Per-problem pass rate (fraction of samples correct), sorted by instance index."""
    by_inst: dict[int, list[int]] = {}
    for r in records:
        by_inst.setdefault(r.instance_idx, []).append(int(r.correct))
    return [sum(v) / len(v) for _i, v in sorted(by_inst.items())]


def passk_from_records(records) -> float:
    """pass@k = fraction of problems solved by AT LEAST ONE of the k samples (diversity-sensitive).

    A complement to pass@1 (the mean pass rate): SFT-on-own-outputs mode collapse can raise pass@1
    (the model commits its single best mode more confidently) while tanking pass@k (it loses the
    diversity that occasionally found harder subsets). Reported alongside pass@1 so a rise in the
    headline metric that is actually collapse is visible."""
    by_inst: dict[int, list[int]] = {}
    for r in records:
        by_inst.setdefault(r.instance_idx, []).append(int(r.correct))
    if not by_inst:
        return 0.0
    return sum(1 for v in by_inst.values() if any(v)) / len(by_inst)


def token_stats_from_records(records) -> dict:
    """Length / compute diagnostics (addresses the "more test-time compute vs new ability" confound):
    mean sampled tokens on the neutral eval, and the same split by correct vs incorrect rollouts."""
    n = len(records)
    if n == 0:
        return {}
    toks = [r.n_sample_tokens for r in records]
    ok = [r.n_sample_tokens for r in records if r.correct]
    bad = [r.n_sample_tokens for r in records if not r.correct]
    return {
        "mean_sample_tokens": sum(toks) / n,
        "mean_sample_tokens_correct": (sum(ok) / len(ok)) if ok else None,
        "mean_sample_tokens_incorrect": (sum(bad) / len(bad)) if bad else None,
        "parse_error_rate": sum(r.parse_error for r in records) / n,
    }


async def neutral_pass_rates(
    sampler: Sampler, instances, *, n_samples: int, cache: FileCache,
    tracker: CostTracker | None = None, assert_cached: bool = False, max_concurrency: int = 128,
) -> list[float]:
    """Per-problem pass rate on the neutral subset-sum prompt (thin wrapper over the records)."""
    records = await neutral_eval_records(
        sampler, instances, n_samples=n_samples, cache=cache, tracker=tracker,
        assert_cached=assert_cached, max_concurrency=max_concurrency)
    return pass_rates_from_records(records)


async def scaffold_pass_rates(
    sampler: Sampler, instances, *, n_samples: int, cache: FileCache,
    tracker: CostTracker | None = None, assert_cached: bool = False, max_concurrency: int = 128,
) -> list[float]:
    """Per-problem subset-sum pass rate under the neutral+scaffold-minus-coupling prompt (B's
    verify/membership/no-reuse checklist + structured Subset/Solved/Answer commit, no coupling).

    Uses the SAME structured forced flow (`sample_coupled_forced`) as B's training/eval, so within
    this prompt type base and a B checkpoint are apples-to-apples (the isolation eval's requirement).
    Scored on the committed `Subset:` line by the external verifier (never trusts `Solved:`)."""
    async def one(i, ss, s):
        res = await sample_coupled_forced(
            [{"role": "user", "content": neutral_scaffold_prompt(ss)}], cache=cache,
            think_budget=config.SUBSET_SUM_THINK_BUDGET, verify_cue=VERIFY_CUE, subset_cue=SUBSET_CUE,
            solved_cue=SOLVED_CUE, answer_cue=SCAFFOLD_ANSWER_CUE, sample_idx=s, tracker=tracker,
            assert_cached=assert_cached, max_concurrency=max_concurrency, **sampler.sample_kwargs())
        chosen = parse_coupled_output(res.answer_text).subset_values
        return i, int(verify_subset_sum_values(ss.values, ss.target, chosen))
    outs = await asyncio.gather(*[one(i, ss, s) for i, ss in enumerate(instances)
                                  for s in range(n_samples)])
    by: dict[int, list[int]] = {}
    for i, ok in outs:
        by.setdefault(i, []).append(ok)
    return [sum(v) / len(v) for _i, v in sorted(by.items())]


def paired_stats(base_rates: list[float], ckpt_rates: list[float]) -> tuple[float, float]:
    """Paired (per-problem) mean difference ckpt-minus-base and its clustered SE.

    The SE is std(per-problem diffs)/sqrt(n_problems) — a paired clustered SE, not an over-optimistic
    iid binomial SE (both arms see the same problems)."""
    n = len(base_rates)
    assert n == len(ckpt_rates) and n > 0, "base/ckpt rate lists must be same nonzero length"
    diffs = [c - b for b, c in zip(base_rates, ckpt_rates)]
    mean_diff = sum(diffs) / n
    var = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    return mean_diff, math.sqrt(var / n)


async def checkpoint_sampler(service, path: str, *, cache_enabled: bool = True) -> Sampler:
    """A cached Sampler around a checkpoint's sampling client (keyed by the checkpoint path)."""
    sc = await service.create_sampling_client_async(model_path=path)
    return Sampler(sampling_client=sc, sampler_id=path, cache_enabled=cache_enabled)


def base_sampler() -> Sampler:
    return BASE_SAMPLER
