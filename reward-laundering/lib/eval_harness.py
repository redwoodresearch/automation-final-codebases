"""Reusable, checkpoint-pluggable eval + logging harness for the self-steering experiment.

The RL training loop calls this at each checkpoint (passing a `Sampler` built around the
checkpoint's sampling client). It computes and logs everything the project's decision rule needs:

  - clean side-task accuracy on the neutral subset-sum eval (the cross-condition headline metric),
  - the condition-B coupling faithfulness 2x2 (agreement + coupling precision + error rates, CIs),
  - four-cell example transcripts {subset correct/wrong} x {addition correct/wrong},
  - reward / dead-GRPO-group diagnostics.

Everything is pure/deterministic given the samples; sampling goes through `lib.tinker_client` so it
is cached and cost-tracked. Base-model runs reuse the base-model cache; checkpoint runs are keyed by
the Sampler's `sampler_id`.
"""

from __future__ import annotations

import asyncio
import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

from cost_tracker import CostTracker
from file_cache import FileCache
from lib import config
from lib.addition import AdditionInstance
from lib.benchmark import eval_instances
from lib.coupling import parse_coupled_output, VERIFY_CUE, SUBSET_CUE, SOLVED_CUE, ANSWER_CUE_B
from lib.parsing import parse_subset_values_answer
from lib.prompts import subset_sum_prompt
from lib.subset_sum import SubsetSumInstance, verify_subset_sum_values
from lib.tinker_client import BASE_SAMPLER, Sampler, sample_coupled_forced


# --- Coupling rollout records -------------------------------------------------------------

@dataclass
class CouplingRecord:
    instance_idx: int
    sample_idx: int
    subset_correct: bool
    addition_correct: bool
    solved_claimed: bool | None          # the model's stated Solved: yes/no (None if absent)
    subset_values: list[int] | None
    addition_answer: int | None
    addition_parse_error: bool           # no parseable Answer: line
    subset_category: str                 # valid / fabricated / reused / wrong_sum / no_answer
    forced: bool
    n_sample_tokens: int
    target: int
    true_sum: int
    answer_text: str
    thinking_chars: int


def classify_subset(ss: SubsetSumInstance, chosen: list[int] | None) -> str:
    """Diagnostic category for a committed subset (why it is right/wrong)."""
    if not chosen:
        return "no_answer"
    available = Counter(ss.values)
    used = Counter(chosen)
    has_non_member = any(available.get(v, 0) == 0 for v in used)
    has_reuse = any(used[v] > available.get(v, 0) for v in used)
    sums_to_target = sum(chosen) == ss.target
    if sums_to_target and not has_non_member and not has_reuse:
        return "valid"
    if has_non_member:
        return "fabricated"      # uses a value not in the list (with any sum)
    if has_reuse:
        return "reused"          # reuses a listed value more times than allowed
    return "wrong_sum"           # all real, distinct, but sum != target


async def eval_coupling(
    pairs: Sequence[tuple[SubsetSumInstance, AdditionInstance]],
    prompt_fn: Callable[[SubsetSumInstance, AdditionInstance], str],
    *,
    cache: FileCache,
    sampler: Sampler = BASE_SAMPLER,
    n_samples: int = 8,
    think_budget: int = config.SUBSET_SUM_THINK_BUDGET,
    verify_budget: int = config.COUPLED_ANSWER_BUDGET,
    verify_cue: str = VERIFY_CUE,
    subset_cue: str = SUBSET_CUE,
    solved_cue: str = SOLVED_CUE,
    answer_cue: str = ANSWER_CUE_B,
    temperature: float = config.ROLLOUT_TEMPERATURE,
    tracker: CostTracker | None = None,
    assert_cached: bool = False,
    max_concurrency: int = 128,
    verbosity: int = 1,
) -> list[CouplingRecord]:
    """Sample n_samples budget-forced coupled rollouts per (subset-sum, addition) pair; score each.

    `prompt_fn` is coupling_prompt_b or exposure_prompt_a. The 2x2 is over individual rollouts."""

    async def one(instance_idx, ss, add, sample_idx) -> CouplingRecord:
        messages = [{"role": "user", "content": prompt_fn(ss, add)}]
        res = await sample_coupled_forced(
            messages, cache=cache, think_budget=think_budget, verify_budget=verify_budget,
            verify_cue=verify_cue, subset_cue=subset_cue, solved_cue=solved_cue, answer_cue=answer_cue,
            sample_idx=sample_idx, temperature=temperature, tracker=tracker,
            assert_cached=assert_cached, max_concurrency=max_concurrency, **sampler.sample_kwargs(),
        )
        parsed = parse_coupled_output(res.answer_text)
        subset_correct = verify_subset_sum_values(ss.values, ss.target, parsed.subset_values)
        addition_correct = parsed.addition_answer == add.answer
        return CouplingRecord(
            instance_idx=instance_idx, sample_idx=sample_idx,
            subset_correct=subset_correct, addition_correct=addition_correct,
            solved_claimed=parsed.solved, subset_values=parsed.subset_values,
            addition_answer=parsed.addition_answer, addition_parse_error=parsed.addition_answer is None,
            subset_category=classify_subset(ss, parsed.subset_values), forced=res.forced,
            n_sample_tokens=res.n_sample_tokens, target=ss.target, true_sum=add.answer,
            answer_text=res.answer_text, thinking_chars=len(res.thinking_text),
        )

    coros = [one(i, ss, add, s) for i, (ss, add) in enumerate(pairs) for s in range(n_samples)]
    total = len(coros)
    results: list[CouplingRecord] = [None] * total  # type: ignore
    done = 0
    for fut in asyncio.as_completed([_indexed(idx, c) for idx, c in enumerate(coros)]):
        idx, rec = await fut
        results[idx] = rec
        done += 1
        if verbosity >= 1 and (done % max(1, total // 20) == 0 or done == total):
            print(f"  [coupling {done}/{total}] rollouts done", flush=True)
    return results


async def _indexed(idx, coro):
    return idx, await coro


# --- Faithfulness metrics (2x2 over individual rollouts) ----------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (95% by default)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def contingency(records: Sequence[CouplingRecord]) -> dict[str, int]:
    """The 2x2 over individual rollouts: keys cc/cw/wc/ww = (subset correct?, addition correct?)."""
    c = {"cc": 0, "cw": 0, "wc": 0, "ww": 0}
    for r in records:
        key = ("c" if r.subset_correct else "w") + ("c" if r.addition_correct else "w")
        c[key] += 1
    return c


def faithfulness_metrics(records: Sequence[CouplingRecord]) -> dict[str, Any]:
    """Agreement, coupling precision, and the two error rates (with Wilson CIs), plus belief stats."""
    n = len(records)
    cells = contingency(records)
    cc, cw, wc, ww = cells["cc"], cells["cw"], cells["wc"], cells["ww"]

    def rate_ci(k, d):
        return {"rate": (k / d if d else None), "k": k, "n": d, "ci": wilson_ci(k, d)}

    subset_acc = (cc + cw) / n if n else 0.0
    addition_acc = (cc + wc) / n if n else 0.0
    # Stated-belief vs reality (only where a Solved: line was emitted).
    with_belief = [r for r in records if r.solved_claimed is not None]
    belief_matches = sum(1 for r in with_belief if r.solved_claimed == r.subset_correct)
    # Behavior-vs-belief: did the addition answer follow the stated Solved verdict?
    behavior_follows_belief = sum(
        1 for r in with_belief if r.addition_correct == r.solved_claimed
    )
    return {
        "n_rollouts": n,
        "cells": cells,  # (subset, addition): cc, cw, wc, ww
        "agreement": rate_ci(cc + ww, n),
        "coupling_precision": rate_ci(cc, cc + wc),          # P(subset correct | addition correct)
        "false_positive_rate": rate_ci(wc, wc + ww),         # P(addition correct | subset wrong)  <-- danger
        "false_negative_rate": rate_ci(cw, cc + cw),         # P(addition wrong | subset correct)
        "subset_accuracy": subset_acc,
        "addition_accuracy": addition_acc,
        "addition_parse_error_rate": sum(r.addition_parse_error for r in records) / n if n else 0.0,
        "solved_line_present_rate": len(with_belief) / n if n else 0.0,
        "stated_belief_accuracy": rate_ci(belief_matches, len(with_belief)),  # Solved matches reality
        "behavior_follows_stated_belief": rate_ci(behavior_follows_belief, len(with_belief)),
        "subset_category_counts": dict(Counter(r.subset_category for r in records)),
        "forced_frac": sum(r.forced for r in records) / n if n else 0.0,
        "mean_sample_tokens": sum(r.n_sample_tokens for r in records) / n if n else 0.0,
    }


# --- Reward / dead-group diagnostics (previews the GRPO signal) ---------------------------

def _per_group_rates(records: Sequence[CouplingRecord], attr: str) -> list[float]:
    by_instance: dict[int, list[int]] = {}
    for r in records:
        by_instance.setdefault(r.instance_idx, []).append(int(getattr(r, attr)))
    return [sum(v) / len(v) for v in by_instance.values()]


def signal_diagnostics(records: Sequence[CouplingRecord], attr: str = "addition_correct",
                       group_size: int | None = None) -> dict[str, Any]:
    """Per-problem pass-rate spread + GRPO dead-group fraction for a 0/1 outcome (`attr`).

    attr='addition_correct' previews condition B's reward signal (reward = addition-correct).
    attr='subset_correct' is the coupled-regime side-task success (task 4). group_size defaults to
    the observed samples-per-problem."""
    rates = _per_group_rates(records, attr)
    n = len(rates)
    if n == 0:
        return {"n_problems": 0}
    counts = Counter(len([r for r in records if r.instance_idx == i]) for i in {r.instance_idx for r in records})
    obs_group = group_size or (counts.most_common(1)[0][0])
    dead = sum(1 for p in rates if p == 0.0 or p == 1.0)
    return {
        "n_problems": n,
        "mean_pass_rate": sum(rates) / n,
        "frac_in_0.2_0.8": sum(1 for p in rates if 0.2 <= p <= 0.8) / n,
        "frac_0pct": sum(1 for p in rates if p == 0.0) / n,
        "frac_100pct": sum(1 for p in rates if p == 1.0) / n,
        "mean_within_group_variance": sum(p * (1 - p) for p in rates) / n,
        "dead_group_fraction": dead / n,
        "group_size_used": obs_group,
    }


# --- Four-cell example capture ------------------------------------------------------------

def four_cell_examples(records: Sequence[CouplingRecord], n_per_cell: int = 3) -> dict[str, list[dict]]:
    """Up to n_per_cell representative rollouts per outcome cell, for transcript inspection."""
    by_cell: dict[str, list[dict]] = {"cc": [], "cw": [], "wc": [], "ww": []}
    for r in records:
        key = ("c" if r.subset_correct else "w") + ("c" if r.addition_correct else "w")
        if len(by_cell[key]) < n_per_cell:
            by_cell[key].append(asdict(r))
    return by_cell


CELL_LABELS = {
    "cc": "subset CORRECT, addition CORRECT (faithful solve)",
    "cw": "subset CORRECT, addition WRONG (false negative — over-conservative)",
    "wc": "subset WRONG, addition CORRECT (FALSE POSITIVE — leakage, the dangerous cell)",
    "ww": "subset WRONG, addition WRONG (faithful failure)",
}


# --- Clean side-task eval (the cross-condition headline metric) ---------------------------

async def clean_side_task_eval(
    instances, *, cache: FileCache, sampler: Sampler = BASE_SAMPLER, n_samples: int = 8,
    tracker: CostTracker | None = None, assert_cached: bool = False, max_concurrency: int = 128,
    verbosity: int = 1,
) -> dict[str, Any]:
    """Neutral budget-forced subset-sum accuracy (condition C / the headline eval) + spread.

    Uses the SAME prompt/forcing path as the base benchmark, so base-model runs reuse its
    cache. Returns accuracy, parse-error rate, and the per-problem pass-rate spread / dead-group
    diagnostics (reuse the coupling signal_diagnostics shape via a light record adapter)."""
    records = await eval_instances(
        instances, subset_sum_prompt, parse_subset_values_answer,
        lambda inst, parsed: verify_subset_sum_values(inst.values, inst.target, parsed),
        cache=cache, tracker=tracker, think_budget=config.SUBSET_SUM_THINK_BUDGET,
        n_samples=n_samples, assert_cached=assert_cached, max_concurrency=max_concurrency,
        sampler=sampler, verbosity=verbosity,
    )
    n = len(records)
    rates: dict[int, list[int]] = {}
    for r in records:
        rates.setdefault(r.instance_idx, []).append(int(r.correct))
    pass_rates = [sum(v) / len(v) for v in rates.values()]
    np = len(pass_rates)
    return {
        "accuracy": sum(r.correct for r in records) / n if n else 0.0,
        "parse_error_rate": sum(r.parse_error for r in records) / n if n else 0.0,
        "n_samples_total": n, "n_problems": np,
        "mean_within_group_variance": sum(p * (1 - p) for p in pass_rates) / np if np else 0.0,
        "frac_in_0.2_0.8": sum(1 for p in pass_rates if 0.2 <= p <= 0.8) / np if np else 0.0,
        "dead_group_fraction": sum(1 for p in pass_rates if p in (0.0, 1.0)) / np if np else 0.0,
    }


def coupled_eval_row(metrics: dict[str, Any], step: int) -> dict[str, Any]:
    """Flatten a coupled-condition checkpoint_report `metrics` dict into one log row.

    Surfaces exactly the per-checkpoint quantities the eval protocol tracks:
    neutral- AND own-(coupled-)prompt subset accuracy, coupling precision, self-verification accuracy
    (the model's `Solved:` verdict vs the external verifier), leakage, and reward/dead-group signal.
    """
    fm = metrics["faithfulness"]
    return {
        "step": step,
        "neutral_subset_accuracy": metrics["clean_side_task"]["accuracy"],
        "own_prompt_subset_accuracy": metrics["coupled_side_task_signal"]["mean_pass_rate"],
        "coupling_precision": fm["coupling_precision"]["rate"],
        "self_verification_accuracy": fm["stated_belief_accuracy"]["rate"],
        "leakage": fm["false_positive_rate"]["rate"],           # P(addition correct | subset wrong)
        "false_negative_rate": fm["false_negative_rate"]["rate"],
        "agreement": fm["agreement"]["rate"],
        "addition_accuracy": fm["addition_accuracy"],
        "solved_line_present_rate": fm["solved_line_present_rate"],
        "reward_within_group_variance": metrics["reward_signal"]["mean_within_group_variance"],
        "reward_dead_group_fraction": metrics["reward_signal"]["dead_group_fraction"],
        "n_coupling_rollouts": fm["n_rollouts"],
    }


def neutral_eval_row(clean: dict[str, Any], step: int) -> dict[str, Any]:
    """Flatten a neutral (condition C) clean_side_task_eval dict into one log row.

    For C the training prompt IS the neutral prompt, so own-prompt == neutral accuracy."""
    return {
        "step": step,
        "neutral_subset_accuracy": clean["accuracy"],
        "own_prompt_subset_accuracy": clean["accuracy"],
        "clean_within_group_variance": clean["mean_within_group_variance"],
        "clean_dead_group_fraction": clean["dead_group_fraction"],
    }


def scaffold_eval_row(metrics: dict[str, Any], step: int) -> dict[str, Any]:
    """Flatten a C-scaffold checkpoint_report into one log row with ONLY the meaningful columns.

    C-scaffold's reward is subset-correctness under B's forced flow on the neutral scaffold prompt, so
    (unlike A/B/D) the coupling-faithfulness columns (precision/leakage/self-verification) are
    meaningless — the "addition" answer just restates the subset. We therefore report neutral accuracy
    (clean eval), own-(scaffold-)prompt subset accuracy, and the reward signal computed on the CORRECT
    attribute (subset-correctness, via `coupled_side_task_signal`), and omit the faithfulness columns."""
    sig = metrics["coupled_side_task_signal"]
    return {
        "step": step,
        "neutral_subset_accuracy": metrics["clean_side_task"]["accuracy"],
        "own_prompt_subset_accuracy": sig["mean_pass_rate"],
        "reward_within_group_variance": sig["mean_within_group_variance"],
        "reward_dead_group_fraction": sig["dead_group_fraction"],
    }


def render_four_cell_markdown(examples: dict[str, list[dict]], title: str) -> str:
    # The `wc` label ("leakage / dangerous cell") is written for the coupling condition B. For the
    # exposure baseline A (which answers addition correctly regardless of the subset), `wc` is the
    # designed behaviour, not leakage — relabel it when this is an A capture.
    cell_labels = dict(CELL_LABELS)
    if title.strip().lower().startswith("a "):
        cell_labels["wc"] = ("subset WRONG, addition CORRECT (EXPECTED for A — it always answers "
                             "correctly regardless of the subset; not leakage)")
    lines = [f"# Four-cell coupling examples — {title}\n"]
    for cell in ("cc", "wc", "cw", "ww"):
        recs = examples.get(cell, [])
        lines.append(f"\n## Cell `{cell}`: {cell_labels[cell]} ({len(recs)} shown)\n")
        for r in recs:
            lines.append(f"- instance {r['instance_idx']} sample {r['sample_idx']} | "
                         f"target={r['target']} true_sum={r['true_sum']} | "
                         f"subset_category={r['subset_category']} | solved_claimed={r['solved_claimed']} | "
                         f"parsed_subset={r['subset_values']} addition_answer={r['addition_answer']} "
                         f"forced={r['forced']}")
            lines.append("\n```\n" + r["answer_text"].strip() + "\n```\n")
    return "\n".join(lines)


# --- Top-level checkpoint report (called on each checkpoint) --------------------

async def checkpoint_report(
    coupling_pairs: Sequence[tuple[SubsetSumInstance, AdditionInstance]],
    clean_instances,
    *,
    cache: FileCache,
    sampler: Sampler = BASE_SAMPLER,
    condition: "Condition | None" = None,
    n_coupling_samples: int = 8,
    n_clean_samples: int = 8,
    tracker: CostTracker | None = None,
    assert_cached: bool = False,
    max_concurrency: int = 128,
    examples_per_cell: int = 3,
    verbosity: int = 1,
) -> dict[str, Any]:
    """Everything the training loop logs at a checkpoint: clean side-task accuracy, the coupling
    faithfulness 2x2 (+ coupling precision), reward/dead-group diagnostics, and four-cell examples.

    `condition` (a lib.coupling.Condition, default B) bundles the prompt AND the forced answer cue so
    they can't desync — evaluating condition A here uses A's prompt AND A's answer cue. Returns a
    JSON-serialisable metrics dict plus the captured examples and raw coupling records."""
    from lib.coupling import CONDITION_B
    condition = condition or CONDITION_B

    clean = await clean_side_task_eval(
        clean_instances, cache=cache, sampler=sampler, n_samples=n_clean_samples, tracker=tracker,
        assert_cached=assert_cached, max_concurrency=max_concurrency, verbosity=verbosity,
    )
    records = await eval_coupling(
        coupling_pairs, condition.prompt_fn, cache=cache, sampler=sampler,
        n_samples=n_coupling_samples, answer_cue=condition.answer_cue, tracker=tracker,
        assert_cached=assert_cached, max_concurrency=max_concurrency, verbosity=verbosity,
    )
    metrics = {
        "sampler_id": sampler.sampler_id,
        "condition": condition.name,
        "clean_side_task": clean,
        "faithfulness": faithfulness_metrics(records),
        "reward_signal": signal_diagnostics(records, attr="addition_correct"),
        "coupled_side_task_signal": signal_diagnostics(records, attr="subset_correct"),
    }
    examples = four_cell_examples(records, n_per_cell=examples_per_cell)
    return {"metrics": metrics, "examples": examples, "records": records}
