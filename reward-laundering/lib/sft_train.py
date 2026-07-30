"""Supervised fine-tuning (SFT) training path — used for the unfiltered SFT baselines.

The paper's two SFT baselines both fine-tune on rollouts from the reward-laundering prompt with NO
reward and NO correctness filter, to test whether the reward-laundering gain is genuine RL credit
assignment or merely imitation of on-distribution reasoning:
  - on-policy unfiltered (`run_sft_training` with `filter="none"`): each round regenerate rollouts
    from the current model and imitate ALL of them.
  - base-rollout unfiltered (`run_base_rollout_sft`): imitate a FIXED pool of base-model rollouts,
    sampled once (no on-policy regeneration).

This module is structurally identical to `lib/rl_train.py`'s GRPO loop with two swaps:
  - the GRPO advantage  ->  a keep/drop step. The shipped baselines keep every rollout
    (`filter="none"`). An optional `filter="correctness"` mode also exists (keep only the rollouts
    whose committed addition Answer is correct — reading the SAME main-task reward channel as
    conditions A/B, so it NEVER inspects the subset-sum answer); it is not used by the paper's
    baselines but is retained for completeness.
  - the importance-sampling policy-gradient loss  ->  supervised CROSS-ENTROPY on the kept
    demonstrations, with the loss weight on EXACTLY the model-sampled spans (search CoT + verify
    span + the committed `Subset:`/`Solved:`/`Answer:` values) and ZERO on the prompt + every
    injected forcing cue.

The rollout generation (`lib.rl_rollout.rollout_coupled`) and the loss masking
(`tinker_cookbook`'s `trajectory_to_data`) are the SAME code the GRPO setup uses — so the SFT loss
trains on precisely the span set that carried the GRPO advantage (`trajectory_to_sft_datums` just
re-labels the trajectory mask as cross-entropy `weights`). The loss masking is locked by
tests/test_sft_masking.py and tests/test_sft_unfiltered.py.

On-policy loop = generate -> (optional keep/drop) -> SFT update -> re-wrap a fresh sampling client
from the updated weights -> repeat, mirroring `lib/rl_train.py` step-for-step (same per-seed problem
schedule, same group_size x batch_size x n_steps, so the total rollout budget, the unique-problem
count, and the gradient-update count all match the banked GRPO runs).
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import attrs
import tinker
from tinker import TensorData

from cost_tracker import CostTracker
from lib import config
from lib.coupling import Condition
from lib.pools import load_addition_pool, load_subset_sum_pool
from lib.rl_conditions import RL_CONDITIONS, RLCondition
from lib.rl_rollout import RolloutInfo, rollout_coupled
from lib.rl_train import _effective_lr, _outcome_diag

from tinker_cookbook.hyperparam_utils import get_lr
from tinker_cookbook.rl.data_processing import trajectory_to_data
from tinker_cookbook.rl.types import Trajectory
from tinker_cookbook.supervised.common import compute_mean_nll

# Conditions that make sense for SFT: the coupled scaffold with the main-task (addition) reward
# channel. Any optional keep/drop step reads addition-correctness ONLY — never the subset answer.
SFT_CONDITION_NAMES = ("a", "b")


def sft_condition(name: str) -> RLCondition:
    """The RLCondition for an SFT arm ('a'/'b'), validated to use the main-task-only reward channel."""
    cond = RL_CONDITIONS[name]
    assert name in SFT_CONDITION_NAMES, f"SFT arm must be one of {SFT_CONDITION_NAMES}, got {name!r}"
    assert cond.kind == "coupled", "SFT arms use the coupled scaffold (A/B prompt)"
    assert cond.reward_attr == "addition_correct", (
        "SFT filter reads the MAIN-TASK channel only (addition-correct); it must never read the "
        "subset-sum answer")
    assert not cond.shuffle_reward, "SFT arms do not use D's reward permutation"
    return cond


def keep_rollout(condition: RLCondition, info: RolloutInfo) -> bool:
    """The filter: keep this rollout iff its committed addition Answer is correct.

    Reads `condition.reward_attr` (== 'addition_correct' for A/B), the exact same reward channel the
    GRPO A/B reward used — so, like that reward, it NEVER inspects the subset-sum answer. For A this
    keeps ~all rollouts; for B it keeps the believed-solved rollouts (the coupling routes the
    main-task filter into a side-task-success filter)."""
    return bool(getattr(info, condition.reward_attr))


FILTER_MODES = ("correctness", "none")


def should_keep(filter_mode: str, condition: RLCondition, info: RolloutInfo) -> bool:
    """Whether to imitate this rollout, given the filter mode.

    - "correctness": optional keep/drop rule (keep iff addition-correct; `keep_rollout`). Not used by
      the paper's baselines; retained for completeness.
    - "none": the UNFILTERED baseline — keep EVERY rollout regardless of correctness (imitate the
      addition-wrong / deliberately-wrong-answer rollouts too), with no reward or selection at all.
      This is the "no selection" end of the spectrum (unfiltered SFT vs GRPO); it tests whether the
      reward-laundering side-task gain is genuine RL credit assignment or mere imitation of
      self-generated on-distribution reasoning."""
    assert filter_mode in FILTER_MODES, f"filter must be one of {FILTER_MODES}, got {filter_mode!r}"
    if filter_mode == "none":
        return True
    return keep_rollout(condition, info)


def trajectory_to_sft_datums(traj: Trajectory, *, reduction: str = "mean") -> list[tinker.Datum]:
    """Supervised (cross-entropy) datums for one kept rollout, reusing the GRPO loss masking verbatim.

    `trajectory_to_data(traj, 1.0)` builds the per-token mask (1.0 on the model-sampled action tokens,
    0.0 on the prompt + injected cues) and the left-shifted target tokens exactly as the GRPO path
    does. We re-label that mask as the cross-entropy `weights` (and drop the advantages/logprobs the
    PG loss needs), so the SFT loss lands on precisely the span set that carried the GRPO advantage.
    Almost always one Datum (the forced flow's observations chain as exact token prefixes); >1 only if
    an observation is not a prefix extension (which the masking test asserts does not happen here).

    `reduction`:
      - "mean" (default): normalize each datum's weights to sum to 1 over its model-sampled tokens
        (token-mean CE). This removes length bias and makes the per-rollout gradient scale uniform
        (removes a length/keep-count confound between conditions).
      - "sum": keep the raw 0/1 mask (token-sum CE), matching the GRPO path's per-token weighting.
    The support (WHICH tokens are trained) is identical for both — only the per-token weight magnitude
    differs."""
    assert reduction in ("mean", "sum"), f"reduction must be 'mean' or 'sum', got {reduction!r}"
    pg_data = trajectory_to_data(traj, traj_advantage=1.0)
    sft_data: list[tinker.Datum] = []
    for datum in pg_data:
        mask = datum.loss_fn_inputs["mask"].to_torch().float()  # left-shifted; 1.0 sampled, 0.0 cues/prompt
        if reduction == "mean":
            total = float(mask.sum())
            weights = mask / total if total > 0 else mask
        else:
            weights = mask
        sft_data.append(tinker.Datum(
            model_input=datum.model_input,
            loss_fn_inputs={
                "target_tokens": datum.loss_fn_inputs["target_tokens"],
                "weights": TensorData.from_torch(weights),
            },
        ))
    return sft_data


@attrs.frozen
class SFTTrainConfig:
    """Config for one on-policy SFT seed. Defaults MATCH the banked GRPO config
    (group 16 x batch 8 x 30 steps, LoRA rank 32, per-seed offset 500 + 240*seed) so the rollout
    budget, unique-problem count, and gradient-update count all match. `lr`/`lr_schedule` default to
    the cookbook-recommended SFT LR with linear decay."""

    condition: str = "a"                 # coupled-scaffold condition name ('a' or 'b')
    filter: str = "none"                 # 'none' (the unfiltered baseline — keep every rollout, no
                                         # reward) or 'correctness' (optional keep/drop, unused here)
    group_size: int = 16                 # rollouts sampled per problem per round (== GRPO group_size)
    batch_size: int = 8                  # distinct problems per round (== GRPO batch_size)
    n_steps: int = 30                    # on-policy rounds (== GRPO n_steps)
    # --- Offline base-model-rollout variant (run_base_rollout_sft) only ---
    n_pool_problems: int = 80            # distinct problems whose base rollouts form the fixed pool
                                         # (pool size = n_pool_problems * group_size; reused across
                                         #  rounds/epochs — sampling is one-time + cached)
    lr: float | None = None              # None -> hyperparam_utils.get_lr(model)
    lr_schedule: str = "linear"          # match GRPO's linear decay over n_steps
    lr_min_frac: float = 0.0
    reduction: str = "mean"              # SFT CE weighting: 'mean' (per-rollout token-mean) or 'sum'
    downsample_kept_frac: float | None = None  # optional: keep this fraction of the kept rollouts each
                                         # round (seeded), a data-volume control (unused by the baselines)
    lora_rank: int = 32
    think_budget: int = config.SUBSET_SUM_THINK_BUDGET
    max_concurrency: int = 128
    seed: int = 0
    problem_offset: int = 500            # per-seed callers pass 500 + 240*seed (disjoint slices, == GRPO)
    fixed_problems: bool = False         # reuse the same batch each round (tiny-batch overfit sanity)
    eval_every: int = 0                  # 0 -> only a final eval (this IS used by run_sft_training)
    # Provenance only (the cadence eval is driven by the driver's CLI + eval_fn, not by these):
    n_eval_clean: int = 100
    n_eval_samples: int = 4              # the locked 100x4 neutral eval
    keep_last_checkpoint: bool = True
    keep_checkpoint_steps: tuple = ()
    tag: str = "smoke"


@dataclass
class SFTStepResult:
    step: int
    condition: str
    infos_by_group: list[list[RolloutInfo]]
    kept_flags_by_group: list[list[bool]]
    metrics: dict
    n_kept: int
    n_train_tokens: int
    checkpoint_path: str | None = None


def _load_coupled_problems(offset: int, n: int):
    ss = load_subset_sum_pool("train")
    add = load_addition_pool("train")
    return [(ss[(offset + i) % len(ss)], add[(offset + i) % len(add)]) for i in range(n)]


def batch_problems(cfg: SFTTrainConfig, step: int) -> list:
    """One problem per group in this round's batch (same schedule as lib.rl_train.batch_problems for
    the coupled kind: a fresh disjoint slice each round, so across n_steps rounds we cover
    n_steps*batch_size unique problems — matching the GRPO seed's unique-problem set)."""
    start = cfg.problem_offset if cfg.fixed_problems else cfg.problem_offset + step * cfg.batch_size
    return _load_coupled_problems(start, cfg.batch_size)


async def rollout_group(
    sampling_client, coupling: Condition, problem, cfg: SFTTrainConfig, step: int, group_idx: int,
    *, tracker: CostTracker | None,
) -> tuple[list[Trajectory], list[RolloutInfo]]:
    """Sample `group_size` on-policy coupled rollouts for one problem (same seed scheme as GRPO)."""
    ss, add = problem

    async def one(rollout_idx: int) -> tuple[Trajectory, RolloutInfo]:
        seed = cfg.seed + 1_000_003 * step + 1009 * group_idx + rollout_idx
        return await rollout_coupled(sampling_client, ss, add, coupling, think_budget=cfg.think_budget,
                                     seed=seed, tracker=tracker, max_concurrency=cfg.max_concurrency)

    pairs = await asyncio.gather(*[one(r) for r in range(cfg.group_size)])
    return [t for t, _ in pairs], [i for _, i in pairs]


def _filter_diag(infos_by_group, kept_flags_by_group) -> dict:
    """Filter + coupling/decoupling/diversity diagnostics (subset-correct fields are DIAGNOSTIC ONLY;
    never used for the keep/drop decision).

    keep_fraction: rollouts passing the addition-correct filter. subset_correct_rate_among_kept: for
    B this ≈ coupling precision (kept ≈ solved); for A it ≈ the base subset rate (kept ≈ all).
    leakage_among_generated = P(addition correct | subset wrong): the decoupling watch — if this climbs
    for B, the model earns the reward without solving and B's filter stops selecting for solved (B → A).
    distinct_kept_subset_frac: fraction of DISTINCT committed subsets among the kept rollouts — a cheap
    diversity proxy; a fall (with flat/falling clean pass@k) flags SFT-on-own-outputs mode collapse."""
    flat_infos = [i for g in infos_by_group for i in g]
    flat_kept = [k for g in kept_flags_by_group for k in g]
    n = len(flat_infos)
    n_kept = sum(flat_kept)
    kept_infos = [i for i, k in zip(flat_infos, flat_kept) if k]
    subset_wrong = [i for i in flat_infos if not i.subset_correct]
    claimed_solved = [i for i in flat_infos if i.solved is True]  # model's own `Solved: yes` verdict
    distinct_kept = {tuple(i.subset_values) if i.subset_values else None for i in kept_infos}
    return {
        "keep_fraction": n_kept / n if n else 0.0,
        "n_generated": n,
        "n_kept": n_kept,
        "subset_correct_rate_all": sum(i.subset_correct for i in flat_infos) / n if n else 0.0,
        "subset_correct_rate_among_kept": (
            sum(i.subset_correct for i in kept_infos) / n_kept if n_kept else 0.0),
        "leakage_among_generated": (
            sum(i.addition_correct for i in subset_wrong) / len(subset_wrong) if subset_wrong else 0.0),
        # self-verify false positive = P(subset actually wrong | model committed `Solved: yes`); if this
        # climbs, the model is gaming its own verifier (distinct from decoupling). Diagnostic only.
        "self_verify_false_positive_rate": (
            sum(not i.subset_correct for i in claimed_solved) / len(claimed_solved)
            if claimed_solved else 0.0),
        "distinct_kept_subset_frac": (len(distinct_kept) / n_kept if n_kept else 0.0),
        "mean_kept_sample_tokens": (
            sum(i.n_sample_tokens for i in kept_infos) / n_kept if n_kept else 0.0),
    }


async def _sft_update(training_client, datums: list[tinker.Datum], lr: float,
                      tracker: CostTracker, model: str) -> tuple[float, int]:
    """One supervised cross-entropy forward-backward + optim step over the kept demonstrations.

    Returns (mean weighted NLL over the kept tokens, n_train_tokens). This is a proper supervised
    MLE step (maximize log-prob of the kept demonstrations), NOT the importance-sampling PG loss."""
    adam = tinker.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)
    fb_future = await training_client.forward_backward_async(datums, loss_fn="cross_entropy")
    optim_future = await training_client.optim_step_async(adam)
    fb_out = await fb_future.result_async()
    await optim_future.result_async()
    logprobs = [out["logprobs"] for out in fb_out.loss_fn_outputs]
    weights = [d.loss_fn_inputs["weights"] for d in datums]
    nll = compute_mean_nll(logprobs, weights)
    n_train_tokens = sum(d.model_input.length + 1 for d in datums)
    tracker.add_tinker_cost(model, train_tokens=n_train_tokens)
    return nll, n_train_tokens


async def run_sft_training(cfg: SFTTrainConfig, *, tracker: CostTracker, log_dir: Path,
                           eval_fn=None, verbosity: int = 1) -> list[SFTStepResult]:
    """Run the on-policy SFT loop for one seed. `eval_fn(step, sampler)` (async) is called on
    the eval cadence (same clean side-task eval the GRPO runs used). Mirrors lib.rl_train.run_training:
    each round generates on-policy rollouts, keeps the addition-correct ones, does one supervised CE
    update on the model's own kept spans, saves weights, and re-wraps a fresh sampling client."""
    assert cfg.filter in FILTER_MODES, f"filter must be one of {FILTER_MODES}, got {cfg.filter!r}"
    coupling = sft_condition(cfg.condition).coupling
    lr = cfg.lr if cfg.lr is not None else get_lr(config.MODEL)
    service = tinker.ServiceClient()
    training_client = await service.create_lora_training_client_async(
        base_model=config.MODEL, rank=cfg.lora_rank)
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / f"metrics_sft_{cfg.condition}_{cfg.tag}.jsonl"
    metrics_path.write_text("")

    # Step-0 sampling client = LoRA-zero weights (behaviourally the base model), same as rl_train.
    sampling_client = await service.create_sampling_client_async(base_model=config.MODEL)
    checkpoint_paths: list[str] = []
    results: list[SFTStepResult] = []

    if verbosity:
        print(f"=== SFT (filter={cfg.filter}): condition {cfg.condition} | group {cfg.group_size} x "
              f"batch {cfg.batch_size} x {cfg.n_steps} rounds | lr {lr:.2e} ({cfg.lr_schedule}) | "
              f"rank {cfg.lora_rank} ===", flush=True)

    for step in range(cfg.n_steps):
        t0 = time.time()
        problems = batch_problems(cfg, step)
        group_out = await asyncio.gather(*[
            rollout_group(sampling_client, coupling, p, cfg, step, g, tracker=tracker)
            for g, p in enumerate(problems)
        ])
        trajs_by_group = [go[0] for go in group_out]
        infos_by_group = [go[1] for go in group_out]
        cond = RL_CONDITIONS[cfg.condition]
        kept_flags_by_group = [[should_keep(cfg.filter, cond, i) for i in infos]
                               for infos in infos_by_group]

        # Flatten the filter-passed rollouts; optionally downsample (A-downsampled volume control).
        kept_trajs = [traj for trajs, keeps in zip(trajs_by_group, kept_flags_by_group)
                      for traj, keep in zip(trajs, keeps) if keep]
        n_kept = len(kept_trajs)
        if cfg.downsample_kept_frac is not None and kept_trajs:
            rng = random.Random(f"{cfg.seed}-{step}-downsample")
            n_target = max(1, round(len(kept_trajs) * cfg.downsample_kept_frac))
            kept_trajs = rng.sample(kept_trajs, min(n_target, len(kept_trajs)))

        # Build supervised data from the kept rollouts (masking reused from the GRPO path).
        datums: list[tinker.Datum] = []
        for traj in kept_trajs:
            datums.extend(trajectory_to_sft_datums(traj, reduction=cfg.reduction))

        step_lr = _effective_lr(lr, cfg.lr_schedule, step, cfg.n_steps, cfg.lr_min_frac)
        if datums:
            sft_nll, n_train_tokens = await _sft_update(training_client, datums, step_lr,
                                                        tracker, config.MODEL)
        else:  # no rollout passed the filter this round (policy unchanged) — rare; log it loudly
            sft_nll, n_train_tokens = float("nan"), 0
            if verbosity:
                print(f"[round {step}] WARNING: 0 rollouts kept — skipping the gradient step", flush=True)

        metrics = {
            "step": step,
            **{f"filter/{k}": v for k, v in _filter_diag(infos_by_group, kept_flags_by_group).items()},
            **{f"outcome/{k}": v for k, v in _outcome_diag(infos_by_group).items()},
            "sft/nll": sft_nll,
            "sft/n_datums": len(datums),
            "sft/n_trained_rollouts": len(kept_trajs),  # kept after any downsample (== n_kept if none)
            "optim/lr": step_lr,
            "n_train_tokens": n_train_tokens,
            "elapsed_s": time.time() - t0,
            "run_cost": tracker.run_cost,
        }

        # Save weights -> fresh on-policy sampling client for the next round.
        fut = await training_client.save_weights_for_sampler_async(
            name=f"sft_{cfg.tag}_{cfg.condition}_step{step}")
        ckpt = await fut.result_async()
        ckpt_path = ckpt.path
        checkpoint_paths.append(ckpt_path)
        sampling_client = await service.create_sampling_client_async(model_path=ckpt_path)

        results.append(SFTStepResult(
            step=step, condition=cfg.condition, infos_by_group=infos_by_group,
            kept_flags_by_group=kept_flags_by_group, metrics=metrics, n_kept=n_kept,
            n_train_tokens=n_train_tokens, checkpoint_path=ckpt_path))
        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        if verbosity:
            print(f"[round {step}] keep={metrics['filter/keep_fraction']:.3f} "
                  f"n_kept={n_kept} subset_all={metrics['outcome/subset_success_rate']:.3f} "
                  f"subset_kept={metrics['filter/subset_correct_rate_among_kept']:.3f} "
                  f"nll={sft_nll:.3f} ({metrics['elapsed_s']:.0f}s, ${tracker.run_cost:.2f})", flush=True)

        if eval_fn is not None and cfg.eval_every and (step + 1) % cfg.eval_every == 0:
            from lib.tinker_client import Sampler
            sampler = Sampler(sampling_client=sampling_client, sampler_id=ckpt_path, cache_enabled=False)
            await eval_fn(step, sampler)

    # Final eval — unless the cadence already evaluated the last round.
    last_step = cfg.n_steps - 1
    cadence_hit_last = cfg.eval_every and (last_step + 1) % cfg.eval_every == 0
    if eval_fn is not None and results and not cadence_hit_last:
        from lib.tinker_client import Sampler
        sampler = Sampler(sampling_client=sampling_client, sampler_id=results[-1].checkpoint_path,
                          cache_enabled=False)
        await eval_fn(last_step, sampler)

    # Cleanup intermediate checkpoints (keep the last if requested, plus any explicitly kept rounds).
    rest = service.create_rest_client()
    keep_idx = set(cfg.keep_checkpoint_steps)
    if cfg.keep_last_checkpoint and checkpoint_paths:
        keep_idx.add(len(checkpoint_paths) - 1)
    to_delete = [p for i, p in enumerate(checkpoint_paths) if i not in keep_idx]
    for path in to_delete:
        try:
            await rest.delete_checkpoint_from_tinker_path_async(path)
        except Exception as exc:  # noqa: BLE001 - cleanup best-effort
            print(f"  [cleanup] failed to delete {path}: {exc}", flush=True)
    if verbosity:
        print(f"deleted {len(to_delete)} intermediate checkpoint(s); "
              f"kept {'last' if cfg.keep_last_checkpoint else 'none'}", flush=True)
    return results


def _pool_diag(infos: list[RolloutInfo]) -> dict:
    """Diagnostics over the fixed base-rollout pool (computed once; constant across SFT rounds).

    `would_keep_fraction_correctness` = the fraction that the CORRECTNESS filter WOULD keep (addition-
    correct) — reported only as context; the unfiltered control keeps all of them. The rest mirror
    `_filter_diag`'s pool-level fields."""
    n = len(infos)
    subset_wrong = [i for i in infos if not i.subset_correct]
    distinct = {tuple(i.subset_values) if i.subset_values else None for i in infos}
    return {
        "n_pool": n,
        "would_keep_fraction_correctness": sum(i.addition_correct for i in infos) / n if n else 0.0,
        "subset_correct_rate_all": sum(i.subset_correct for i in infos) / n if n else 0.0,
        "leakage_among_generated": (
            sum(i.addition_correct for i in subset_wrong) / len(subset_wrong) if subset_wrong else 0.0),
        "distinct_subset_frac": len(distinct) / n if n else 0.0,
        "mean_sample_tokens": sum(i.n_sample_tokens for i in infos) / n if n else 0.0,
        "addition_parse_error_rate": sum(i.addition_parse_error for i in infos) / n if n else 0.0,
    }


async def sample_base_pool(cfg: SFTTrainConfig, coupling: Condition, *, cache, tracker: CostTracker,
                           assert_cached: bool = False, verbosity: int = 1):
    """Sample the fixed base-model coupling-prompt rollout pool ONCE (cached).

    Uses a frozen base-model sampling client, so each rollout is a deterministic function of its inputs
    and is cached by (prompt tokens, params, sampler_id) — re-runs are free / `--assert-cached`-checkable.
    Seeds depend on (problem_idx, rollout_idx) only (NOT on any training step), so the pool is identical
    across re-runs. Returns (trajs, infos), both flat lists of length n_pool_problems * group_size."""
    service = tinker.ServiceClient()
    base_client = await service.create_sampling_client_async(base_model=config.MODEL)
    problems = _load_coupled_problems(cfg.problem_offset, cfg.n_pool_problems)
    sampler_id = f"base_coupled_{cfg.condition}"

    async def one(problem_idx: int, rollout_idx: int):
        ss, add = problems[problem_idx]
        seed = cfg.seed + 1009 * problem_idx + rollout_idx
        return await rollout_coupled(
            base_client, ss, add, coupling, think_budget=cfg.think_budget, seed=seed,
            tracker=tracker, max_concurrency=cfg.max_concurrency, cache=cache, sampler_id=sampler_id,
            assert_cached=assert_cached)

    pairs = await asyncio.gather(*[
        one(p, r) for p in range(cfg.n_pool_problems) for r in range(cfg.group_size)])
    trajs = [t for t, _ in pairs]
    infos = [i for _, i in pairs]
    if verbosity:
        d = _pool_diag(infos)
        print(f"[base pool] {len(trajs)} rollouts | subset_correct={d['subset_correct_rate_all']:.3f} "
              f"would-keep(correctness)={d['would_keep_fraction_correctness']:.3f} "
              f"distinct_subset={d['distinct_subset_frac']:.3f} "
              f"mean_tokens={d['mean_sample_tokens']:.0f}", flush=True)
    return trajs, infos


async def run_base_rollout_sft(cfg: SFTTrainConfig, *, tracker: CostTracker, log_dir: Path, cache,
                               eval_fn=None, assert_cached: bool = False,
                               verbosity: int = 1) -> list[SFTStepResult]:
    """UNFILTERED base-model-rollout SFT (the base-rollout baseline).

    Sample a FIXED pool of base-model coupling-prompt rollouts ONCE (the rollout distribution stays the
    untrained base model's — no on-policy regeneration), then SFT on ALL of them (no correctness filter,
    no reward) over `n_steps` minibatch rounds (cycling through the pool). Each round evaluates the
    SFT-updated checkpoint's neutral subset-sum accuracy on the reserved pool (same eval as on-policy).
    Mirrors `run_sft_training` (same masking, same CE update, same eval/summary), differing only in that
    the rollouts come from the frozen base model, sampled once."""
    assert cfg.filter == "none", "run_base_rollout_sft is the UNFILTERED control (filter must be 'none')"
    coupling = sft_condition(cfg.condition).coupling
    lr = cfg.lr if cfg.lr is not None else get_lr(config.MODEL)
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / f"metrics_sft_{cfg.condition}_{cfg.tag}.jsonl"
    metrics_path.write_text("")

    # 1. Sample the fixed base pool once (cached), build the SFT datums for EVERY rollout.
    trajs, infos = await sample_base_pool(cfg, coupling, cache=cache, tracker=tracker,
                                          assert_cached=assert_cached, verbosity=verbosity)
    pool_datums = [trajectory_to_sft_datums(t, reduction=cfg.reduction) for t in trajs]
    pool_diag = _pool_diag(infos)

    service = tinker.ServiceClient()
    training_client = await service.create_lora_training_client_async(
        base_model=config.MODEL, rank=cfg.lora_rank)
    if verbosity:
        print(f"=== base-rollout SFT (filter=none): condition {cfg.condition} | pool {len(trajs)} "
              f"| minibatch {cfg.batch_size * cfg.group_size} x {cfg.n_steps} rounds | lr {lr:.2e} "
              f"({cfg.lr_schedule}) | rank {cfg.lora_rank} ===", flush=True)

    minibatch = cfg.batch_size * cfg.group_size
    order = list(range(len(trajs)))
    random.Random(f"{cfg.seed}-basepool-order").shuffle(order)

    checkpoint_paths: list[str] = []
    results: list[SFTStepResult] = []
    for step in range(cfg.n_steps):
        t0 = time.time()
        idxs = [order[(step * minibatch + j) % len(order)] for j in range(minibatch)]
        datums = [d for i in idxs for d in pool_datums[i]]
        step_lr = _effective_lr(lr, cfg.lr_schedule, step, cfg.n_steps, cfg.lr_min_frac)
        sft_nll, n_train_tokens = await _sft_update(training_client, datums, step_lr, tracker, config.MODEL)

        metrics = {
            "step": step,
            **{f"pool/{k}": v for k, v in pool_diag.items()},
            "filter/keep_fraction": 1.0,  # unfiltered by construction (sanity check the filter is off)
            "sft/nll": sft_nll,
            "sft/n_datums": len(datums),
            "sft/n_trained_rollouts": len(idxs),
            "optim/lr": step_lr,
            "n_train_tokens": n_train_tokens,
            "elapsed_s": time.time() - t0,
            "run_cost": tracker.run_cost,
        }
        fut = await training_client.save_weights_for_sampler_async(
            name=f"sft_{cfg.tag}_{cfg.condition}_step{step}")
        ckpt = await fut.result_async()
        ckpt_path = ckpt.path
        checkpoint_paths.append(ckpt_path)

        results.append(SFTStepResult(
            step=step, condition=cfg.condition, infos_by_group=[infos] if step == 0 else [],
            kept_flags_by_group=[[True] * len(infos)] if step == 0 else [],
            metrics=metrics, n_kept=len(idxs), n_train_tokens=n_train_tokens, checkpoint_path=ckpt_path))
        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        if verbosity:
            print(f"[round {step}] nll={sft_nll:.3f} n_datums={len(datums)} lr={step_lr:.2e} "
                  f"({metrics['elapsed_s']:.0f}s, ${tracker.run_cost:.2f})", flush=True)

        if eval_fn is not None and cfg.eval_every and (step + 1) % cfg.eval_every == 0:
            from lib.tinker_client import Sampler
            eval_client = await service.create_sampling_client_async(model_path=ckpt_path)
            sampler = Sampler(sampling_client=eval_client, sampler_id=ckpt_path, cache_enabled=False)
            await eval_fn(step, sampler)

    last_step = cfg.n_steps - 1
    cadence_hit_last = cfg.eval_every and (last_step + 1) % cfg.eval_every == 0
    if eval_fn is not None and results and not cadence_hit_last:
        from lib.tinker_client import Sampler
        eval_client = await service.create_sampling_client_async(model_path=results[-1].checkpoint_path)
        sampler = Sampler(sampling_client=eval_client, sampler_id=results[-1].checkpoint_path,
                          cache_enabled=False)
        await eval_fn(last_step, sampler)

    # Cleanup intermediate checkpoints (keep the last if requested, plus any explicitly kept rounds).
    rest = service.create_rest_client()
    keep_idx = set(cfg.keep_checkpoint_steps)
    if cfg.keep_last_checkpoint and checkpoint_paths:
        keep_idx.add(len(checkpoint_paths) - 1)
    to_delete = [p for i, p in enumerate(checkpoint_paths) if i not in keep_idx]
    for path in to_delete:
        try:
            await rest.delete_checkpoint_from_tinker_path_async(path)
        except Exception as exc:  # noqa: BLE001 - cleanup best-effort
            print(f"  [cleanup] failed to delete {path}: {exc}", flush=True)
    if verbosity:
        print(f"deleted {len(to_delete)} intermediate checkpoint(s)", flush=True)
    return results
