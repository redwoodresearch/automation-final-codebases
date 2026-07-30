"""Thin GRPO training loop for the self-steering experiment (conditions A/B/C/D).

Route (b): each step samples
`group_size` on-policy trajectories per problem with `lib.rl_rollout` (per-phase budgets, cues
masked), computes per-condition GRPO rewards (`lib.rl_conditions`), then reuses the cookbook's
`compute_advantages` + `assemble_training_data` + `train_step(loss_fn="importance_sampling")` +
`optim_step` verbatim (the load-bearing masking/loss code). After each step it re-wraps a sampling
client from the fresh weights (on-policy) and, on the eval cadence, calls the eval harness.

This is deliberately NOT the cookbook's `train.Config` CLI: that path builds one policy with a single
fixed `max_tokens` for every step, which can't express our search/verify/subset/solved/answer budgets.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import attrs
import tinker

from cost_tracker import CostTracker
from file_cache import FileCache
from lib import config
from lib.pools import load_addition_pool, load_subset_sum_pool
from lib.rl_conditions import RL_CONDITIONS, RLCondition, group_rewards, raw_rewards
from lib.rl_rollout import RolloutInfo, rollout_budget_forced, rollout_coupled
from lib.tinker_client import Sampler

from tinker_cookbook.hyperparam_utils import get_lr
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
from tinker_cookbook.rl.metrics import compute_kl_sample_train
from tinker_cookbook.rl.train import train_step
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup
from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier


@attrs.frozen
class RLTrainConfig:
    condition: str = "b"                 # 'a' / 'b' / 'c' / 'd'
    group_size: int = 8
    batch_size: int = 8                  # distinct problems per step
    n_steps: int = 3
    lr: float | None = None              # None -> hyperparam_utils.get_lr(model)
    lr_schedule: str = "constant"        # 'constant' / 'linear' / 'cosine' (decay over n_steps)
    lr_min_frac: float = 0.0             # floor on the schedule multiplier (decay-to-floor; erosion probe)
    lora_rank: int = 32
    think_budget: int = config.SUBSET_SUM_THINK_BUDGET
    max_concurrency: int = 128
    seed: int = 0
    problem_offset: int = 500            # train-pool start index (disjoint from faithfulness 0-399)
    fixed_problems: bool = False         # reuse the same batch each step (tiny-batch overfit sanity)
    # Eval cadence (harness). eval_every=0 -> only at the end.
    eval_every: int = 0
    n_eval_coupling: int = 40
    n_eval_clean: int = 80
    n_eval_samples: int = 8
    keep_last_checkpoint: bool = True
    keep_checkpoint_steps: tuple = ()    # extra step indices whose checkpoints are NOT deleted
    tag: str = "smoke"


@dataclass
class StepResult:
    step: int
    condition: str
    rewards_by_group: list[list[float]]      # final (possibly permuted) rewards
    raw_rewards_by_group: list[list[float]]  # pre-permutation rewards (== final for A/B/C)
    infos_by_group: list[list[RolloutInfo]]
    metrics: dict
    n_train_tokens: int
    checkpoint_path: str | None = None


def _load_coupled_problems(offset: int, n: int):
    ss = load_subset_sum_pool("train")
    add = load_addition_pool("train")
    return [(ss[(offset + i) % len(ss)], add[(offset + i) % len(add)]) for i in range(n)]


def _load_neutral_problems(offset: int, n: int):
    ss = load_subset_sum_pool("train")
    return [ss[(offset + i) % len(ss)] for i in range(n)]


def batch_problems(condition: RLCondition, cfg: RLTrainConfig, step: int) -> list:
    """One problem per group in this step's batch (cycled through the train pool)."""
    start = cfg.problem_offset if cfg.fixed_problems else cfg.problem_offset + step * cfg.batch_size
    if condition.kind == "coupled":
        return _load_coupled_problems(start, cfg.batch_size)
    return _load_neutral_problems(start, cfg.batch_size)


async def rollout_one_group(
    sampling_client, condition: RLCondition, problem, cfg: RLTrainConfig, step: int, group_idx: int,
    *, tracker: CostTracker | None,
) -> tuple[TrajectoryGroup, list[RolloutInfo], list[float], list[float]]:
    """Sample group_size on-policy trajectories for one problem; assign per-condition GRPO rewards."""
    async def one(rollout_idx: int) -> tuple[Trajectory, RolloutInfo]:
        seed = cfg.seed + 1_000_003 * step + 1009 * group_idx + rollout_idx
        if condition.kind == "coupled":
            ss, add = problem
            return await rollout_coupled(sampling_client, ss, add, condition.coupling,
                                         think_budget=cfg.think_budget, seed=seed, tracker=tracker,
                                         max_concurrency=cfg.max_concurrency)
        return await rollout_budget_forced(sampling_client, problem, think_budget=cfg.think_budget,
                                           seed=seed, tracker=tracker, max_concurrency=cfg.max_concurrency)

    pairs = await asyncio.gather(*[one(r) for r in range(cfg.group_size)])
    trajs = [t for t, _ in pairs]
    infos = [i for _, i in pairs]
    final = group_rewards(condition, infos, step=step, group_idx=group_idx, seed=cfg.seed)
    raw = raw_rewards(condition, infos)
    # The scalar trajectory reward is the group-level reward (per-transition step rewards stay 0),
    # so get_total_rewards() == final[i]. For D this is the permuted assignment (the natural
    # compute_group_rewards hook). compute_advantages then mean-centers within the group.
    tg = TrajectoryGroup(trajectories_G=trajs, final_rewards_G=[float(r) for r in final],
                         metrics_G=[{} for _ in trajs])
    return tg, infos, final, raw


def _group_diag(rewards_by_group: list[list[float]]) -> dict:
    """GRPO reward diagnostics (Bernoulli p(1-p))."""
    ps = [sum(g) / len(g) for g in rewards_by_group if g]
    n = len(ps)
    if n == 0:
        return {}
    dead = sum(1 for p in ps if p in (0.0, 1.0))
    return {
        "reward_mean": sum(ps) / n,
        "mean_within_group_variance": sum(p * (1 - p) for p in ps) / n,
        "dead_group_fraction": dead / n,
        "n_groups": n,
    }


def _reinforced_token_mass(infos_by_group, rewards_by_group) -> dict:
    """Mean sampled-token count of reward-1 vs reward-0 rollouts (audits B-vs-D length matching)."""
    pos_tok, neg_tok = [], []
    for infos, rews in zip(infos_by_group, rewards_by_group):
        for info, r in zip(infos, rews):
            (pos_tok if r >= 0.5 else neg_tok).append(info.n_sample_tokens)
    return {
        "mean_tokens_reward1": (sum(pos_tok) / len(pos_tok)) if pos_tok else 0.0,
        "mean_tokens_reward0": (sum(neg_tok) / len(neg_tok)) if neg_tok else 0.0,
        "n_reward1": len(pos_tok), "n_reward0": len(neg_tok),
    }


def _outcome_diag(infos_by_group) -> dict:
    flat = [i for g in infos_by_group for i in g]
    n = len(flat)
    return {
        "subset_success_rate": sum(i.subset_correct for i in flat) / n,
        "addition_correct_rate": sum(i.addition_correct for i in flat) / n,
        "forced_frac": sum(i.forced for i in flat) / n,
        "addition_parse_error_rate": sum(i.addition_parse_error for i in flat) / n,
        "mean_sample_tokens": sum(i.n_sample_tokens for i in flat) / n,
    }


async def _train_on_groups(training_client, groups: list[TrajectoryGroup], lr: float,
                           tracker: CostTracker, model: str) -> tuple[dict, int]:
    advantages_P = compute_advantages(groups)
    data_D, _meta = assemble_training_data(groups, advantages_P)
    metrics: dict = {}
    training_logprobs_D = await train_step(
        data_D, training_client, learning_rate=lr, num_substeps=1,
        loss_fn="importance_sampling", metrics=metrics,
    )
    metrics.update(compute_kl_sample_train(data_D, training_logprobs_D))
    n_train_tokens = sum(d.model_input.length + 1 for d in data_D)
    tracker.add_tinker_cost(model, train_tokens=n_train_tokens)
    # Advantage-shape sanity (mean-centered within group): record max |group-mean advantage|.
    max_abs_group_mean = 0.0
    for adv_G in advantages_P:
        max_abs_group_mean = max(max_abs_group_mean, float(adv_G.mean().abs()))
    metrics["adv/max_abs_group_mean"] = max_abs_group_mean
    metrics["adv/n_datums"] = len(data_D)
    metrics["optim/lr"] = lr
    return metrics, n_train_tokens


def _effective_lr(base_lr: float, schedule: str, step: int, n_steps: int,
                  lr_min_frac: float = 0.0) -> float:
    """Base LR times the schedule's decay multiplier for this step (constant -> unchanged).

    `lr_min_frac` floors the multiplier so a decaying schedule levels off at that fraction of the peak
    instead of going to ~0. The erosion probe uses this (decay to a live floor, then hold) so a
    long-horizon leakage rise is a genuine decoupling read, not an artifact of the LR going to zero."""
    return base_lr * max(lr_min_frac, compute_schedule_lr_multiplier(schedule, step, n_steps))


async def run_training(cfg: RLTrainConfig, *, tracker: CostTracker, log_dir: Path,
                       eval_fn=None, verbosity: int = 1) -> list[StepResult]:
    """Run the smoke training loop. `eval_fn(step, sampler)` (async) is called on the eval cadence.

    Returns per-step results. Checkpoints are deleted at the end except the last (if keep_last)."""
    condition = RL_CONDITIONS[cfg.condition]
    lr = cfg.lr if cfg.lr is not None else get_lr(config.MODEL)
    service = tinker.ServiceClient()
    training_client = await service.create_lora_training_client_async(
        base_model=config.MODEL, rank=cfg.lora_rank)
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / f"metrics_{cfg.condition}_{cfg.tag}.jsonl"
    metrics_path.write_text("")

    # Step-0 sampling client = LoRA-zero weights (behaviourally the base model).
    sampling_client = await service.create_sampling_client_async(base_model=config.MODEL)
    checkpoint_paths: list[str] = []
    results: list[StepResult] = []

    if verbosity:
        print(f"=== RL: condition {cfg.condition} | group_size {cfg.group_size} x "
              f"batch {cfg.batch_size} x {cfg.n_steps} steps | lr {lr:.2e} ({cfg.lr_schedule}) | "
              f"rank {cfg.lora_rank} ===", flush=True)

    for step in range(cfg.n_steps):
        t0 = time.time()
        problems = batch_problems(condition, cfg, step)
        group_out = await asyncio.gather(*[
            rollout_one_group(sampling_client, condition, p, cfg, step, g, tracker=tracker)
            for g, p in enumerate(problems)
        ])
        groups = [go[0] for go in group_out]
        infos_by_group = [go[1] for go in group_out]
        rewards_by_group = [go[2] for go in group_out]
        raw_by_group = [go[3] for go in group_out]

        step_lr = _effective_lr(lr, cfg.lr_schedule, step, cfg.n_steps, cfg.lr_min_frac)
        train_metrics, n_train_tokens = await _train_on_groups(
            training_client, groups, step_lr, tracker, config.MODEL)

        metrics = {
            "step": step,
            **{f"reward/{k}": v for k, v in _group_diag(rewards_by_group).items()},
            **{f"raw_reward/{k}": v for k, v in _group_diag(raw_by_group).items()},
            **{f"outcome/{k}": v for k, v in _outcome_diag(infos_by_group).items()},
            **{f"tokmass/{k}": v for k, v in _reinforced_token_mass(infos_by_group, rewards_by_group).items()},
            **train_metrics,
            "elapsed_s": time.time() - t0,
            "run_cost": tracker.run_cost,
        }

        # Save weights -> fresh on-policy sampling client for the next step.
        fut = await training_client.save_weights_for_sampler_async(name=f"{cfg.tag}_{cfg.condition}_step{step}")
        ckpt = await fut.result_async()
        ckpt_path = ckpt.path
        checkpoint_paths.append(ckpt_path)
        sampling_client = await service.create_sampling_client_async(model_path=ckpt_path)

        results.append(StepResult(
            step=step, condition=cfg.condition, rewards_by_group=rewards_by_group,
            raw_rewards_by_group=raw_by_group, infos_by_group=infos_by_group, metrics=metrics,
            n_train_tokens=n_train_tokens, checkpoint_path=ckpt_path))
        with open(metrics_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        if verbosity:
            rd = metrics.get("reward/mean_within_group_variance", 0.0)
            print(f"[step {step}] reward_mean={metrics.get('reward/reward_mean', 0):.3f} "
                  f"within_group_var={rd:.3f} dead={metrics.get('reward/dead_group_fraction', 0):.2f} "
                  f"subset={metrics['outcome/subset_success_rate']:.3f} "
                  f"entropy={metrics.get('optim/entropy', float('nan')):.3f} "
                  f"kl={metrics.get('optim/kl_sample_train_v1', float('nan')):.4f} "
                  f"({metrics['elapsed_s']:.0f}s, ${tracker.run_cost:.2f})", flush=True)

        if eval_fn is not None and cfg.eval_every and (step + 1) % cfg.eval_every == 0:
            sampler = Sampler(sampling_client=sampling_client, sampler_id=ckpt_path, cache_enabled=False)
            await eval_fn(step, sampler)

    # Final eval — unless the cadence already evaluated the last step (avoid a redundant heavy eval).
    last_step = cfg.n_steps - 1
    cadence_hit_last = cfg.eval_every and (last_step + 1) % cfg.eval_every == 0
    if eval_fn is not None and results and not cadence_hit_last:
        sampler = Sampler(sampling_client=sampling_client, sampler_id=results[-1].checkpoint_path,
                          cache_enabled=False)
        await eval_fn(last_step, sampler)

    # Cleanup intermediate checkpoints (keep the last if requested, plus any explicitly kept steps).
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
