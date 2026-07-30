"""Single-seed GRPO pilot / LR-sweep driver for one condition (A/B/C/D).

This is the per-condition RL training entry point. It reuses `lib.rl_train.run_training` (route b) and adds
the full per-checkpoint eval/diagnostics protocol: at each eval step it logs, per
condition, subset-sum accuracy under BOTH the neutral prompt (the cross-condition headline metric,
on the reserved eval pool) and the condition's OWN training prompt, plus coupling precision,
self-verification accuracy (P(`Solved:` == external verifier)), leakage, reward/dead-group
diagnostics, and it captures four-cell example transcripts across training.

Two uses (one driver):
  - Pilot:   .venv/bin/python run_rl_pilot.py --condition b --n_steps 36 --group_size 16 \
                 --batch_size 8 --lr 1e-4 --lr_schedule linear --eval_every 6 --tag pilot
  - LR sweep: .venv/bin/python run_rl_pilot.py --condition c --fixed_problems --no_eval \
                 --n_steps 14 --group_size 16 --batch_size 4 --lr 3e-4 --tag lrsweep
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import attrs

from cost_tracker import CostTracker
from file_cache import FileCache
from lib.eval_harness import (
    checkpoint_report, clean_side_task_eval, coupled_eval_row, neutral_eval_row,
    render_four_cell_markdown, scaffold_eval_row,
)
from lib.pools import load_coupling_split, load_subset_sum_pool
from lib.rl_conditions import RL_CONDITIONS
from lib.rl_train import RLTrainConfig, run_training

RESULTS_DIR = Path(__file__).resolve().parent / "results"
LOG_DIR = RESULTS_DIR / "rl_pilot"
CACHE = FileCache("./file_cache_dir/")


def build_eval_fn(condition_name: str, args, tracker, out_base: str):
    """An async eval_fn(step, sampler) that runs the harness and appends a flat metrics row.

    Coupled conditions (A/B/D) get the full checkpoint_report (both-prompt subset accuracy + the
    faithfulness/self-verification breakdown + four-cell transcripts); C gets the neutral clean eval.
    Uses the reserved 400-instance EVAL pool for the neutral (headline) metric and the held-out
    coupling split for the coupled-prompt metric — both fixed across steps/conditions."""
    coupling_pairs = load_coupling_split("heldout")[: args.n_eval_coupling]
    clean_instances = load_subset_sum_pool("eval")[: args.n_eval_clean]
    condition = RL_CONDITIONS[condition_name]
    eval_path = RESULTS_DIR / f"{out_base}_evals.jsonl"

    # C-scaffold (subset reward under B's forced flow) is a coupled-flow condition but its faithfulness
    # columns are meaningless (its "answer" restates the subset); it gets the scaffold row instead.
    is_scaffold_reward = condition.coupling is not None and condition.reward_attr == "subset_correct"

    async def eval_fn(step, sampler):
        t0 = time.time()
        if condition.coupling is not None:
            out = await checkpoint_report(
                coupling_pairs, clean_instances, cache=CACHE, sampler=sampler,
                condition=condition.coupling, n_coupling_samples=args.n_eval_samples,
                n_clean_samples=args.n_eval_samples, tracker=tracker,
                max_concurrency=args.concurrency, examples_per_cell=3, verbosity=0)
            row = scaffold_eval_row(out["metrics"], step) if is_scaffold_reward \
                else coupled_eval_row(out["metrics"], step)
            (RESULTS_DIR / f"{out_base}_step{step}_four_cell.md").write_text(
                render_four_cell_markdown(out["examples"], f"{condition_name} step {step}"))
            (RESULTS_DIR / f"{out_base}_step{step}_metrics.json").write_text(
                json.dumps(out["metrics"], indent=2))
        else:
            clean = await clean_side_task_eval(
                clean_instances, cache=CACHE, sampler=sampler, n_samples=args.n_eval_samples,
                tracker=tracker, max_concurrency=args.concurrency, verbosity=0)
            row = neutral_eval_row(clean, step)
        row["elapsed_s"] = time.time() - t0
        with open(eval_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  [eval step {step}] " + "  ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in row.items() if k != "elapsed_s") + f"  ({row['elapsed_s']:.0f}s)", flush=True)

    return eval_fn


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["a", "b", "c", "d", "cs"], required=True)
    ap.add_argument("--group_size", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_schedule", choices=["constant", "linear", "cosine"], default="constant")
    ap.add_argument("--lr_min_frac", type=float, default=0.0,
                    help="floor on the LR-schedule multiplier (decay-to-floor; erosion probe)")
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--think_budget", type=int, default=None)
    ap.add_argument("--problem_offset", type=int, default=500)
    ap.add_argument("--fixed_problems", action="store_true", help="reuse same batch each step (LR sweep)")
    ap.add_argument("--eval_every", type=int, default=6, help="0 -> only final eval")
    ap.add_argument("--no_eval", action="store_true", help="skip harness evals entirely (LR sweep)")
    ap.add_argument("--n_eval_coupling", type=int, default=40)
    ap.add_argument("--n_eval_clean", type=int, default=100)
    ap.add_argument("--n_eval_samples", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep_steps", default="", help="comma-separated extra step indices to keep as checkpoints")
    ap.add_argument("--tag", default="pilot")
    args = ap.parse_args()
    keep_steps = tuple(int(s) for s in args.keep_steps.split(",") if s.strip())

    kwargs = dict(condition=args.condition, group_size=args.group_size, batch_size=args.batch_size,
                  n_steps=args.n_steps, lr=args.lr, lr_schedule=args.lr_schedule,
                  lr_min_frac=args.lr_min_frac,
                  lora_rank=args.lora_rank, problem_offset=args.problem_offset,
                  fixed_problems=args.fixed_problems, eval_every=args.eval_every,
                  n_eval_coupling=args.n_eval_coupling, n_eval_clean=args.n_eval_clean,
                  n_eval_samples=args.n_eval_samples, max_concurrency=args.concurrency,
                  seed=args.seed, keep_checkpoint_steps=keep_steps, tag=args.tag)
    if args.think_budget is not None:
        kwargs["think_budget"] = args.think_budget
    cfg = RLTrainConfig(**kwargs)

    tracker = CostTracker(Path("total_cost.jsonl"), run_description=f"rl_pilot_{args.condition}_{args.tag}")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_base = f"rl_pilot_{args.condition}_{args.tag}_{stamp}"

    # Provenance: record the full config (incl. problem_offset/seed) + git hash at run START, so a run's
    # train slice is machine-identifiable even before it finishes (the summary.json is only written at the
    # end). This is the reproducibility record the per-step metrics rows intentionally stay lean of.
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        git_hash = None
    (RESULTS_DIR / f"{out_base}_config.json").write_text(json.dumps(
        {"git_hash": git_hash, "stamp": stamp, "config": attrs.asdict(cfg)}, indent=2, default=str))

    eval_fn = None if args.no_eval else build_eval_fn(args.condition, args, tracker, out_base)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results = await run_training(cfg, tracker=tracker, log_dir=LOG_DIR, eval_fn=eval_fn)

    summary = {"condition": args.condition, "config": attrs.asdict(cfg),
               "final_checkpoint": results[-1].checkpoint_path if results else None,
               "kept_checkpoints": {r.step: r.checkpoint_path for r in results
                                    if r.step in keep_steps or r.step == cfg.n_steps - 1},
               "steps": [r.metrics for r in results]}
    (RESULTS_DIR / f"{out_base}_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    # Save a small sample of raw rollouts per step for hand inspection.
    sample_rollouts = []
    for r in results:
        for g_idx, infos in enumerate(r.infos_by_group[:2]):
            for s_idx, info in enumerate(infos[:2]):
                sample_rollouts.append({"step": r.step, "group": g_idx, "rollout": s_idx,
                                        "reward": r.rewards_by_group[g_idx][s_idx],
                                        "raw_reward": r.raw_rewards_by_group[g_idx][s_idx],
                                        **{k: v for k, v in asdict(info).items()}})
    with open(RESULTS_DIR / f"{out_base}_rollout_samples.jsonl", "w") as f:
        for rec in sample_rollouts:
            f.write(json.dumps(rec, default=str) + "\n")
    print(f"\nwrote {out_base}_summary.json (final ckpt: {summary['final_checkpoint']})")
    print(f"Run cost: ${tracker.run_cost:.3f}  |  cumulative: ${tracker.total_cost():.3f}")


if __name__ == "__main__":
    asyncio.run(main())
