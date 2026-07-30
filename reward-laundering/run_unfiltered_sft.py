"""The two UNFILTERED SFT baselines.

SFT the model on rollouts from the reward-laundering prompt (condition B) WITHOUT any correctness
filter — imitate EVERY rollout regardless of whether the subset or addition was right, with NO reward
signal at all. This is the null control for whether the reward-laundering side-task gain is genuine RL
credit assignment vs. mere imitation of self-generated, on-distribution reasoning.

Two variants (same eval, same reserved 400-pool metric as the GRPO headline):
  --variant onpolicy : regenerate rollouts from the CURRENT (SFT-updated) model each round, SFT on ALL
                       of them, re-wrap a fresh sampling client, repeat (run_sft_training, filter=none).
  --variant base     : sample rollouts from the untrained BASE model ONCE (cached), then SFT on ALL of
                       them offline (rollout distribution stays the base model's; run_base_rollout_sft).

Both write a GRPO-schema summary under results/sft/ (so run_endpoint_evals.py discovers them). Distinct
tags (unf_onpolicy_* / unf_base_*) keep the two variants apart at eval/analysis time.

Examples:
  .venv/bin/python run_unfiltered_sft.py --variant onpolicy --seed 0 --problem_offset 500 \
      --n_steps 20 --eval_every 5 --keep_steps 9,14,19 --tag unf_onpolicy_seed0
  .venv/bin/python run_unfiltered_sft.py --variant base --seed 0 --problem_offset 500 \
      --n_steps 20 --n_pool_problems 80 --eval_every 5 --keep_steps 9,14,19 --tag unf_base_seed0
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
from lib.pools import load_subset_sum_pool
from lib.sft_eval import build_neutral_eval_fn
from lib.sft_train import SFTTrainConfig, run_base_rollout_sft, run_sft_training

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SFT_DIR = RESULTS_DIR / "sft"
LOG_DIR = SFT_DIR / "logs"
CACHE = FileCache("./file_cache_dir/")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["onpolicy", "base"], required=True)
    ap.add_argument("--condition", choices=["a", "b"], default="b",
                    help="coupling prompt for the rollouts (default b = the reward-laundering prompt)")
    ap.add_argument("--group_size", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_steps", type=int, default=20, help="SFT rounds")
    ap.add_argument("--n_pool_problems", type=int, default=80,
                    help="base variant only: distinct problems in the fixed base-rollout pool")
    ap.add_argument("--lr", type=float, default=1e-4, help="prereg fallback 1e-4 (the GRPO-tuned value)")
    ap.add_argument("--lr_schedule", choices=["constant", "linear", "cosine"], default="linear")
    ap.add_argument("--reduction", choices=["mean", "sum"], default="mean")
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--think_budget", type=int, default=None)
    ap.add_argument("--problem_offset", type=int, default=500, help="per seed use 500 + 240*seed")
    ap.add_argument("--eval_every", type=int, default=5, help="0 -> only final eval")
    ap.add_argument("--no_eval", action="store_true")
    ap.add_argument("--n_eval_clean", type=int, default=100)
    ap.add_argument("--n_eval_samples", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep_steps", default="", help="comma-separated kept checkpoint rounds")
    ap.add_argument("--assert_cached", action="store_true", help="base variant: verify pool is cached")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    keep_steps = tuple(int(s) for s in args.keep_steps.split(",") if s.strip())

    kwargs = dict(condition=args.condition, filter="none", group_size=args.group_size,
                  batch_size=args.batch_size, n_steps=args.n_steps, n_pool_problems=args.n_pool_problems,
                  lr=args.lr, lr_schedule=args.lr_schedule, reduction=args.reduction,
                  lora_rank=args.lora_rank, problem_offset=args.problem_offset,
                  eval_every=args.eval_every, n_eval_clean=args.n_eval_clean,
                  n_eval_samples=args.n_eval_samples, max_concurrency=args.concurrency,
                  seed=args.seed, keep_checkpoint_steps=keep_steps, tag=args.tag)
    if args.think_budget is not None:
        kwargs["think_budget"] = args.think_budget
    cfg = SFTTrainConfig(**kwargs)

    tracker = CostTracker(Path("total_cost.jsonl"),
                          run_description=f"unfiltered_sft_{args.variant}_{args.tag}")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_base = f"sft_{args.condition}_{args.tag}_{stamp}"
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        git_hash = None
    (SFT_DIR / f"{out_base}_config.json").write_text(json.dumps(
        {"git_hash": git_hash, "stamp": stamp, "variant": args.variant, "config": attrs.asdict(cfg)},
        indent=2, default=str))

    eval_fn = None
    if not args.no_eval:
        clean = load_subset_sum_pool("eval")[: args.n_eval_clean]
        eval_fn = build_neutral_eval_fn(
            clean_instances=clean, eval_path=SFT_DIR / f"{out_base}_evals.jsonl",
            n_eval_samples=args.n_eval_samples, concurrency=args.concurrency, cache=CACHE,
            tracker=tracker)

    if args.variant == "onpolicy":
        results = await run_sft_training(cfg, tracker=tracker, log_dir=LOG_DIR, eval_fn=eval_fn)
    else:
        results = await run_base_rollout_sft(cfg, tracker=tracker, log_dir=LOG_DIR, cache=CACHE,
                                             eval_fn=eval_fn, assert_cached=args.assert_cached)

    summary = {"condition": args.condition, "variant": args.variant, "config": attrs.asdict(cfg),
               "final_checkpoint": results[-1].checkpoint_path if results else None,
               "kept_checkpoints": {r.step: r.checkpoint_path for r in results
                                    if r.step in keep_steps or r.step == cfg.n_steps - 1},
               "steps": [r.metrics for r in results]}
    (SFT_DIR / f"{out_base}_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Raw rollout samples for hand inspection (kept flag == True everywhere by construction).
    sample_rollouts = []
    for r in results:
        for g_idx, infos in enumerate(r.infos_by_group[:2]):
            for s_idx, info in enumerate(infos[:6]):
                sample_rollouts.append({"step": r.step, "group": g_idx, "rollout": s_idx,
                                        "kept": r.kept_flags_by_group[g_idx][s_idx],
                                        **{k: v for k, v in asdict(info).items()}})
    with open(SFT_DIR / f"{out_base}_rollout_samples.jsonl", "w") as f:
        for rec in sample_rollouts:
            f.write(json.dumps(rec, default=str) + "\n")
    print(f"\nwrote {out_base}_summary.json (final ckpt: {summary['final_checkpoint']})")
    print(f"Run cost: ${tracker.run_cost:.3f}  |  cumulative: ${tracker.total_cost():.3f}")


if __name__ == "__main__":
    asyncio.run(main())
