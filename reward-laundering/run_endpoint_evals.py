"""Decision-rule endpoint evals: neutral subset-sum accuracy of every (condition, seed) checkpoint
on a large fixed eval set, paired against the matched base.

Discovers the kept late checkpoints from the training summaries (`results/rl_pilot/..._summary.json`
via `results/rl_pilot_<cond>_<tag>_*_summary.json`), evaluates the base once and each checkpoint at
`--n_instances x --n_samples` on the reserved 400-pool eval instances, and writes one JSONL row per
(condition, seed, step) with accuracy, paired Δ-vs-base, and per-problem rates. Checkpoint samples are
cached by tinker path so re-runs are free (`--assert-cached` verifies no new sampling).

The verdict aggregation (`analyze_multiseed.py`) reads these rows and applies the locked decision rule.

Usage:
  .venv/bin/python run_endpoint_evals.py --tags seed0 --n_instances 200 --n_samples 8
  .venv/bin/python run_endpoint_evals.py --tags seed0,seed1,seed2,seed3,seed4 \
       --steps 23,29 --n_instances 200 --n_samples 8 --out multiseed
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import time
from pathlib import Path

import tinker

from cost_tracker import CostTracker
from file_cache import FileCache
from lib.endpoint_eval import (
    checkpoint_sampler, neutral_eval_records, pass_rates_from_records, passk_from_records,
    paired_stats, token_stats_from_records,
)
from lib.pools import load_subset_sum_pool
from lib.tinker_client import BASE_SAMPLER

CACHE = FileCache("./file_cache_dir/")
RESULTS = Path(__file__).resolve().parent / "results"


def discover_checkpoints(tags: list[str], steps: list[int] | None,
                         conditions: list[str] | None = None, source: str = "all") -> list[dict]:
    """Kept late checkpoints for the requested tags: [{condition, seed, step, path, tag}].

    If `conditions` is given, only those conditions are returned, ordered to match `conditions` (so a
    caller can prioritise the gate conditions B/D ahead of the descriptive C when throughput is tight).

    Discovers the GRPO summaries (`rl_pilot_*_summary.json`) and/or the on-policy SFT summaries
    (`sft/sft_*_summary.json`), which share the schema (config.condition/seed/tag + kept_checkpoints)
    and the eval path. `source` restricts which family is discovered: "all" (both), "sft", or "grpo".
    IMPORTANT: the two families can COLLIDE on `tag` — e.g. the GRPO multiseed runs use tags
    `seed0..seed4` with conditions a/b, and an SFT run could reuse the same `seed<s>` tag, so a
    `--tags seed0..` filter alone would match BOTH and conflate GRPO+SFT under the same condition label.
    Pass `source="sft"` (or "grpo") to disambiguate when tags overlap."""
    assert source in ("all", "sft", "grpo"), f"source must be all/sft/grpo, got {source!r}"
    grpo_paths = sorted(glob.glob(str(RESULTS / "rl_pilot_*_summary.json"))) if source in ("all", "grpo") else []
    sft_paths = sorted(glob.glob(str(RESULTS / "sft" / "sft_*_summary.json"))) if source in ("all", "sft") else []
    rows: list[dict] = []
    summary_paths = grpo_paths + sft_paths
    for summ_path in summary_paths:
        summ = json.loads(Path(summ_path).read_text())
        cfg = summ.get("config", {})
        tag = cfg.get("tag")
        if tag not in tags:
            continue
        cond, seed = cfg.get("condition"), cfg.get("seed")
        if conditions is not None and cond not in conditions:
            continue
        kept = dict(summ.get("kept_checkpoints", {}))
        # Defensive fallback: if the summary recorded no kept checkpoints (older summary code),
        # use the final checkpoint at step n_steps-1 so discovery never silently returns nothing.
        if not kept and summ.get("final_checkpoint"):
            kept = {str(cfg.get("n_steps", 0) - 1): summ["final_checkpoint"]}
        for step_str, path in kept.items():
            step = int(step_str)
            if steps is not None and step not in steps:
                continue
            if not path:
                continue
            rows.append({"condition": cond, "seed": seed, "step": step, "path": path, "tag": tag})
    if conditions is not None:
        order = {c: i for i, c in enumerate(conditions)}
        rows.sort(key=lambda r: (order.get(r["condition"], 99), r["seed"], r["step"]))
    return rows


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="comma-separated run tags, e.g. seed0,seed1")
    ap.add_argument("--steps", default=None, help="comma-separated kept steps to eval (default: all kept)")
    ap.add_argument("--n_instances", type=int, default=200)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--assert_cached", action="store_true")
    ap.add_argument("--conditions", default=None,
                    help="comma-separated condition subset+order (e.g. b,d,a,c to eval the gate "
                         "conditions before the descriptive C); default all, order a,b,c,d")
    ap.add_argument("--source", choices=["all", "sft", "grpo"], default="all",
                    help="restrict discovery to SFT or GRPO summaries when tags collide "
                         "(use --source sft for SFT runs that reuse a GRPO seed<s> tag)")
    ap.add_argument("--out", default=None, help="output tag (default: first --tags value)")
    args = ap.parse_args()

    tags = [t for t in args.tags.split(",") if t.strip()]
    steps = [int(s) for s in args.steps.split(",")] if args.steps else None
    conditions = [c for c in args.conditions.split(",") if c.strip()] if args.conditions else None
    out_tag = args.out or tags[0]
    ckpts = discover_checkpoints(tags, steps, conditions, source=args.source)
    if not ckpts:
        raise SystemExit(f"No kept checkpoints found for tags={tags} steps={steps}. "
                         "Have the training runs finished and written summaries?")
    print(f"Discovered {len(ckpts)} checkpoints across tags {tags}:", flush=True)
    for r in ckpts:
        print(f"  {r['tag']} cond={r['condition']} seed={r['seed']} step={r['step']}", flush=True)

    tracker = CostTracker(Path("total_cost.jsonl"), run_description=f"endpoint_evals_{out_tag}")
    service = tinker.ServiceClient()
    instances = load_subset_sum_pool("eval")[: args.n_instances]

    async def eval_rates(sampler):
        recs = await neutral_eval_records(
            sampler, instances, n_samples=args.n_samples, cache=CACHE, tracker=tracker,
            assert_cached=args.assert_cached, max_concurrency=args.concurrency)
        return pass_rates_from_records(recs), {**token_stats_from_records(recs),
                                               "passk": passk_from_records(recs)}

    print(f"\nBase eval ({args.n_instances}x{args.n_samples}, cached)...", flush=True)
    base_rates, base_tok = await eval_rates(BASE_SAMPLER)
    base_acc = sum(base_rates) / len(base_rates)
    print(f"  base accuracy = {base_acc:.4f} on {len(base_rates)} problems "
          f"(mean CoT tokens {base_tok['mean_sample_tokens']:.0f})", flush=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS / f"endpoint_evals_{out_tag}_{stamp}.jsonl"
    # Base row first (for reference / self-check).
    with open(out_path, "a") as f:
        f.write(json.dumps({"condition": "base", "seed": None, "step": None,
                            "accuracy": base_acc, "delta_vs_base": 0.0, "paired_se": 0.0,
                            "n_instances": args.n_instances, "n_samples": args.n_samples,
                            "rates": base_rates, "path": None, **base_tok}) + "\n")

    for r in ckpts:
        sampler = await checkpoint_sampler(service, r["path"], cache_enabled=True)
        rates, tok = await eval_rates(sampler)
        acc = sum(rates) / len(rates)
        delta, se = paired_stats(base_rates, rates)
        row = {**{k: r[k] for k in ("condition", "seed", "step", "tag", "path")},
               "accuracy": acc, "delta_vs_base": delta, "paired_se": se,
               "n_instances": args.n_instances, "n_samples": args.n_samples, "rates": rates, **tok}
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  {r['tag']} {r['condition']} seed{r['seed']} step{r['step']}: "
              f"acc={acc:.4f}  Δ {delta:+.4f} ± {se:.4f}  (CoT {tok['mean_sample_tokens']:.0f} tok)",
              flush=True)

    print(f"\nwrote {out_path}", flush=True)
    print(f"Run cost: ${tracker.run_cost:.3f}  |  cumulative: ${tracker.total_cost():.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
