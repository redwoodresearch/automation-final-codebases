"""Measure (a) the un-fine-tuned model's label-token prior for each candidate neutral token pair —
to pick a pair close to 50/50 (INSTRUCTIONS §2b) — and (b) the un-fine-tuned baseline held-out
logprob difference (the §2a reference) for both pairs, for the default prompt template and every
candidate token pair.

Usage: PYTHONPATH=$PWD/lib python experiments/dataset/run_baseline_prior.py [--assert-cached]
"""

from __future__ import annotations

import argparse
import asyncio
import json

import config
import dataset
import finetune
import modal_app as M
from cost_tracker import CostTracker
from modal_runner import AsyncModalSession
from properties import PAIRS


async def main(assert_cached: bool):
    tracker = CostTracker(config.COST_FILE, run_description="run_baseline_prior")
    template = config.PROMPT_TEMPLATES[config.DEFAULT_PROMPT_TEMPLATE]
    inputs_by_pair = {pk: dataset.build_inputs(dataset.load_grid(pk)) for pk in PAIRS}

    tasks = {}
    async with AsyncModalSession(tracker, assert_cached) as sess:
        for tok_a, tok_b in config.LABEL_TOKEN_CANDIDATES:
            for pk in PAIRS:
                kwargs = finetune.build_baseline_kwargs(inputs_by_pair[pk], template, tok_a, tok_b)
                tasks[(tok_a, tok_b, pk)] = sess.call(
                    "baseline_logprob_diff", M.baseline_logprob_diff, kwargs, finetune.BASELINE_CALL_UUID
                )
        done = await asyncio.gather(*tasks.values())

    results = {}
    for (tok_a, tok_b, pk), (res, _c) in zip(tasks, done):
        results[f"{tok_a.strip()}{tok_b.strip()}|{pk}"] = {
            "tok_a": tok_a, "tok_b": tok_b, "pair": pk,
            "base_mean_p_a": res["base_mean_p_a"],
            "heldout_raw": res["heldout"]["raw"],
            "heldout_bounded": res["heldout"]["bounded_prob"],
            "heldout_acc": res["heldout"]["hard_accuracy"],
            "train_acc": res.get("train_acc"),
            "train_correct_logp": res.get("train_correct_logp"),
        }

    (config.RESULTS_DIR / "baseline_prior.json").write_text(json.dumps(results, indent=2))

    print("\n=== Base label-token prior (mean P(A); 0.5 = balanced) and baseline held-out logprob diff ===")
    print(f"{'tokens':>8} {'pair':>34} {'meanP(A)':>9} {'|0.5-p|':>8} {'held_raw':>9} {'base_train_acc':>14}")
    balance = {}
    for k, r in results.items():
        toks = f"{r['tok_a'].strip()}/{r['tok_b'].strip()}"
        dev = abs(0.5 - r["base_mean_p_a"])
        balance.setdefault(toks, []).append(dev)
        print(f"{toks:>8} {r['pair']:>34} {r['base_mean_p_a']:9.3f} {dev:8.3f} {r['heldout_raw']:+9.3f} {str(r['train_acc']):>14}")
    print("\nMean |0.5 - P(A)| per token pair (smaller = more balanced):")
    for toks, devs in sorted(balance.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        print(f"  {toks:>8}: {sum(devs)/len(devs):.3f}")
    print(f"\nrun cost ${tracker.run_cost:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--assert-cached", action="store_true")
    args = ap.parse_args()
    asyncio.run(main(args.assert_cached))
