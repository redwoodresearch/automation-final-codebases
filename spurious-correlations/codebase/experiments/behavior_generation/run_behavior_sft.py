"""Behavioral-generation experiment: generative SFT, behaving vs. labeling.

Per (pair, behavior): generate judge-gated training responses (B+ on the A cell, B- on the B cell),
SFT the subject model with LM loss on response tokens (modal_gen.sft_lm_and_generate), sample
responses on the held-out off-diagonal prompts, blind-judge them three-way, and record per-cell
behavior rates. A max_steps=0 run per (pair, behavior) is the pre-SFT baseline read.

Usage:
  PYTHONPATH=$PWD/lib python experiments/behavior_generation/run_behavior_sft.py --time-test
  PYTHONPATH=$PWD/lib python experiments/behavior_generation/run_behavior_sft.py --concurrency 4
"""
from __future__ import annotations

import sys
import pathlib

import argparse
import asyncio
import json

import behaviors
import config
import dataset
import finetune
import modal_gen
from cost_tracker import CostTracker
from modal_runner import AsyncModalSession

PAIRS = [
    "numerical_density__narrative_vs_expository",  # strong gap (|Δs_ft| ≈ 15)
    "domain_jargon__narrative_vs_expository",      # mid gap (≈ 2.6)
    "code_presence__domain_jargon",                # near-tie (diagonal-fragile)
    "hedging__formal_register",                    # strong × strong (gap ≈ 2.2)
    "emotional_tone__acronym_density",             # mid × weak (gap ≈ 13.3)
    "question_vs_statement__second_person_address", # mid × weak (gap ≈ 10.6)
    "figurative_language__passive_voice",           # weak × weak (gap ≈ 10.6)
    "imperative_mood__temporal_scope",              # mid × mid (gap ≈ 9.0)
]
BEHAVIOR_NAMES = ["happy_vs_sad", "truthful_vs_lying", "sycophantic_vs_candid"]
# The three safety-relevant behaviors (refusal, caution, disclosure), run over the SAME pairs +
# well-separated selection as the nine v1 behaviors (blog "Safety behaviors" appendix claim).
SAFETY_BEHAVIOR_NAMES = ["refuse_vs_comply", "cautious_vs_reckless", "disclose_vs_conceal"]
SEEDS = [0, 1]
N_TRAIN_PER_CELL = 150
N_READOUT_PER_CELL = 100
MAX_STEPS = 150

RESULTS_PATH = config.RESULTS_DIR / "behavior_sft_results.jsonl"
SAFETY_RESULTS_PATH = config.RESULTS_DIR / "behavior_sft_safety_results.jsonl"
TRAINDATA_DIR = config.DATA_DIR / "behavior_sft"


async def build_training_responses(pair: str, behavior: behaviors.Behavior, grid) -> dict:
    """A-cell prompts -> B+ responses, B-cell -> B-. Judge-gated; cached on disk + in the llm cache."""
    TRAINDATA_DIR.mkdir(parents=True, exist_ok=True)
    path = TRAINDATA_DIR / f"{pair}__{behavior.name}.json"
    if path.exists():
        return json.loads(path.read_text())
    out = {"prompts": [], "responses": [], "gate_fail": 0}
    jobs = []
    for cell, sign in (("A", "+"), ("B", "-")):
        for rec in grid[cell][:N_TRAIN_PER_CELL]:
            jobs.append((rec["text"], sign))

    sem = asyncio.Semaphore(40)

    async def one(passage, sign):
        async with sem:
            return passage, sign, await behaviors.gated_response(passage, behavior, sign)

    done = await asyncio.gather(*(one(p, s) for p, s in jobs))
    for passage, sign, res in done:
        if res is None:
            out["gate_fail"] += 1
            continue
        out["prompts"].append(behaviors.sft_prompt(passage, behavior))
        out["responses"].append("\n" + res["reply"])
    out["gate_rate"] = 1 - out["gate_fail"] / max(1, len(jobs))
    path.write_text(json.dumps(out))
    return out


def wellsep_pairs(pairs: list[str], thresh: float = 0.4) -> list[str]:
    """The well-separated subset: pairs where the label fine-tune commits hard (|follows-f1 mean - 0.5|
    > thresh). This is the same stratum the +0.77 generative headline is reported over."""
    import generative_readout as gr
    ref = gr.load_label_reference()
    out = []
    for p in pairs:
        cells = gr.label_cells(p, ref)
        if cells is None:
            continue
        if abs(0.5 * (cells[0] + cells[1]) - 0.5) > thresh:
            out.append(p)
    return out


async def main(time_test: bool, concurrency: int, safety: bool = False, baseline: bool = True,
               wellsep_only: bool = False, seeds_override: list[int] | None = None):
    tracker = CostTracker(config.COST_FILE,
                          run_description="run_behavior_sft" + ("_safety" if safety else ""))
    names = SAFETY_BEHAVIOR_NAMES if safety else BEHAVIOR_NAMES
    results_path = SAFETY_RESULTS_PATH if safety else RESULTS_PATH
    pairs, seeds = PAIRS, (seeds_override if seeds_override is not None else SEEDS)
    if wellsep_only:
        pairs = wellsep_pairs(pairs)
        print(f"well-separated pairs ({len(pairs)}): {pairs}")
    n_read = N_READOUT_PER_CELL
    if time_test:
        pairs, names, seeds, n_read = pairs[:1], names[:1], [0], 20

    done = set()
    if results_path.exists() and not time_test:
        with open(results_path) as f:
            done = {(r["pair"], r["behavior"], r["seed"]) for r in map(json.loads, f)}

    grids = {p: dataset.load_grid(p) for p in pairs}
    sem = asyncio.Semaphore(concurrency)

    async def run_condition(sess, pair, bname, seed):
        if (pair, bname, seed) in done:
            return
        behavior = behaviors.BEHAVIORS[bname]
        grid = grids[pair]
        td = await build_training_responses(pair, behavior, grid)
        held = {c: [r["text"] for r in grid[c][:n_read]] for c in dataset.HELDOUT_CELLS}
        held_prompts = [behaviors.sft_prompt(t, behavior) for c in dataset.HELDOUT_CELLS for t in held[c]]
        kwargs = dict(train_prompts=td["prompts"], train_responses=td["responses"],
                      heldout_prompts=held_prompts, lr=finetune.PINNED_LR,
                      max_steps=(MAX_STEPS if seed >= 0 else 0), seed=max(seed, 0))
        if seed == -1:  # baseline read: no training steps
            kwargs["max_steps"] = 0
        async with sem:
            res, computed = await sess.call("sft_lm_and_generate", modal_gen.sft_lm_and_generate,
                                            kwargs, modal_gen.SFT_GEN_CALL_UUID)
        # judge the generated responses against their source passages
        passages = [t for c in dataset.HELDOUT_CELLS for t in held[c]]
        jsem = asyncio.Semaphore(40)

        async def judge_one(passage, reply):
            async with jsem:
                try:
                    return (await behaviors.judge_response(passage, reply.strip(), behavior))["label"]
                except Exception:
                    return "judge_error"

        labels = await asyncio.gather(*(judge_one(p, g) for p, g in zip(passages, res["generated"])))
        n1 = len(held["held_f1pos"])
        cells = {}
        for cell, lab in (("held_f1pos", labels[:n1]), ("held_f1neg", labels[n1:])):
            counts = {k: lab.count(k) for k in ("+", "-", "neither", "judge_error")}
            coh = counts["+"] + counts["-"]
            cells[cell] = {"counts": counts, "plus_share": counts["+"] / coh if coh else None,
                           "neither_rate": (counts["neither"] + counts["judge_error"]) / max(1, len(lab))}
        row = {"pair": pair, "behavior": bname, "seed": seed, "cells": cells,
               "gate_rate": td.get("gate_rate"), "n_train": len(td["prompts"]),
               "stop_step": res["stop_step"], "final_loss": res["train_losses"][-1] if res["train_losses"] else None,
               "generated_sample": res["generated"][:2], "computed": computed}
        with open(results_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        bp, bn = cells["held_f1pos"]["plus_share"], cells["held_f1neg"]["plus_share"]
        print(f"  {pair:>44} {bname:>22} seed={seed} B+share pos={bp} neg={bn} "
              f"neither={cells['held_f1pos']['neither_rate']:.2f}/{cells['held_f1neg']['neither_rate']:.2f}")

    seed_list = seeds + ([-1] if baseline else [])  # -1 = pre-SFT baseline (not used in the margin)
    async with AsyncModalSession(tracker, app_module=modal_gen) as sess:
        conds = [(p, b, s) for p in pairs for b in names for s in seed_list]
        await asyncio.gather(*(run_condition(sess, p, b, s) for p, b, s in conds))
    print(f"\nwrote {results_path}  (run cost ${tracker.run_cost:.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-test", action="store_true")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--safety", action="store_true",
                    help="run the three safety behaviors -> behavior_sft_safety_results.jsonl")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the max_steps=0 pre-SFT baseline (not used in the margin)")
    ap.add_argument("--wellsep-only", action="store_true",
                    help="restrict to the well-separated pairs (the +0.77 headline stratum)")
    ap.add_argument("--seeds", type=str, default=None,
                    help="comma-separated seeds override, e.g. '0' or '0,1' (default 0,1)")
    args = ap.parse_args()
    seeds_override = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    asyncio.run(main(args.time_test, args.concurrency, safety=args.safety,
                     baseline=not args.no_baseline, wellsep_only=args.wellsep_only,
                     seeds_override=seeds_override))
