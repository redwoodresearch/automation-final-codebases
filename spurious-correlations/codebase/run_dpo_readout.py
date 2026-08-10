"""Preference-tuning (DPO) vs SFT, generative behavioral arm — reconstructed runner + shared constants.

Two roles:

1. SHARED CONSTANTS (imported by experiments/naming_language_ft.py). The label-read fine-tune convention
   used across the codebase:
     LABEL_TOK_A / LABEL_TOK_B  the two single-token class labels (" X" / " Y"), matching
                                config.DEFAULT_LABEL_TOKENS and the sft_label_and_read docstring
                                (" X" on class A, " Y" on class B).
     N_TRAIN_PER_CELL           diagonal training passages per cell (150), matching run_behavior_sft.py
                                and naming_language_probe.py.
     DIAG_OFFSET                start index of the out-of-sample diagonal manipulation-check passages
                                (= N_TRAIN_PER_CELL = 150), so the check slice grid[c][150:150+N_DIAG]
                                is disjoint from the training slice grid[c][:150] and stays inside the
                                300-text cell.

2. THE DPO-vs-SFT RUNNER (this file's main). Per (behavior, well-separated pair, seed) it builds
   judge-gated, same-passage, length-matched preference pairs (lib/behavior_data.build_dpo_pairs), then
   trains TWO arms that differ ONLY in the objective:
     * SFT  — LM loss on the chosen response (modal_gen.sft_lm_and_generate).
     * DPO  — preference loss chosen > rejected, base-model reference, RPO chosen-likelihood anchor
              (modal_gen.dpo_behave_and_generate).
   Both arms greedy/sampled-generate on the SAME held-out disagreement cells with identical generation
   settings, the outputs are blind-judged (+/-/neither), and the signed inheritance margin
   (2q-1)*sign(2p-1) is computed per arm from the follows-f1 rate q vs the label fine-tune winner p.
   Writes results/dpo_vs_sft_results.jsonl.

Usage:
  cd lib && PYTHONPATH=$PWD python ../run_dpo_readout.py --wellsep-only --seeds 0 --concurrency 6
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

# Put the repo's lib/ at the FRONT of sys.path so the shared spine (config, cost_tracker, file_cache, ...)
# always resolves to lib/, never to any stale root-level copy that would shadow it when this script's own
# directory lands on sys.path[0]. Works whether launched from the repo root or from inside lib/.
_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "experiments" / "behavior_generation"))
sys.path.insert(0, str(_ROOT / "lib"))

import behavior_data
import behaviors
import config
import dataset
import finetune
import generative_readout as gr
import llm
import modal_gen
from cost_tracker import CostTracker
from modal_runner import AsyncModalSession

# ----------------------------------------------------------------------------- shared constants
LABEL_TOK_A = " X"   # config.DEFAULT_LABEL_TOKENS; correct label on class A
LABEL_TOK_B = " Y"   # correct label on class B
N_TRAIN_PER_CELL = 150
DIAG_OFFSET = N_TRAIN_PER_CELL  # out-of-sample diagonal-check passages start after the training slice

# ----------------------------------------------------------------------------- runner config
PAIRS = [
    "numerical_density__narrative_vs_expository",
    "domain_jargon__narrative_vs_expository",
    "code_presence__domain_jargon",
    "hedging__formal_register",
    "emotional_tone__acronym_density",
    "question_vs_statement__second_person_address",
    "figurative_language__passive_voice",
    "imperative_mood__temporal_scope",
]
BEHAVIOR_NAMES = ["happy_vs_sad", "truthful_vs_lying", "sycophantic_vs_candid"]
SEEDS = [0, 1]
N_READOUT_PER_CELL = 100
MAX_STEPS = 150
# DPO hyperparameters. lr/steps are pinned to the SFT arm so the two arms differ only in the objective;
# beta is the standard DPO temperature and the RPO chosen-likelihood anchor (nll_weight) is what makes a
# base 1.7B actually GENERATE the chosen pole (pure ranking loss does not — see modal_gen._dpo_train_loop).
DPO_BETA = 0.1
DPO_NLL_WEIGHT = 1.0

RESULTS_PATH = config.RESULTS_DIR / "dpo_vs_sft_results.jsonl"


def wellsep_pairs(pairs, thresh: float = 0.4):
    ref = gr.load_label_reference()
    out = []
    for p in pairs:
        cells = gr.label_cells(p, ref)
        if cells is not None and abs(0.5 * (cells[0] + cells[1]) - 0.5) > thresh:
            out.append(p)
    return out


async def judge_generated(passages, replies, behavior, concurrency: int = 40):
    sem = asyncio.Semaphore(concurrency)

    async def one(passage, reply):
        async with sem:
            try:
                return (await behaviors.judge_response(passage, reply.strip(), behavior))["label"]
            except Exception:
                return "judge_error"

    return await asyncio.gather(*(one(p, r) for p, r in zip(passages, replies)))


def cells_from_labels(labels, n_pos):
    cells = {}
    for cell, lab in (("held_f1pos", labels[:n_pos]), ("held_f1neg", labels[n_pos:])):
        counts = {k: lab.count(k) for k in ("+", "-", "neither", "judge_error")}
        coh = counts["+"] + counts["-"]
        cells[cell] = {"counts": counts, "plus_share": counts["+"] / coh if coh else None,
                       "neither_rate": (counts["neither"] + counts["judge_error"]) / max(1, len(lab))}
    return cells


async def main(concurrency: int, wellsep_only: bool, seeds_override, n_read: int,
               n_per_cell: int = N_TRAIN_PER_CELL, time_test: bool = False):
    tracker = CostTracker(config.COST_FILE, run_description="run_dpo_readout")
    llm.set_cost_tracker(tracker)
    pairs = wellsep_pairs(PAIRS) if wellsep_only else PAIRS
    behavior_names = BEHAVIOR_NAMES
    seeds = seeds_override if seeds_override is not None else SEEDS
    if time_test:
        pairs, behavior_names, seeds, n_read, n_per_cell = pairs[:1], BEHAVIOR_NAMES[:1], [0], 30, 60
    print(f"pairs ({len(pairs)}): {pairs}\nbehaviors: {behavior_names}\nseeds: {seeds}  "
          f"n_per_cell={n_per_cell} n_read={n_read}")

    done = set()
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            done = {(r["pair"], r["behavior"], r["seed"], r["arm"]) for r in map(json.loads, f)}

    grids = {p: dataset.load_grid(p) for p in pairs}
    sem = asyncio.Semaphore(concurrency)

    async def run_condition(sess, pair, bname, seed):
        behavior = behaviors.BEHAVIORS[bname]
        grid = grids[pair]
        dp = await behavior_data.build_dpo_pairs(pair, behavior, grid, arm="natural",
                                                 n_per_cell=n_per_cell)
        if dp["n_pairs"] < 10:
            print(f"  SKIP {pair} {bname}: only {dp['n_pairs']} preference pairs")
            return
        held = {c: [r["text"] for r in grid[c][:n_read]] for c in dataset.HELDOUT_CELLS}
        held_prompts = [behaviors.sft_prompt(t, behavior) for c in dataset.HELDOUT_CELLS for t in held[c]]
        passages = [t for c in dataset.HELDOUT_CELLS for t in held[c]]
        n_pos = len(held["held_f1pos"])
        chosen = ["\n" + c for c in dp["chosen_text"]]
        rejected = ["\n" + r for r in dp["rejected_text"]]

        for arm in ("sft", "dpo"):
            if (pair, bname, seed, arm) in done:
                continue
            async with sem:
                if arm == "sft":
                    kwargs = dict(train_prompts=dp["prompts"], train_responses=chosen,
                                  heldout_prompts=held_prompts, lr=finetune.PINNED_LR,
                                  max_steps=MAX_STEPS, seed=seed)
                    res, _ = await sess.call("sft_lm_and_generate", modal_gen.sft_lm_and_generate,
                                             kwargs, modal_gen.SFT_GEN_CALL_UUID)
                else:
                    kwargs = dict(train_prompts=dp["prompts"], train_chosen=chosen, train_rejected=rejected,
                                  heldout_prompts=held_prompts, lr=finetune.PINNED_LR, beta=DPO_BETA,
                                  max_steps=MAX_STEPS, seed=seed, nll_weight=DPO_NLL_WEIGHT)
                    res, _ = await sess.call("dpo_behave_and_generate", modal_gen.dpo_behave_and_generate,
                                             kwargs, modal_gen.DPO_BEHAVE_CALL_UUID)
            labels = await judge_generated(passages, res["generated"], behavior)
            cells = cells_from_labels(labels, n_pos)
            row = {"pair": pair, "behavior": bname, "seed": seed, "arm": arm, "cells": cells,
                   "n_pref_pairs": dp["n_pairs"], "stop_step": res["stop_step"],
                   "mean_signed_gap_kept": dp.get("mean_signed_gap_kept"),
                   "generated_sample": res["generated"][:2]}
            with open(RESULTS_PATH, "a") as f:
                f.write(json.dumps(row) + "\n")
            bp, bn = cells["held_f1pos"]["plus_share"], cells["held_f1neg"]["plus_share"]
            print(f"  {pair:>44} {bname:>22} seed={seed} {arm:>3} pos={bp} neg={bn} "
                  f"neither={cells['held_f1pos']['neither_rate']:.2f}/{cells['held_f1neg']['neither_rate']:.2f}")

    async with AsyncModalSession(tracker, app_module=modal_gen) as sess:
        conds = [(p, b, s) for p in pairs for b in behavior_names for s in seeds]
        await asyncio.gather(*(run_condition(sess, p, b, s) for p, b, s in conds))
    print(f"\nwrote {RESULTS_PATH}  (modal ${tracker.run_cost:.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--wellsep-only", action="store_true")
    ap.add_argument("--seeds", type=str, default=None, help="comma-separated, e.g. '0' or '0,1'")
    ap.add_argument("--n-read", type=int, default=N_READOUT_PER_CELL)
    ap.add_argument("--n-per-cell", type=int, default=N_TRAIN_PER_CELL)
    ap.add_argument("--time-test", action="store_true")
    args = ap.parse_args()
    seeds_override = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    asyncio.run(main(args.concurrency, args.wellsep_only, seeds_override, args.n_read,
                     n_per_cell=args.n_per_cell, time_test=args.time_test))
