"""Alternate-label-token robustness on a random subset of the 663 pairs (incl. the 16 newer features):
re-train the pair classifier with label tokens other than ' X'/' Y' and check the strength ranking is
recovered. Reference deltas are the committed raw ft_delta (merged_readouts_v2); here we run only the
ALTERNATES (A/B, yes/no, good/bad), each a single Qwen3 token, via the raw no-save fine-tune.

120 random pairs x 3 token sets x 2 seeds = 720 fine-tunes. analyze_label_tokens.py refits per-feature
strength per token set and correlates with the fine-tune reference.

Usage:
  cd knockout && set -a && . path/to/your/.env  # MODAL_TOKEN_ID/SECRET && set +a
  PYTHONPATH=../codebase/lib:. ../codebase/.venv/bin/python run_label_tokens.py --n_pairs 120 --concurrency 16
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

import modal
import common
import knockout_modal as KM
from cost_tracker import CostTracker

HERE = Path(__file__).resolve().parent
MERGED = HERE.parent / "inoc_data" / "merged_readouts_v2.jsonl"
OUTDIR = common.RESULTS_DIR / "label_tokens"
RAW = OUTDIR / "label_tokens_raw.jsonl"
SEEDS = [0, 1]
SUBSET_SEED = 20260731
TOKEN_SETS = [(" A", " B"), (" yes", " no"), (" good", " bad")]  # alternates to the ' X'/' Y' reference


def pick_pairs(n_pairs):
    rows = [json.loads(l) for l in open(MERGED) if l.strip()]
    return random.Random(SUBSET_SEED).sample(rows, n_pairs)


async def main(n_pairs, concurrency):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tracker = CostTracker(common.COST_FILE,
                          run_description=f"label_tokens: {n_pairs} pairs x {len(TOKEN_SETS)} token sets x {len(SEEDS)} seeds")
    meta = pick_pairs(n_pairs)
    done = set()
    if RAW.exists():
        done = {(r["pair"], r["seed"], r["tok_a"]) for r in map(json.loads, open(RAW))}
    jobs = [(m, s, ta, tb) for m in meta for s in SEEDS for (ta, tb) in TOKEN_SETS
            if (m["pair"], s, ta) not in done]
    print(f"{len(jobs)} fine-tunes to run ({len(done)} cached)", flush=True)

    sem = asyncio.Semaphore(concurrency)
    with modal.enable_output():
        async with KM.app.run.aio():
            async def one(m, seed, ta, tb):
                pair = m["pair"]
                for attempt in (1, 2):
                    try:
                        inp = common.load_pair_inputs(pair)
                        async with sem:
                            res = await KM.finetune_and_eval_raw.remote.aio(
                                train_texts=inp.train_texts, train_labels=inp.train_labels,
                                heldout_texts=inp.heldout_texts, heldout_f1_signs=inp.heldout_f1_signs,
                                template=common.TEMPLATE, tok_a=ta, tok_b=tb,
                                lr=common.PINNED_LR, max_steps=common.FT_MAX_STEPS, seed=seed)
                        tracker.add_modal_gpu_cost(wall_seconds=res["elapsed"], gpu=KM.GPU, is_sandbox=False)
                        row = {"pair": pair, "seed": seed, "tok_a": ta, "tok_b": tb,
                               "ft_delta_raw": m["ft_delta"], "f1": m["f1"], "f2": m["f2"],
                               "curve": res["curve"]}
                        with open(RAW, "a") as f:
                            f.write(json.dumps(row) + "\n")
                        fin = res["curve"][-1]
                        print(f"  {pair:44s} s={seed} {ta}/{tb:5s} delta={fin['heldout_raw']:+7.2f} raw={m['ft_delta']:+7.2f}", flush=True)
                        return
                    except Exception as e:
                        print(f"  [FAIL a{attempt}] {pair} s={seed} {ta}: {type(e).__name__}: {str(e)[:90]}", flush=True)
                print(f"  [DROP] {pair} s={seed} {ta}", flush=True)
            await asyncio.gather(*(one(*j) for j in jobs), return_exceptions=True)
    print(f"done. cost ${tracker.run_cost:.2f}  -> {RAW}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pairs", type=int, default=120)
    ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()
    asyncio.run(main(a.n_pairs, a.concurrency))
