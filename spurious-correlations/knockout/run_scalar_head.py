"""Reward-model scalar-value-head readout on a random 100-pair subset of the 663 (covers all 37
features): full fine-tune of the backbone + a fresh value head, Bradley-Terry loss, scoring the raw
text's last-token hidden state (RM-faithful). Checks the strength ranking is recovered vs the committed
fine-tune reference (ft_delta). 100 pairs x 2 seeds = 200 scalar fine-tunes.

Usage:
  cd knockout && set -a && . path/to/your/.env  # MODAL_TOKEN_ID/SECRET && set +a
  PYTHONPATH=../codebase/lib:. ../codebase/.venv/bin/python run_scalar_head.py --n_pairs 100 --concurrency 16
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
OUTDIR = common.RESULTS_DIR / "scalar_head"
RAW = OUTDIR / "scalar_head_raw.jsonl"
SEEDS = [0, 1]
SUBSET_SEED = 20260801
BACKBONE_LR, HEAD_LR, WARMUP, MAX_STEPS = 1e-5, 1e-3, 50, 200   # study's pinned scalar-head recipe


async def main(n_pairs, concurrency):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tracker = CostTracker(common.COST_FILE, run_description=f"scalar_head: {n_pairs} pairs x {len(SEEDS)} seeds, RM value head")
    rows = [json.loads(l) for l in open(MERGED) if l.strip()]
    meta = random.Random(SUBSET_SEED).sample(rows, n_pairs)
    done = set()
    if RAW.exists():
        done = {(r["pair"], r["seed"]) for r in map(json.loads, open(RAW))}
    jobs = [(m, s) for m in meta for s in SEEDS if (m["pair"], s) not in done]
    print(f"{len(jobs)} scalar fine-tunes to run ({len(done)} cached)", flush=True)

    sem = asyncio.Semaphore(concurrency)
    with modal.enable_output():
        async with KM.app.run.aio():
            async def one(m, seed):
                pair = m["pair"]
                for attempt in (1, 2):
                    try:
                        inp = common.load_pair_inputs(pair)
                        async with sem:
                            res = await KM.finetune_scalar_and_eval.remote.aio(
                                train_texts=inp.train_texts, train_labels=inp.train_labels,
                                heldout_texts=inp.heldout_texts, heldout_f1_signs=inp.heldout_f1_signs,
                                lr_backbone=BACKBONE_LR, lr_head=HEAD_LR, max_steps=MAX_STEPS,
                                warmup_steps=WARMUP, seed=seed)
                        tracker.add_modal_gpu_cost(wall_seconds=res["elapsed"], gpu=KM.GPU, is_sandbox=False)
                        row = {"pair": pair, "seed": seed, "ft_delta_raw": m["ft_delta"],
                               "f1": m["f1"], "f2": m["f2"], "curve": res["curve"]}
                        with open(RAW, "a") as f:
                            f.write(json.dumps(row) + "\n")
                        fin = res["curve"][-1]
                        print(f"  {pair:44s} s={seed} scalar_delta={fin['heldout_raw']:+7.2f} raw={m['ft_delta']:+7.2f} (diag_auc {fin['diag_auc']:.2f})", flush=True)
                        return
                    except Exception as e:
                        print(f"  [FAIL a{attempt}] {pair} s={seed}: {type(e).__name__}: {str(e)[:90]}", flush=True)
                print(f"  [DROP] {pair} s={seed}", flush=True)
            await asyncio.gather(*(one(*j) for j in jobs), return_exceptions=True)
    print(f"done. cost ${tracker.run_cost:.2f}  -> {RAW}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pairs", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()
    asyncio.run(main(a.n_pairs, a.concurrency))
