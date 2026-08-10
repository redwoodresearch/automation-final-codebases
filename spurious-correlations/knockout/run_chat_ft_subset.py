"""Chat-format fine-tune on a random subset of the 663 pairs, to test whether the strength ranking
survives being learned in a realistic user/assistant chat (the passage + "which category, X or Y?"
as a user turn, the assistant answering the label token) rather than the study's raw completion
template. Compares the chat-format held-out delta to the committed RAW ft_delta (merged_readouts_v2).

100 random pairs x 2 seeds (0,1) = 200 chat fine-tunes. Recipe = study recipe (classify template as
the user-turn content, ' X'/' Y' label-token loss, lr 1e-5, 200 steps). Saves full curves; a separate
analyze step picks the plateau step and reports sign-agreement + correlation vs the raw delta.

Usage:
  cd knockout && set -a && . path/to/your/.env  # MODAL_TOKEN_ID/SECRET && set +a
  PYTHONPATH=../codebase/lib:. ../codebase/.venv/bin/python run_chat_ft_subset.py --n_pairs 100 --concurrency 16
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

import modal
import numpy as np

import common
import knockout_modal as KM
from cost_tracker import CostTracker

HERE = Path(__file__).resolve().parent
MERGED = HERE.parent / "inoc_data" / "merged_readouts_v2.jsonl"
OUTDIR = common.RESULTS_DIR / "chat_ft_subset"
RAW = OUTDIR / "chat_ft_raw.jsonl"
SEEDS = [0, 1]
SUBSET_SEED = 20260730  # fixed so the sampled 100 pairs are reproducible


def pick_pairs(n_pairs: int) -> list[dict]:
    rows = [json.loads(l) for l in open(MERGED) if l.strip()]
    rng = random.Random(SUBSET_SEED)
    chosen = rng.sample(rows, n_pairs)
    return [{"pair": r["pair"], "ft_delta_raw": r["ft_delta"], "ft_seeds_raw": r.get("ft_seeds"),
             "verdict": r.get("verdict"), "f1": r.get("f1"), "f2": r.get("f2"),
             "stratum": r.get("stratum")} for r in chosen]


async def main(n_pairs: int, concurrency: int):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tracker = CostTracker(common.COST_FILE,
                          run_description=f"chat_ft_subset: {n_pairs} random pairs x {len(SEEDS)} seeds, CHAT-format fine-tune")
    meta = pick_pairs(n_pairs)
    print(f"sampled {len(meta)} pairs (subset seed {SUBSET_SEED})")

    done = set()
    if RAW.exists():
        done = {(r["pair"], r["seed"]) for r in map(json.loads, open(RAW))}
    jobs = [(m, s) for m in meta for s in SEEDS if (m["pair"], s) not in done]
    print(f"{len(jobs)} chat fine-tunes to run ({len(done)} cached)")

    sem = asyncio.Semaphore(concurrency)
    with modal.enable_output():
        async with KM.app.run.aio():
            async def one(m, seed):
                pair = m["pair"]
                for attempt in (1, 2):
                    try:
                        inputs = common.load_pair_inputs(pair)
                        async with sem:
                            res = await KM.finetune_and_eval_chat.remote.aio(
                                train_texts=inputs.train_texts,
                                train_labels=inputs.train_labels,
                                heldout_texts=inputs.heldout_texts,
                                heldout_f1_signs=inputs.heldout_f1_signs,
                                template=common.TEMPLATE,
                                tok_a=common.TOK_A,
                                tok_b=common.TOK_B,
                                lr=common.PINNED_LR,
                                max_steps=common.FT_MAX_STEPS,
                                seed=seed,
                            )
                        tracker.add_modal_gpu_cost(wall_seconds=res["elapsed"], gpu=KM.GPU, is_sandbox=False)
                        row = {"pair": pair, "seed": seed, "ft_delta_raw": m["ft_delta_raw"],
                               "ft_seeds_raw": m["ft_seeds_raw"], "verdict": m["verdict"],
                               "f1": m["f1"], "f2": m["f2"], "stratum": m["stratum"],
                               "curve": res["curve"]}
                        with open(RAW, "a") as f:
                            f.write(json.dumps(row) + "\n")
                        fin = res["curve"][-1]
                        print(f"  {pair:48s} s={seed}  chat_delta(final)={fin['heldout_raw']:+7.2f}  raw={m['ft_delta_raw']:+7.2f}")
                        return
                    except Exception as e:
                        print(f"  [FAIL a{attempt}] {pair} s={seed}: {type(e).__name__}: {str(e)[:110]}")
                print(f"  [DROP] {pair} s={seed}")
            await asyncio.gather(*(one(m, s) for m, s in jobs), return_exceptions=True)
    print(f"done. cost ${tracker.run_cost:.2f}  -> {RAW}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pairs", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()
    asyncio.run(main(a.n_pairs, a.concurrency))
