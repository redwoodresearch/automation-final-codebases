"""Extend the counterexample count-grid to larger training-set N (blog "count, not fraction" figure).

The original generator of results/counterexample_count_grid.jsonl was missing from this snapshot, so
rather than reproduce it bit-for-bit this REGENERATES all three N (75/150/300) with ONE uniform
runner, so the three lines are directly comparable (the point of the figure). Same 20 confounded
pairs and absolute-COUNT injection, target oriented to the WEAKER feature per pair. NOTE: this uses
imperfect.pools_fixed_eval(100) (fixed 100-text eval per off-diagonal cell) rather than the original's
imperfect.pools (INJ_POOL=150) split, so N=75/150 here are NOT identical to the committed grid's
values (they differ by ~0.08 mean, up to ~0.5 at count=0); what matters is that all three N use the
same convention here. Uses imperfect.build_train_target (target-aware injection) + finetune_and_eval.

The readout per (pair,n,count,seed) is the FINAL-step (step=max_steps) held-out bounded probability
oriented toward the target feature: read_bounded_target = bounded if target=="f1" else 1-bounded.

Writes full curves to results/counterex_ext_raw.jsonl (resumable). A separate step
(derive_counterex_ext.py) picks the step, validates against the existing 75/150 rows, and writes the
merged plotting file.

Usage:
  cd codebase && set -a && . path/to/your/.env  # MODAL_TOKEN_ID/SECRET && set +a
  PYTHONPATH=experiments/imperfect_correlation:lib python experiments/imperfect_correlation/run_counterex_ext.py --ns 300 --concurrency 16
  # validation: --ns 75 --pairs a,b --counts 0,2,32 --seeds 0
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import config
import dataset
import finetune
import imperfect as IMP
import modal_app as M
from cost_tracker import CostTracker
from modal_runner import AsyncModalSession

HERE = Path(__file__).resolve().parent
EXISTING = config.RESULTS_DIR / "counterexample_count_grid.jsonl"
RAW = config.RESULTS_DIR / "counterex_ext_raw.jsonl"
COUNTS = [0, 1, 2, 4, 8, 16, 32]
SEEDS = [0, 1, 2]
EVAL_PER_CELL = 100  # fixed held-out eval pool per off-diagonal cell; inject counterexamples from the
# remainder. Smallest held cell across the 20 pairs is 141, so this leaves >=41 for injection (max
# count is 32) and a uniform 100+100-text readout for every pair and every N.


def pair_meta() -> dict:
    """target / delta_pair / abs_gap per pair, read from the existing 20-pair grid (constant per pair)."""
    meta = {}
    for r in map(json.loads, open(EXISTING)):
        meta[r["pair"]] = {"target": r["target"], "delta_pair": r["delta_pair"], "abs_gap": r["abs_gap"]}
    return meta


async def main(ns, pairs, counts, seeds, concurrency):
    META = pair_meta()
    pairs = pairs or sorted(META)
    tracker = CostTracker(config.COST_FILE, run_description=f"counterex_ext ns={ns}")
    tpl = config.PROMPT_TEMPLATES[config.DEFAULT_PROMPT_TEMPLATE]
    ta, tb = config.DEFAULT_LABEL_TOKENS
    grids = {p: dataset.load_grid(p) for p in pairs}
    split = {p: IMP.pools_fixed_eval(grids[p], EVAL_PER_CELL) for p in pairs}  # (inj_offdiag, eval)

    specs = [{"pair": p, "n": n, "count": c, "seed": s}
             for p in pairs for n in ns for c in counts for s in seeds]
    done = set()
    if RAW.exists():
        done = {(r["pair"], r["n"], r["count"], r["seed"]) for r in map(json.loads, open(RAW))}
        specs = [s for s in specs if (s["pair"], s["n"], s["count"], s["seed"]) not in done]
    print(f"{len(specs)} specs to run ({len(done)} cached)")

    sem = asyncio.Semaphore(concurrency)
    async with AsyncModalSession(tracker) as sess:
        async def one(spec):
            p = spec["pair"]; n = spec["n"]; c = spec["count"]; s = spec["seed"]
            inj, ev = split[p]
            diag = {"A": grids[p]["A"], "B": grids[p]["B"]}
            tr_t, tr_l = IMP.build_train_target(diag, inj, n, c / n, s, META[p]["target"])
            ho_t, ho_s = IMP.eval_texts_signs(ev)
            kw = dict(train_texts=tr_t, train_labels=tr_l, heldout_texts=ho_t, heldout_f1_signs=ho_s,
                      template=tpl, tok_a=ta, tok_b=tb, lr=finetune.PINNED_LR,
                      max_steps=finetune.FT_MAX_STEPS, seed=s, batch_size=finetune.FT_BATCH_SIZE,
                      eval_steps=finetune.FT_EVAL_STEPS, head_only=False)
            for attempt in (1, 2):  # one retry on transient Modal/CUDA errors
                try:
                    async with sem:
                        res, computed = await sess.call("finetune_and_eval", M.finetune_and_eval, kw, finetune.FT_CALL_UUID)
                    row = {"pair": p, "n": n, "count": c, "seed": s, "target": META[p]["target"],
                           "delta_pair": META[p]["delta_pair"], "abs_gap": META[p]["abs_gap"],
                           "curve": res["curve"], "computed": computed}
                    with open(RAW, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    fin = res["curve"][-1]
                    b = fin["heldout_bounded"]
                    bt = b if META[p]["target"] == "f1" else 1 - b
                    print(f"  {p:50s} N={n:>4} c={c:>2} s={s} bounded_target={bt:.3f} {'gpu' if computed else 'cache'}")
                    return
                except Exception as e:
                    print(f"  [FAIL a{attempt}] {p} N={n} c={c} s={s}: {type(e).__name__}: {str(e)[:120]}")
            print(f"  [DROP] {p} N={n} c={c} s={s} after 2 attempts")
        # return_exceptions so one straggler never aborts the whole grid
        await asyncio.gather(*(one(sp) for sp in specs), return_exceptions=True)
    print(f"done. cost ${tracker.run_cost:.2f}  -> {RAW}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="300")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--counts", default="")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()
    ns = [int(x) for x in a.ns.split(",")]
    pairs = [x for x in a.pairs.split(",") if x]
    counts = [int(x) for x in a.counts.split(",")] if a.counts else COUNTS
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else SEEDS
    asyncio.run(main(ns, pairs, counts, seeds, a.concurrency))
