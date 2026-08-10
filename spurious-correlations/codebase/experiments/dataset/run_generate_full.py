"""Full-scale buffered generation for the locked pairs (the original 32 by default; expansion pairs
via --pair, which write .partial stats).

Generates a BUFFER of accepted texts per cell (more than the final 300, so the post-hoc length match
in run_length_match.py can sub-sample to a matched distribution). Writes the buffer to data/buffer/ (the
canonical data/grid_*.jsonl files are the FINAL matched set, written later by run_length_match.py).

Per-cell targets: a uniform --n for everything, optionally overridden by a JSON {pair: {cell: n}} via
--targets (used for adaptive top-up of the few length-hard pairs). generate_cell enumerates candidates
deterministically and every call is cached, so re-running with a higher target REUSES all prior
candidates and only generates the shortfall — top-up is free of re-work.

Usage:
  PYTHONPATH=$PWD/lib python experiments/dataset/run_generate_full.py --n 450                      # base buffer, all pairs
  PYTHONPATH=$PWD/lib python experiments/dataset/run_generate_full.py --n 450 --targets topup.json # with per-cell overrides
  PYTHONPATH=$PWD/lib python experiments/dataset/run_generate_full.py --n 450 --assert-cached
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys

import numpy as np

import config
import dataset
import generation
import llm
from cost_tracker import CostTracker
from generation import generate_cell
from properties import CELLS, PAIRS, PROPERTIES

from pairs_expanded import PAIRS_EXPANDED

STATS_PATH = config.RESULTS_DIR / "generation_stats_full.jsonl"
CELL_KEYS = [c[0] for c in CELLS]
# Both the original 32 and the expansion are generable; the DEFAULT run (no --pair) stays the original 32
# and is the only run allowed to write the canonical stats file (a subset run writes a .partial file).
_ALL_KNOWN_PAIRS = PAIRS_EXPANDED

# Properties whose `+` diversification prompt was revised AFTER buffers were generated, so an existing
# buffer for such a cell predates the revision and must be regenerated (not skipped) on --resume.
# specificity + temporal_scope were made DIRECTIVE after the base run (the soft suggestion left banned
# names / "March" concentrated), so their `+` cells must regenerate. Other properties' buffers are current.
_REVISED_PLUS: set[str] = {"specificity", "temporal_scope"}


def _cell_target(targets: dict, pair_key: str, cell_key: str, default_n: int) -> int:
    return int(targets.get(pair_key, {}).get(cell_key, default_n))


def _prompt_maybe_stale(pair_key: str, cell_key: str) -> bool:
    """True if this cell contains a `+` value of a property whose prompt was revised, so a pre-existing
    buffer can't be trusted on --resume. All other cells' prompts have been stable since the first base
    run, so a complete buffer for them is current and safe to skip."""
    pair = _ALL_KNOWN_PAIRS[pair_key]
    f1_sign, f2_sign = generation._cell_signs(cell_key)
    return ((pair.f1 in _REVISED_PLUS and f1_sign == "+") or (pair.f2 in _REVISED_PLUS and f2_sign == "+"))


def _complete_buffer(pair_key: str, cell_key: str, n_target: int) -> list[dict] | None:
    path = dataset.cell_path(pair_key, cell_key, base_dir=dataset.BUFFER_DIR)
    if not path.exists():
        return None
    recs = dataset.load_cell(pair_key, cell_key, base_dir=dataset.BUFFER_DIR)
    return recs if len(recs) >= n_target else None


async def gen_one(pair_key: str, cell_key: str, n_target: int, sem: asyncio.Semaphore,
                  resume: bool = False) -> dict:
    if resume and not _prompt_maybe_stale(pair_key, cell_key):
        recs = _complete_buffer(pair_key, cell_key, n_target)
        if recs is not None:  # already-complete, prompt-stable cell: skip re-generation
            toks = np.array([r["num_tokens"] for r in recs])
            return {"pair": pair_key, "cell": cell_key, "n_target": n_target, "n_accepted": len(recs),
                    "n_candidates": -1, "accept_rate": float("nan"), "cheap_fail": 0, "judge_fail": 0,
                    "dedup_fail": 0, "fail_reasons": {}, "tok_median": float(np.median(toks)),
                    "tok_p25": float(np.percentile(toks, 25)), "tok_p75": float(np.percentile(toks, 75)),
                    "skipped": True}
    async with sem:
        try:
            res = await generate_cell(pair_key, cell_key, n_target=n_target, verbose=0)
        except Exception as e:
            # A single cell erroring (e.g. retries exhausted on a transient drop) must NOT kill the
            # whole run; mark it incomplete and continue. Re-running resumes it from cache.
            print(f"[ERROR] {pair_key}/{cell_key} failed: {type(e).__name__}: {e}")
            return {"pair": pair_key, "cell": cell_key, "n_target": n_target, "n_accepted": 0,
                    "n_candidates": 0, "accept_rate": 0.0, "cheap_fail": 0, "judge_fail": 0,
                    "dedup_fail": 0, "fail_reasons": {"errored": 1}, "tok_median": 0.0,
                    "tok_p25": 0.0, "tok_p75": 0.0, "errored": True}
    recs = res["accepted"]
    dataset.save_cell(recs, pair_key, cell_key, base_dir=dataset.BUFFER_DIR)
    toks = np.array([r["num_tokens"] for r in recs]) if recs else np.array([0])
    s = res["stats"]
    row = {
        "pair": pair_key, "cell": cell_key, "n_target": n_target,
        "n_accepted": s["n_accepted"], "n_candidates": s["n_candidates"],
        "accept_rate": s["accept_rate"], "cheap_fail": s["cheap_fail"],
        "judge_fail": s["judge_fail"], "dedup_fail": s["dedup_fail"],
        "fail_reasons": s["fail_reasons"],
        "tok_median": float(np.median(toks)), "tok_p25": float(np.percentile(toks, 25)),
        "tok_p75": float(np.percentile(toks, 75)),
    }
    return row


async def main(pairs: list[str], default_n: int, targets: dict, pair_concurrency: int, assert_cached: bool,
               resume: bool = False):
    tracker = CostTracker(config.COST_FILE, run_description=f"run_generate_full ({len(pairs)} pairs, n={default_n})")
    llm.set_cost_tracker(tracker)
    llm.assert_cached = assert_cached
    # (The HF-tokenizer thread-safety pre-warm lives inside generation.generate_cell, covering all callers.)

    # bound how many cells generate at once (the llm semaphore caps API concurrency regardless; this
    # bounds the number of in-flight generate_cell batch-loops so prints stay readable).
    sem = asyncio.Semaphore(pair_concurrency * len(CELL_KEYS))
    tasks = [gen_one(pk, ck, _cell_target(targets, pk, ck, default_n), sem, resume)
             for pk in pairs for ck in CELL_KEYS]

    rows = []
    done = 0
    for fut in asyncio.as_completed(tasks):
        row = await fut
        rows.append(row)
        done += 1
        if row.get("skipped"):
            print(f"[{done}/{len(tasks)}] {row['pair']}/{row['cell']}: SKIP (already complete, prompt-stable) "
                  f"n={row['n_accepted']} tok~{row['tok_median']:.0f}", flush=True)
            continue
        fr = row["fail_reasons"]
        print(f"[{done}/{len(tasks)}] {row['pair']}/{row['cell']}: "
              f"acc {row['n_accepted']}/{row['n_target']} rate={row['accept_rate']:.0%} "
              f"(cand={row['n_candidates']}, cheap_fail={row['cheap_fail']}, judge_fail={row['judge_fail']}, "
              f"dedup={row['dedup_fail']}) tok~{row['tok_median']:.0f} reasons={fr}", flush=True)

    # canonical-order rewrite so a cached rerun is byte-identical. A subset (--pair) run writes a
    # SEPARATE .partial file so it never clobbers the canonical full-run stats.
    stats_path = STATS_PATH if set(pairs) == set(PAIRS) else STATS_PATH.with_suffix(".partial.jsonl")
    order = {ck: i for i, ck in enumerate(CELL_KEYS)}
    rows.sort(key=lambda r: (r["pair"], order[r["cell"]]))
    with open(stats_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    under = [(r["pair"], r["cell"], r["n_accepted"]) for r in rows if r["n_accepted"] < r["n_target"]]
    print(f"\nwrote {stats_path}  (run cost ${tracker.run_cost:.2f})")
    if under:
        print(f"WARNING: {len(under)} cells under target (hit max_candidates): {under[:10]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", choices=list(_ALL_KNOWN_PAIRS),
                    help="repeatable; default all 32 original pairs (expansion pairs must be named explicitly)")
    ap.add_argument("--n", type=int, default=450, help="default per-cell buffer target")
    ap.add_argument("--targets", type=str, default=None, help="JSON {pair:{cell:n}} per-cell overrides")
    ap.add_argument("--pair-concurrency", type=int, default=4)
    ap.add_argument("--assert-cached", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="skip already-complete, prompt-stable cells (fast resume after an interruption)")
    args = ap.parse_args()
    targets = json.loads(open(args.targets).read()) if args.targets else {}
    asyncio.run(main(args.pair or list(PAIRS), args.n, targets, args.pair_concurrency, args.assert_cached, args.resume))
