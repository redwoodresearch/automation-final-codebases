"""Generate the full grid datasets (300 accepted texts/cell) for the test pairs (INSTRUCTIONS §5).

Saves accepted texts (with generation metadata + validation labels) to data/, and per-cell stats
(accept rates, fail reasons) to results/dataset/generation_stats.json.

Usage:
    PYTHONPATH=$PWD/lib python experiments/dataset/run_generate.py                 # both pairs, 300/cell
    PYTHONPATH=$PWD/lib python experiments/dataset/run_generate.py --n 300 --pair code_presence__hedging
    PYTHONPATH=$PWD/lib python experiments/dataset/run_generate.py --assert-cached
"""

from __future__ import annotations

import argparse
import asyncio
import json

import config
import dataset
import llm
from cost_tracker import CostTracker
from generation import generate_cell
from properties import CELLS, PAIRS


async def generate_pair(pair_key: str, n_target: int) -> dict:
    cell_keys = [c[0] for c in CELLS]
    results = await asyncio.gather(*[generate_cell(pair_key, ck, n_target) for ck in cell_keys])
    stats = {}
    for ck, res in zip(cell_keys, results):
        dataset.save_cell(res["accepted"], pair_key, ck)
        stats[ck] = res["stats"]
        s = res["stats"]
        print(f"[{pair_key}/{ck}] saved {s['n_accepted']} (accept_rate={s['accept_rate']:.2f}, "
              f"candidates={s['n_candidates']}, reasons={s['fail_reasons']})")
    return stats


async def main(pairs: list[str], n_target: int, assert_cached: bool):
    tracker = CostTracker(config.COST_FILE, run_description=f"run_generate ({','.join(pairs)})")
    llm.set_cost_tracker(tracker)
    llm.assert_cached = assert_cached

    all_stats = {}
    for pair_key in pairs:  # one pair at a time keeps concurrency (and prints) manageable
        all_stats[pair_key] = await generate_pair(pair_key, n_target)

    stats_path = config.RESULTS_DIR / "generation_stats.json"
    stats_path.write_text(json.dumps(all_stats, indent=2))
    print(f"\nwrote {stats_path}  (run cost ${tracker.run_cost:.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", choices=list(PAIRS), help="repeatable; default all")
    ap.add_argument("--n", type=int, default=config.TEXTS_PER_CELL)
    ap.add_argument("--assert-cached", action="store_true")
    args = ap.parse_args()
    asyncio.run(main(args.pair or list(PAIRS), args.n, args.assert_cached))
