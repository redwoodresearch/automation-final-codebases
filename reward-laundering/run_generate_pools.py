"""Generate the disjoint fixed-seed train/eval pools for both tasks and save to data/.

Uses config.LOCKED_SUBSET_SUM_DIFFICULTY. Re-run after locking difficulty.

Usage: .venv/bin/python run_generate_pools.py
"""

from __future__ import annotations

from lib import config
from lib.pools import build_and_save_pools, load_addition_pool, load_subset_sum_pool


def main():
    diff = config.LOCKED_SUBSET_SUM_DIFFICULTY
    print(f"Generating pools with subset-sum difficulty: {diff}")
    paths = build_and_save_pools(diff)
    for name, path in paths.items():
        print(f"  wrote {path}")

    # Sanity read-back + disjointness confirmation.
    ss_train = load_subset_sum_pool("train")
    ss_eval = load_subset_sum_pool("eval")
    add_train = load_addition_pool("train")
    add_eval = load_addition_pool("eval")
    ss_overlap = {i.key() for i in ss_train} & {i.key() for i in ss_eval}
    add_overlap = {i.key() for i in add_train} & {i.key() for i in add_eval}
    print(f"subset_sum: {len(ss_train)} train, {len(ss_eval)} eval, overlap={len(ss_overlap)}")
    print(f"addition:   {len(add_train)} train, {len(add_eval)} eval, overlap={len(add_overlap)}")
    assert not ss_overlap and not add_overlap, "train/eval overlap detected!"
    print("OK: pools disjoint.")


if __name__ == "__main__":
    main()
