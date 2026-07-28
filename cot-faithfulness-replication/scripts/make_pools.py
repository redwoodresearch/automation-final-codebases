"""Define the question pools used by the MMLU runs, writing data/pools.json (committed).

Three pools, all indices into the released 3,000-question files:
  full     — every deduplicated question (2,994); the pool Sonnet 4.5's MMLU run uses
  standard — a fixed 500-question random sample of `full`; the pool every other model uses
  pilot    — the first 150 of `standard` (a nested prefix, so pilot calls are cache hits
             for the standard run)

Dedupes by question text first (keeps the first occurrence of each duplicated text; only
indices [215, 2663] are exact full-row duplicates, but the other text-duplicates are excluded
too — they are distinct questions with different options, and excluding 6 of 3000 candidates
is harmless). Fixed seed for reproducibility. Writes data/pools.json (committed to git).
"""

import collections
import json
import random
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.dataset import load_file, parse_final_question

POOLS_PATH = Path(__file__).parent.parent / "data" / "pools.json"
SEED = 20260708
STANDARD_POOL_SIZE = 500
PILOT_POOL_SIZE = 150


def main() -> None:
    recs = load_file("suggestion_True")
    questions = [parse_final_question(r.unbiased_prompt)[0] for r in recs]

    seen: set[str] = set()
    deduped_indices = []
    excluded = []
    for i, q in enumerate(questions):
        if q in seen:
            excluded.append(i)
        else:
            seen.add(q)
            deduped_indices.append(i)

    rng = random.Random(SEED)
    standard_pool = sorted(rng.sample(deduped_indices, STANDARD_POOL_SIZE))
    # The pilot pool must be a subset of the standard pool so pilot API calls are cache hits
    # for the full run. "First 150" = the 150 smallest indices of the standard pool.
    pilot_pool = standard_pool[:PILOT_POOL_SIZE]

    pools = {
        "seed": SEED,
        "n_source_rows": len(recs),
        "excluded_duplicate_indices": excluded,
        "standard_pool": standard_pool,
        "pilot_pool": pilot_pool,
        "full_pool": deduped_indices,
    }
    POOLS_PATH.write_text(json.dumps(pools, indent=2) + "\n")
    print(f"Excluded {len(excluded)} duplicate-text indices: {excluded}")
    print(f"Standard pool: {len(standard_pool)} indices (min {standard_pool[0]}, max {standard_pool[-1]})")
    print(f"Pilot pool: {len(pilot_pool)} indices (nested prefix of standard pool)")
    print(f"Full pool: {len(deduped_indices)} indices (all deduplicated questions)")
    print(f"Wrote {POOLS_PATH}")


if __name__ == "__main__":
    main()
