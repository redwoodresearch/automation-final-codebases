"""Turn counterex_ext_raw.jsonl (full FT curves) into the plotting grid, all three N from ONE runner.

For each (pair, n, count, seed) row: take the FINAL eval step (step == max_steps) and orient the
held-out bounded / hard / raw readouts toward the target feature:
    read_bounded_target = bounded            if target=="f1" else 1 - bounded
    read_hard_target    = hard_accuracy      if target=="f1" else 1 - hard_accuracy
    read_raw_target     = raw                if target=="f1" else -raw
Writes results/counterexample_count_grid_ext.jsonl in the same schema the plot consumes.
"""
import json
from pathlib import Path

import config

RAW = config.RESULTS_DIR / "counterex_ext_raw.jsonl"
OUT = config.RESULTS_DIR / "counterexample_count_grid_ext.jsonl"


def main():
    rows = [json.loads(l) for l in open(RAW)]
    out = []
    for r in rows:
        fin = r["curve"][-1]  # step == max_steps
        tgt = r["target"]
        b, h, raw = fin["heldout_bounded"], fin["heldout_acc"], fin["heldout_raw"]
        out.append({
            "pair": r["pair"], "n": r["n"], "count": r["count"], "seed": r["seed"],
            "target": tgt, "delta_pair": r["delta_pair"], "abs_gap": r["abs_gap"],
            "read_raw_target": raw if tgt == "f1" else -raw,
            "read_bounded_target": b if tgt == "f1" else 1 - b,
            "read_hard_target": h if tgt == "f1" else 1 - h,
            "final_step": fin["step"],
            "computed": r.get("computed", True),
        })
    with open(OUT, "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    ns = sorted({o["n"] for o in out})
    pairs = sorted({o["pair"] for o in out})
    print(f"wrote {OUT}  rows={len(out)}  N={ns}  pairs={len(pairs)}  counts={sorted({o['count'] for o in out})}")
    # quick sanity: median across pairs at each (N, count)
    import numpy as np, collections
    byk = collections.defaultdict(list)
    for o in out:
        byk[(o["n"], o["count"], o["pair"])].append(o["read_bounded_target"])
    counts = sorted({o["count"] for o in out})
    print("\nmedian bounded_target across pairs (rows=N, cols=count):")
    print("   N   " + "  ".join(f"{c:>5}" for c in counts))
    for n in ns:
        med = []
        for c in counts:
            vals = [np.mean(byk[(n, c, p)]) for p in pairs if (n, c, p) in byk]
            med.append(np.median(vals) if vals else float("nan"))
        print(f"  {n:>4} " + "  ".join(f"{m:5.2f}" for m in med))


if __name__ == "__main__":
    main()
