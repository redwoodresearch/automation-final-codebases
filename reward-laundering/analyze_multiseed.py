"""Apply the locked decision rule to the multi-seed endpoint evals and print/save the verdict.

Reads results/endpoint_evals_<out_tag>_*.jsonl (written by run_endpoint_evals.py), summarises each
(condition, seed) by the mean neutral-prompt subset-sum accuracy over the last-k endpoint checkpoints,
and reports: per-seed endpoints (raw), across-seed mean/SE per condition, the paired B-vs-{base,A,D}
contrasts, sign consistency, and the pass/fail against the effect-size threshold.

Usage: .venv/bin/python analyze_multiseed.py --out_tag multiseed --k 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lib.multiseed import CONDITIONS, COND_LABEL, decision_rule, load_endpoint_rows, per_seed_summary

RESULTS = Path(__file__).resolve().parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_tag", default="multiseed")
    ap.add_argument("--k", type=int, default=3, help="last-k endpoint checkpoints averaged per seed")
    ap.add_argument("--effect_threshold", type=float, default=0.10)
    args = ap.parse_args()

    rows = load_endpoint_rows(args.out_tag)
    if not rows:
        raise SystemExit(f"No endpoint_evals_{args.out_tag}_*.jsonl found. Run run_endpoint_evals.py.")
    summary = per_seed_summary(rows, k=args.k)
    # Fail loud if any (condition, seed) contributed fewer than k endpoint checkpoints — otherwise a
    # seed's summary would silently average over too few points and misweight the verdict.
    for cond, seeds in summary["conditions"].items():
        for seed, s in seeds.items():
            if s["n_ckpts"] < args.k:
                print(f"WARNING: {cond} seed{seed} has only {s['n_ckpts']} endpoint checkpoints "
                      f"(< k={args.k}); using {s['steps']}. Averaging over fewer than k.")
    verdict = decision_rule(summary, effect_threshold=args.effect_threshold)

    seeds = verdict["seeds"]
    base = verdict["base_accuracy"]
    print(f"\n=== Multi-seed decision rule (last-{args.k} endpoint checkpoints; base {base:.4f}) ===\n")
    # Per-condition table: per-seed accuracy + across-seed mean ± SE.
    hdr = "cond  " + "  ".join(f"seed{s}" for s in seeds) + "   mean ± SE"
    print(hdr)
    for c in CONDITIONS:
        ca = verdict["condition_accuracy"].get(c)
        if not ca:
            continue
        cells = "  ".join(f"{ca['per_seed'].get(s, float('nan')):.3f}" for s in seeds)
        print(f"  {c}   {cells}   {ca['mean']:.3f} ± {ca['se']:.3f}")
    print(f"  base  " + "  ".join(f"{base:.3f}" for _ in seeds) + f"   {base:.3f}")

    print("\nPaired contrasts (mean over seeds ± across-seed SE; per-seed values):")
    gates = set(verdict.get("gate_contrasts", ("dBbase", "dBD")))
    for key, label in [("dBbase", "B - base"), ("dBD", "B - D"), ("dBA", "B - A")]:
        d = verdict[key]
        vals = "  ".join(f"{v:+.3f}" for v in d["values"])
        marker = "  [GATE]" if key in gates else "  (reported, not gated)"
        print(f"  {label:9s}: {d['mean']:+.3f} ± {d['se']:.3f}   "
              f"(sign {d['n_positive']}/{d['n_seeds']} positive)   [{vals}]{marker}")

    print(f"\nCo-primary gates = ΔB−base and ΔB−D (ΔB−A reported but NOT load-bearing — A degrades).")
    print(f"Effect-size pass (mean ΔB−base & ΔB−D ≥ {args.effect_threshold}): "
          f"{verdict['effect_size_pass']}")
    print(f"Sign-consistency pass (B>base & B>D per seed): {verdict['sign_consistency_pass']}  "
          f"{verdict['sign_consistency']}")
    print(f"\n>>> POSITIVE RESULT (locked decision rule): {verdict['positive_result']}\n")

    out = RESULTS / f"multiseed_verdict_{args.out_tag}_k{args.k}.json"
    out.write_text(json.dumps({"k": args.k, "summary": summary, "verdict": verdict}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
