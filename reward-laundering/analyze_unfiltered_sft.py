"""Analyze the UNFILTERED SFT null control: each variant's neutral-prompt subset-sum accuracy vs the
base model and the reward-laundering (B-GRPO) result.

Reads the endpoint-eval JSONL from run_endpoint_evals.py (base row + per-(seed, step) rows carrying
per-problem `rates`, `passk`, token stats). The two unfiltered variants share condition "b" (the
coupling prompt), so they are separated by TAG: "onpolicy" vs "base" substrings. For each variant it
reports the across-seed mean ± across-seed SE of the last-k-checkpoint neutral accuracy, the raw
per-seed endpoints, pass@1 vs pass@k (collapse guard), CoT-token length, parse-error rate, and the
fraction of the B-GRPO gain the variant reproduces.

Interpretation:
  unfiltered ≈ base (0.367)  -> the gain needs selection / RL credit assignment (mere imitation of
                                on-distribution reasoning is not enough) — strengthens reward-laundering.
  unfiltered ≈ B-GRPO (0.579)-> the gain is largely mere imitation of self-generated reasoning.
  in between                 -> partial; report the fraction reproduced.

Usage: .venv/bin/python analyze_unfiltered_sft.py --evals 'results/endpoint_evals_unf_*.jsonl' --k 3
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict

BASE_HEADLINE = 0.367
B_GRPO_BANKED = 0.579
A_GRPO_BANKED = 0.230


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _mean_se(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return m, math.sqrt(var / n)


def _variant_of(tag: str) -> str | None:
    if tag is None:
        return None
    if "onpolicy" in tag:
        return "on-policy unfiltered"
    if "base" in tag:
        return "base-rollout unfiltered"
    return None


def _last_k_by_seed(rows, variant, field, k):
    """{seed: mean-over-last-k of a scalar per-checkpoint field} for one variant."""
    by_seed_step = defaultdict(dict)
    for r in rows:
        if _variant_of(r.get("tag")) == variant and r.get("seed") is not None and r.get(field) is not None:
            by_seed_step[r["seed"]][r["step"]] = r[field]
    return {seed: _mean([sv[s] for s in sorted(sv)[-k:]]) for seed, sv in by_seed_step.items()}


def _rates_last_k_by_seed(rows, variant, k):
    """{seed: per-problem rate vector averaged over the last-k checkpoints} for one variant."""
    by_seed_step = defaultdict(dict)
    for r in rows:
        if _variant_of(r.get("tag")) == variant and r.get("seed") is not None and r.get("rates"):
            by_seed_step[r["seed"]][r["step"]] = r["rates"]
    out = {}
    for seed, sr in by_seed_step.items():
        vecs = [sr[s] for s in sorted(sr)[-k:]]
        n = len(vecs[0])
        out[seed] = [_mean([v[i] for v in vecs]) for i in range(n)]
    return out


def analyze(rows, k=3):
    base_row = next((r for r in rows if r.get("condition") == "base"), None)
    base_rates = base_row["rates"] if base_row else None
    base_acc = _mean(base_rates) if base_rates else BASE_HEADLINE
    base_passk = base_row.get("passk") if base_row else None

    out = {"k": k, "base_accuracy": base_acc, "base_passk": base_passk,
           "b_grpo": B_GRPO_BANKED, "a_grpo": A_GRPO_BANKED, "variants": {}}
    for variant in ("on-policy unfiltered", "base-rollout unfiltered"):
        acc_by_seed = _last_k_by_seed(rows, variant, "accuracy", k)
        if not acc_by_seed:
            continue
        passk_by_seed = _last_k_by_seed(rows, variant, "passk", k)
        tok_by_seed = _last_k_by_seed(rows, variant, "mean_sample_tokens", k)
        perr_by_seed = _last_k_by_seed(rows, variant, "parse_error_rate", k)
        rates_by_seed = _rates_last_k_by_seed(rows, variant, k)

        # Paired-vs-base delta per seed (per-problem paired), then across seeds.
        paired = {}
        if base_rates:
            for s, rv in rates_by_seed.items():
                n = min(len(rv), len(base_rates))
                paired[s] = _mean([rv[i] - base_rates[i] for i in range(n)])
        m_acc, se_acc = _mean_se(list(acc_by_seed.values()))
        m_pk, se_pk = _mean_se(list(passk_by_seed.values()))
        m_d, se_d = _mean_se(list(paired.values())) if paired else (float("nan"), float("nan"))
        gain_frac = ((m_acc - base_acc) / (B_GRPO_BANKED - base_acc)
                     if B_GRPO_BANKED != base_acc else float("nan"))
        out["variants"][variant] = {
            "n_seeds": len(acc_by_seed),
            "pass1_mean": m_acc, "pass1_se": se_acc, "pass1_per_seed": acc_by_seed,
            "passk_mean": m_pk, "passk_se": se_pk, "passk_per_seed": passk_by_seed,
            "delta_vs_base_mean": m_d, "delta_vs_base_se": se_d, "delta_per_seed": paired,
            "gain_fraction_of_bgrpo": gain_frac,
            "mean_cot_tokens": _mean(list(tok_by_seed.values())),
            "parse_error_rate": _mean(list(perr_by_seed.values())),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", required=True, help="endpoint_evals JSONL path or glob")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default="results/unfiltered_sft_verdict.json")
    args = ap.parse_args()
    paths = sorted(glob.glob(args.evals))
    if not paths:
        raise SystemExit(f"no endpoint files match {args.evals}")
    rows = []
    for p in paths:
        with open(p) as f:
            rows.extend(json.loads(line) for line in f if line.strip())

    v = analyze(rows, k=args.k)
    print(f"\n=== Unfiltered SFT null control (last-{v['k']}) ===")
    print(f"  reference: base {v['base_accuracy']:.3f} (pass@k {v['base_passk']})  |  "
          f"B-GRPO {v['b_grpo']:.3f} (reward-laundering)  |  A-GRPO {v['a_grpo']:.3f}\n")
    for variant, d in v["variants"].items():
        per_seed = ", ".join(f"{s}:{x:.3f}" for s, x in sorted(d["pass1_per_seed"].items()))
        print(f"  {variant}  (n={d['n_seeds']} seeds)")
        print(f"    pass@1 neutral acc : {d['pass1_mean']:.3f} ± {d['pass1_se']:.3f}   per-seed {per_seed}")
        print(f"    pass@k             : {d['passk_mean']:.3f} ± {d['passk_se']:.3f}")
        print(f"    Δ vs base (paired) : {d['delta_vs_base_mean']:+.3f} ± {d['delta_vs_base_se']:.3f}")
        print(f"    gain fraction of B-GRPO gain (+{v['b_grpo']-v['base_accuracy']:.3f}) : "
              f"{d['gain_fraction_of_bgrpo']:.2f}")
        print(f"    mean CoT tokens {d['mean_cot_tokens']:.0f} | parse-error {d['parse_error_rate']:.3f}\n")
    from pathlib import Path
    Path(args.out).write_text(json.dumps(v, indent=2, default=str))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
