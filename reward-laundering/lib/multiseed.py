"""Loaders + decision-rule aggregation for the multi-seed headline matrix.

Each (condition, seed) run writes:
  - results/rl_pilot_<cond>_<seedtag>_<stamp>_evals.jsonl   (per eval-step trajectory rows)
  - results/rl_pilot/metrics_<cond>_<seedtag>.jsonl          (per training-step rows)
  - results/rl_pilot_<cond>_<seedtag>_<stamp>_summary.json   (config + kept checkpoints)
and the endpoint driver writes results/endpoint_evals_<out>_<stamp>.jsonl.

`seedtag` is the run tag, e.g. "seed0". The decision rule summarises each
(condition, seed) by the mean neutral-prompt subset-sum accuracy over the last-k endpoint checkpoints
(on the large fixed eval set), then reports across-seed mean/SE and the paired B-vs-{base,A,D}
contrasts with sign consistency.
"""

from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

CONDITIONS = ("a", "b", "c", "d")
COND_LABEL = {
    "a": "A — exposure baseline",
    "b": "B — self-steering (coupling)",
    "c": "C — direct side reward",
    "d": "D — variance-matched placebo",
}


def _latest(pattern: str) -> str | None:
    files = sorted(glob.glob(str(RESULTS / pattern)))
    return files[-1] if files else None


def load_trajectory(cond: str, seedtag: str) -> list[dict]:
    """Per eval-step rows for one (condition, seedtag) run (latest file if several)."""
    f = _latest(f"rl_pilot_{cond}_{seedtag}_*_evals.jsonl")
    return [json.loads(l) for l in open(f) if l.strip()] if f else []


def load_metrics(cond: str, seedtag: str) -> list[dict]:
    f = RESULTS / "rl_pilot" / f"metrics_{cond}_{seedtag}.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []


def load_summary(cond: str, seedtag: str) -> dict | None:
    f = _latest(f"rl_pilot_{cond}_{seedtag}_*_summary.json")
    return json.loads(Path(f).read_text()) if f else None


def load_endpoint_rows(out_tag: str) -> list[dict]:
    """All endpoint-eval rows for an output tag (latest file)."""
    f = _latest(f"endpoint_evals_{out_tag}_*.jsonl")
    return [json.loads(l) for l in open(f) if l.strip()] if f else []


def per_seed_summary(endpoint_rows: list[dict], k: int = 3) -> dict:
    """Per (condition, seed): mean neutral accuracy over the last-k endpoint checkpoints.

    Returns {condition: {seed: {"acc": float, "steps": [...], "n_ckpts": int}}} plus the base
    accuracy under key ("base", None). Uses whichever kept steps were evaluated; takes the k with the
    largest step indices."""
    base_acc = None
    by_cond_seed: dict = {}
    for r in endpoint_rows:
        if r["condition"] == "base":
            base_acc = r["accuracy"]
            continue
        by_cond_seed.setdefault(r["condition"], {}).setdefault(r["seed"], []).append(r)
    out: dict = {"base_accuracy": base_acc, "conditions": {}}
    for cond, seeds in by_cond_seed.items():
        out["conditions"][cond] = {}
        for seed, rows in seeds.items():
            rows_sorted = sorted(rows, key=lambda r: r["step"])
            last_k = rows_sorted[-k:]
            accs = [r["accuracy"] for r in last_k]
            out["conditions"][cond][seed] = {
                "acc": sum(accs) / len(accs),
                "steps": [r["step"] for r in last_k],
                "n_ckpts": len(last_k),
                "per_ckpt_acc": accs,
            }
    return out


def _mean_se(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return (float("nan"), float("nan"))
    mean = sum(xs) / n
    se = statistics.stdev(xs) / (n ** 0.5) if n > 1 else 0.0
    return mean, se


def decision_rule(summary: dict, effect_threshold: float = 0.10) -> dict:
    """Apply the locked decision rule (with the Phase-2 refinement) to a per_seed_summary output.

    Refinement (driven by the pre-verdict seed-0 observation that A degrades *below*
    base): the two required, co-primary gates are ΔB−base and ΔB−D, NOT ΔB−A.
      1. ΔB−base ≥ effect_threshold (absolute gain over the no-op reference), B > base in every seed
         (≥ n−1 of n at n≥5); AND
      2. ΔB−D ≥ effect_threshold (coupling-isolating: D shares B's exact scaffold + reward variance,
         only the reward↔success correlation differs), B > D in every seed.
    ΔB−A is still computed and reported but is NOT load-bearing — because A degrades, ΔB−A is inflated
    by A's collapse rather than B's gain, so it is the *least* meaningful contrast. Reliability:
    across-seed mean/SE + the raw per-seed values for all three contrasts."""
    conds = summary["conditions"]
    base = summary["base_accuracy"]
    seeds = sorted(conds.get("b", {}).keys())
    per_seed = {}
    for s in seeds:
        row = {"base": base}
        for c in CONDITIONS:
            row[c] = conds.get(c, {}).get(s, {}).get("acc")
        row["dBA"] = (row["b"] - row["a"]) if (row["b"] is not None and row["a"] is not None) else None
        row["dBbase"] = (row["b"] - base) if (row["b"] is not None and base is not None) else None
        row["dBD"] = (row["b"] - row["d"]) if (row["b"] is not None and row["d"] is not None) else None
        per_seed[s] = row

    def contrast_stats(key):
        vals = [per_seed[s][key] for s in seeds if per_seed[s].get(key) is not None]
        mean, se = _mean_se(vals)
        n_pos = sum(1 for v in vals if v > 0)
        return {"mean": mean, "se": se, "n_seeds": len(vals), "n_positive": n_pos, "values": vals}

    cond_acc = {c: {"mean": _mean_se([conds[c][s]["acc"] for s in sorted(conds.get(c, {}))])[0],
                    "se": _mean_se([conds[c][s]["acc"] for s in sorted(conds.get(c, {}))])[1],
                    "per_seed": {s: conds[c][s]["acc"] for s in sorted(conds.get(c, {}))}}
                for c in CONDITIONS if conds.get(c)}

    dBA, dBbase, dBD = contrast_stats("dBA"), contrast_stats("dBbase"), contrast_stats("dBD")
    n = len(seeds)

    def sign_pass(c):
        # All seeds positive, or (with n>=5) at least n-1 of n.
        need = n - 1 if n >= 5 else n
        return c["n_positive"] >= need and c["n_seeds"] == n

    # Co-primary gates: ΔB−base and ΔB−D (ΔB−A is reported but NOT gated — see docstring).
    effect_size_pass = dBbase["mean"] >= effect_threshold and dBD["mean"] >= effect_threshold
    sign_pass_all = sign_pass(dBbase) and sign_pass(dBD)
    return {
        "seeds": seeds,
        "base_accuracy": base,
        "condition_accuracy": cond_acc,
        "per_seed": per_seed,
        "dBA": dBA, "dBbase": dBbase, "dBD": dBD,
        "gate_contrasts": ("dBbase", "dBD"),
        "effect_size_pass": effect_size_pass,
        "sign_consistency": {
            "BgtA": f"{dBA['n_positive']}/{dBA['n_seeds']}",
            "Bgtbase": f"{dBbase['n_positive']}/{dBbase['n_seeds']}",
            "BgtD": f"{dBD['n_positive']}/{dBD['n_seeds']}",
        },
        "sign_consistency_pass": sign_pass_all,
        "positive_result": effect_size_pass and sign_pass_all,
    }
