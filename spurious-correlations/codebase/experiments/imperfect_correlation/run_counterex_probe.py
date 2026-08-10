"""Probe analogue of the fine-tune counterexample-injection experiment (local-only, no Modal/GPU).

Mirrors experiments/imperfect_correlation/run_counterex_ext.py exactly, swapping the fine-tune
instrument for the frozen linear probe:

  * SAME 20 confounded pairs, SAME counts {0,1,2,4,8,16,32}, SAME seeds {0,1,2}, SAME train sizes
    N in {75,150,300}, SAME target (weaker/intended feature) per pair (all read from the fine-tune's
    results/counterex_ext_raw.jsonl).
  * SAME injection convention: imperfect.build_train_target_parts with dose = count/N, target-aware
    (decorrelate the DOMINANT feature; inject `count` counterexamples per class carrying the intended
    feature). SAME fixed-eval pool split: imperfect.pools_fixed_eval(grid, 100) -> the first 100 texts
    per off-diagonal cell are the held-out readout, the remainder is the injection pool. The counterexample
    TEXT sampling is reproduced byte-for-byte from build_train_target_parts by drawing the same rng
    index sequence (random.sample selects positions), then gathering those texts' cached activations.
  * Instrument: mean-pooled activations (the read the fine-tune-comparison uses), fit with
    probe_fit.fit_grid, uncontrolled, C=0.1, at the SHIPPED global mean-pooled layer 27
    (results/probe_full_values.jsonl: rep=mean -> layer 27 for all pairs), 5 internal seeds (the frozen
    read_extraction instrument). Held-out readout = metric.py oriented log-odds toward f1, then oriented
    toward the target: read_bounded_target = bounded if target=="f1" else 1 - bounded.

Per (pair, N, count, data_seed) we build the injected training set, fit the probe, and record the
held-out bounded prob-of-following-intended (mean over the 5 internal probe seeds). Writes
results/counterex_probe/counterex_probe_raw.jsonl.

METHODOLOGY CHOICES (flagged):
  * Layer 27 / C=0.1 / mean-pooled / uncontrolled: the frozen probe instrument used for the
    fine-tune-vs-probe comparison elsewhere. Uncontrolled matches the fine-tune (which applies no
    length/topic control).
  * probe_fit.fit_one does an internal 75/25 stratified (by class) train/val split, so ~25% of the
    injected counterexamples land in the probe's held-in val fold at fit time rather than in the fit.
    This is intrinsic to the frozen instrument; it slightly dampens the low-count response vs a
    fit-on-everything probe. Reported as-is (the point is the frozen instrument's behaviour).

Usage:
  cd codebase
  PYTHONPATH=experiments/imperfect_correlation:lib .venv/bin/python \
      experiments/imperfect_correlation/run_counterex_probe.py
"""
from __future__ import annotations

import json
import random
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)  # perfect-separation val-CE logs
np.seterr(divide="ignore", invalid="ignore")

import config
import dataset
import imperfect as IMP
import metric
import probe_fit

HERE = Path(__file__).resolve().parent
RAW_FT = config.RESULTS_DIR / "counterex_ext_raw.jsonl"
OUT_DIR = config.RESULTS_DIR / "counterex_probe"
MAP = OUT_DIR / "mean_npz_map.json"
RAW = OUT_DIR / "counterex_probe_raw.jsonl"

COUNTS = [0, 1, 2, 4, 8, 16, 32]
SEEDS = [0, 1, 2]
NS = [75, 150, 300]
EVAL_PER_CELL = 100          # identical to run_counterex_ext.EVAL_PER_CELL
LAYER = 21                   # frozen mean-pooled readout layer used across the post's ladder / additive
                             # fit (probe_layer_profiles.json frozen_layer:21; refit_v2_meanpooled.json,
                             # per_layer_law_v2_meanpooled.json, probe_l21_meanpooled in merged_readouts_v2)
PROBE_C = 0.1
N_PROBE_SEEDS = 5            # frozen read_extraction instrument
CELLS = ["A", "B", "held_f1pos", "held_f1neg"]
UNCONTROLLED = probe_fit.Control("uncontrolled")


def ft_meta() -> dict:
    """target / delta_pair / abs_gap per pair from the fine-tune raw file (identical source)."""
    meta = {}
    for r in map(json.loads, open(RAW_FT)):
        meta[r["pair"]] = {"target": r["target"], "delta_pair": r["delta_pair"], "abs_gap": r["abs_gap"]}
    return meta


def injected_indices(nA: int, nB: int, n_pool_a: int, n_pool_b: int, n: int, dose: float,
                     seed: int, target: str):
    """Reproduce imperfect.build_train_target_parts' selection as INDICES (random.sample picks positions,
    so drawing range(len) with the same rng state yields the indices of the same texts)."""
    rng = random.Random(f"rhofull|{target}|{n}|{dose}|{seed}")
    n_off = round(dose * n)
    n_diag = n - n_off
    inj_a_cell = "held_f1pos" if target == "f1" else "held_f1neg"
    inj_b_cell = "held_f1neg" if target == "f1" else "held_f1pos"
    a_diag = rng.sample(range(nA), n_diag)
    a_inj = rng.sample(range(n_pool_a), n_off)
    b_diag = rng.sample(range(nB), n_diag)
    b_inj = rng.sample(range(n_pool_b), n_off)
    return a_diag, a_inj, b_diag, b_inj, inj_a_cell, inj_b_cell, n_off


def fit_probe_bounded(acts_train_A, acts_train_B, acts_eval_pos, acts_eval_neg) -> dict:
    """One probe fit at layer LAYER on injected training activations; held-out readout on the fixed eval
    pool. Uses fit_grid's per-seed fit_one at a single layer, averaged over N_PROBE_SEEDS."""
    y = np.array([0] * acts_train_A.shape[0] + [1] * acts_train_B.shape[0])
    Xtr = np.concatenate([acts_train_A, acts_train_B], axis=0)          # [ntr, hidden]
    Xho = np.concatenate([acts_eval_pos, acts_eval_neg], axis=0)        # [nho, hidden]
    signs = np.array([1] * acts_eval_pos.shape[0] + [-1] * acts_eval_neg.shape[0])
    prep = probe_fit.LayerPrep(Xtr, Xho, None, 1.0)
    per_seed = [probe_fit.fit_one(prep, y, signs, PROBE_C, s) for s in range(N_PROBE_SEEDS)]
    agg = probe_fit.aggregate_seeds(per_seed)
    return {"raw_mean": agg["raw_mean"], "bounded_mean": agg["bounded_mean"],
            "hard_mean": agg["hard_mean"], "raw_seeds": agg["raw_seeds"],
            "bounded_seeds": agg["bounded_seeds"]}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META = ft_meta()
    npz_map = json.loads(MAP.read_text())
    pairs = sorted(npz_map)
    assert set(pairs) == set(META), f"pair mismatch: {set(META) - set(pairs)}"

    done = set()
    if RAW.exists():
        done = {(r["pair"], r["n"], r["count"], r["seed"]) for r in map(json.loads, open(RAW))}

    for p in pairs:
        grid = dataset.load_grid(p)
        with np.load(npz_map[p]) as z:
            acts = {c: z[c][:, LAYER, :] for c in CELLS}          # mean-pooled, layer 27 -> [n, hidden]
        # fixed-eval split at the ACTIVATION level (mirrors imperfect.pools_fixed_eval on records)
        eval_pos = acts["held_f1pos"][:EVAL_PER_CELL]
        eval_neg = acts["held_f1neg"][:EVAL_PER_CELL]
        pool_pos = acts["held_f1pos"][EVAL_PER_CELL:]              # injection pool (index i -> full-cell EVAL+i)
        pool_neg = acts["held_f1neg"][EVAL_PER_CELL:]
        pools = {"held_f1pos": pool_pos, "held_f1neg": pool_neg}
        nA, nB = acts["A"].shape[0], acts["B"].shape[0]
        target = META[p]["target"]
        for n in NS:
            for c in COUNTS:
                for s in SEEDS:
                    if (p, n, c, s) in done:
                        continue
                    dose = c / n
                    a_diag, a_inj, b_diag, b_inj, inj_a_cell, inj_b_cell, n_off = injected_indices(
                        nA, nB, pool_pos.shape[0], pool_neg.shape[0], n, dose, s, target)
                    A_tr = np.concatenate([acts["A"][a_diag], pools[inj_a_cell][a_inj]], axis=0)
                    B_tr = np.concatenate([acts["B"][b_diag], pools[inj_b_cell][b_inj]], axis=0)
                    res = fit_probe_bounded(A_tr, B_tr, eval_pos, eval_neg)
                    b = res["bounded_mean"]
                    bt = b if target == "f1" else 1 - b
                    raw = res["raw_mean"]
                    rt = raw if target == "f1" else -raw
                    row = {"pair": p, "n": n, "count": c, "seed": s, "target": target,
                           "delta_pair": META[p]["delta_pair"], "abs_gap": META[p]["abs_gap"],
                           "n_off": n_off, "layer": LAYER,
                           "read_bounded_target": bt, "read_raw_target": rt,
                           "bounded_mean_f1": b, "raw_mean_f1": raw,
                           "bounded_seeds_f1": res["bounded_seeds"]}
                    with open(RAW, "a") as f:
                        f.write(json.dumps(row) + "\n")
            print(f"  {p:55s} N={n} done", flush=True)
    print(f"done -> {RAW}")


if __name__ == "__main__":
    main()
