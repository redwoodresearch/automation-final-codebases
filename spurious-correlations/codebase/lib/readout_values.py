"""Extract the per-pair frozen readout values from raw probe / fine-tune results, and the
win/loss/near-tie verdict convention (canonical design) — shared by the anchor fit (stage 0)
and every later extension readout.

Probe value:      heldout_raw mean over seeds at ONE recorded global layer.
Fine-tune value:  heldout_raw at the training-loss-plateau checkpoint, pinned LR, mean over seeds.
Verdict:          win/loss = same sign in all seeds AND 95% bootstrap CI (over held-out texts of
                  the per-text seed-mean d) excluding 0; else near-tie. Gap strata: near-tie
                  |Δ| < 3 nats, mid-gap 3–8, well-separated >= 8.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import finetune
import probe


# ------------------------------------------------------------------------------- probe
def probe_pair_values(probe_results: dict[str, dict], layer: int) -> dict[str, float]:
    return {pk: probe.aggregate_layers(res)[layer].heldout_raw_mean for pk, res in probe_results.items()}


# ------------------------------------------------------------------------------- fine-tune
def load_ft_rows(grid_path: Path, lr: float | None = finetune.PINNED_LR) -> list[dict]:
    rows = [json.loads(l) for l in open(grid_path) if l.strip()]
    rows = [r for r in rows if r.get("curve")]
    if lr is not None:
        rows = [r for r in rows if r["lr"] == lr]
    return rows


def plateau_step(row: dict) -> int:
    """The pinned checkpoint for one fine-tune row (training-signal-only; fit_law.py convention)."""
    return finetune.loss_plateau_step(row["train_losses"], finetune.FT_EVAL_STEPS) or finetune.FT_MAX_STEPS


def _row_checkpoint(row: dict, key: str = "heldout_raw"):
    return finetune.checkpoint_value(row["curve"], plateau_step(row), key)


def ft_pair_values(rows: list[dict]) -> dict[str, float]:
    """Per-pair mean-over-seeds heldout_raw at the plateau checkpoint (one LR expected in rows)."""
    by_pair: dict[str, list[float]] = {}
    for r in rows:
        by_pair.setdefault(r["pair"], []).append(_row_checkpoint(r))
    return {pk: float(np.mean(v)) for pk, v in by_pair.items()}


def ft_seed_values(rows: list[dict]) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for r in rows:
        out.setdefault(r["pair"], {})[r["seed"]] = float(_row_checkpoint(r))
    return out


def ft_seed_stats(rows_for_pair: list[dict]) -> list[dict]:
    """Per-seed plateau-checkpoint stats, incl. the constant-label-collapse flag (the model emits
    one label confidently EVERYWHERE: unoriented d has one sign on ~all held texts and hard
    accuracy ~0.5 after orientation). Same convention as the stage 0 tracer analysis."""
    out = []
    for r in sorted(rows_for_pair, key=lambda r: r["seed"]):
        step = plateau_step(r)
        cp = next(c for c in r["curve"] if c["step"] == step)
        d = np.asarray(cp["d_values"])  # UNORIENTED d = logit(X) - logit(Y); [pos..., neg...]
        frac_same_sign = float(max((d > 0).mean(), (d < 0).mean()))
        out.append({
            "seed": r["seed"], "plateau_step": step,
            "train_acc": cp["train_acc"], "train_correct_logp": cp["train_correct_logp"],
            "heldout_raw": cp["heldout_raw"], "heldout_acc": cp["heldout_acc"],
            "frac_same_unoriented_sign": frac_same_sign,
            "mean_abs_d": float(np.abs(d).mean()),
            "collapse_suspect": bool(frac_same_sign > 0.95 and cp["heldout_acc"] < 0.6),
        })
    return out


# ------------------------------------------------------------------------------- verdicts
def stratum(abs_delta: float) -> str:
    """Gap strata thresholds match the shipped salience_flip canonical_pairs convention."""
    return "well_separated" if abs_delta >= 8.0 else ("near_tie" if abs_delta < 3.0 else "mid_gap")


def ft_verdict(rows_for_pair: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """Win/loss/near-tie for one pair from its pinned-LR seed rows.

    Oriented d-values (metric convention: d * f1_sign, positive = follows f1) are taken per text at
    each seed's plateau checkpoint; the bootstrap resamples HELD-OUT TEXTS of the per-text
    seed-mean. win/loss requires all-seed sign agreement AND the 95% CI excluding 0.
    """
    per_seed = []
    for r in sorted(rows_for_pair, key=lambda r: r["seed"]):
        step = plateau_step(r)
        d = np.asarray(finetune.checkpoint_value(r["curve"], step, "d_values"), dtype=float)
        n_pos = len(d) // 2  # heldout order: [held_f1pos..., held_f1neg...] (dataset.build_inputs)
        signs = np.array([1.0] * n_pos + [-1.0] * (len(d) - n_pos))
        oriented = d * signs
        # Fail fast if the equal-held-cell assumption behind n_pos ever breaks: the recomputed
        # oriented mean must reproduce the remotely-computed heldout_raw at this checkpoint.
        remote = finetune.checkpoint_value(r["curve"], step, "heldout_raw")
        assert abs(oriented.mean() - remote) < 1e-6, (
            f"oriented-mean mismatch (seed {r['seed']}): recomputed {oriented.mean():.6f} vs "
            f"remote {remote:.6f} — held cells probably not equal-count/ordered as assumed")
        per_seed.append(oriented)
    mat = np.vstack(per_seed)                    # [n_seeds, n_texts] oriented
    seed_means = mat.mean(axis=1)
    text_means = mat.mean(axis=0)                # per-text seed-mean
    rng = np.random.default_rng(seed)
    boots = np.array([text_means[rng.integers(0, len(text_means), len(text_means))].mean()
                      for _ in range(n_boot)])
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    delta = float(text_means.mean())
    sign_consistent = bool(np.all(np.sign(seed_means) == np.sign(seed_means[0])) and seed_means[0] != 0)
    decisive = sign_consistent and (lo > 0 or hi < 0)
    return {
        "delta": delta,
        "seed_means": [float(x) for x in seed_means],
        "ci95": [lo, hi],
        "sign_consistent": sign_consistent,
        "verdict": ("follows_f1" if delta > 0 else "follows_f2") if decisive else "near_tie",
        "stratum": stratum(abs(delta)),
        "n_texts": int(mat.shape[1]),
    }
