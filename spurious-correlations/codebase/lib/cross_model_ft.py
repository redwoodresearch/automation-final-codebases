"""Cross-model FINE-TUNE strength ladder (Stage 5 Phase 1, `cross_model_finetune`): assemble the
per-model Δ_ft ladder from the fine-tune sweep, the gated diagonal, and the probe↔fine-tune comparison.

The instrument is the standard label-token fine-tune (modal_app.finetune_lora_and_eval; full-FT for the
1.7B reference + the 8B A3 control): fine-tune on the confounded diagonal, read the oriented held-out
log-odds Δ_ft at the training-loss-plateau checkpoint. We compare RANKINGS / winner-signs vs the Qwen3-1.7B
reference (never raw nats — the logit scale differs across models AND across fine-tune amounts, LoRA vs
full-FT). This module holds model-agnostic assembly + the reference/probe loaders; the GPU call lives in
`modal_app.finetune_lora_and_eval` and the entry points in `experiments/cross_model/`.

Reuses the probe-phase ranking machinery: `cross_model.refit_s` / `additive_diagnostics` /
`s_spearman_vs_ref` (ladder-agnostic) and `ladder_compare.gap_stratum` / `compare_ladder` (the fine-tune
|Δ_ft| scale, not the probe scale).
"""
from __future__ import annotations

import json

import numpy as np

import config
import cross_model as cm
import transitivity

FT_DIR = config.results_dir("cross_model_ft")


def ft_path(model_key: str, ft_type: str):
    return FT_DIR / f"ft_{model_key}_{ft_type}.jsonl"


def load_ft_rows(model_key: str, ft_type: str) -> list[dict]:
    p = ft_path(model_key, ft_type)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def assemble_ft_ladder(rows: list[dict], gate: bool = True) -> dict:
    """{pair: seed-mean Δ_ft} over the gated seeds, plus the per-pair gate bookkeeping. A (pair, seed) with
    a failed diagonal gate is dropped (its held read is meaningless); a pair with NO gated seed is dropped
    from the ladder entirely (reported in `dropped_pairs`)."""
    by_pair: dict[str, list[dict]] = {}
    for r in rows:
        by_pair.setdefault(r["pair"], []).append(r)
    ladder, gate_stats, dropped = {}, {}, []
    for pair, rs_ in by_pair.items():
        used = [r for r in rs_ if (r["gate_pass"] or not gate)]
        gate_stats[pair] = {"n_seeds": len(rs_), "n_gated_pass": sum(r["gate_pass"] for r in rs_),
                            "min_diag_acc": round(min(r["diag_acc"] for r in rs_), 4)}
        if not used:
            dropped.append(pair)
            continue
        ladder[pair] = float(np.mean([r["heldout_raw"] for r in used]))
    return {"ladder": ladder, "gate_stats": gate_stats, "dropped_pairs": dropped}


def ft_seed_sd(rows: list[dict], gate: bool = True) -> dict[str, float]:
    """Per-pair seed-to-seed SD of Δ_ft over the gated seeds (the scale-free decisiveness denominator)."""
    by_pair: dict[str, list[float]] = {}
    for r in rows:
        if r["gate_pass"] or not gate:
            by_pair.setdefault(r["pair"], []).append(r["heldout_raw"])
    return {p: float(np.std(v, ddof=1)) for p, v in by_pair.items() if len(v) > 1}


# ----------------------------------------------------------------------------- the 1.7B reference (committed full-FT)
def reference_ft_deltas() -> dict[str, float]:
    """The committed full-FT 1.7B per-pair Δ_ft (results/_pair_deltas_ft.json, 202 pairs, seed-mean)."""
    return json.loads((config.RESULTS_DIR / "_pair_deltas_ft.json").read_text())


def reference_ft_ladder(pairs: list[str]) -> dict[str, float]:
    """The 1.7B reference Δ_ft restricted/oriented to `pairs` (the roster), so the reference s_ft is refit
    from the SAME pair set as every other model (a like-for-like comparison; INSTRUCTIONS Part B)."""
    d = reference_ft_deltas()
    out = {}
    for p in pairs:
        a, b = p.split("__")
        v = transitivity.oriented_delta(d, a, b)
        if v is not None:
            out[p] = float(v)
    return out


# ----------------------------------------------------------------------------- the Phase-0 probe s_pr (for probe↔ft)
def load_probe_s_pr(model_key: str, with_language: bool = False) -> dict[str, float] | None:
    """The Phase-0 PROBE per-feature s_pr for a model, refit from the same roster, from the committed
    cross_model_analysis.json. Non-language by default (the confirmatory set; `language` is layer-fragile).
    Returns None if the model was not in the Phase-0 sweep."""
    p = config.results_dir("cross_model") / "cross_model_analysis.json"
    if not p.exists():
        return None
    analysis = json.loads(p.read_text())
    pm = analysis.get("per_model", {})
    if model_key not in pm:
        return None
    return dict(pm[model_key]["s_pr_nonlang"])


def probe_s_pr_reference_from_roster(pairs: list[str]) -> dict[str, float]:
    """Refit the 1.7B PROBE s_pr from the per-pair Δ_pr at the read layer over `pairs` — the same way the
    Phase-0 analysis built each model's s_pr. Used only if cross_model_analysis.json lacks a model; the
    normal path reads s_pr_nonlang straight out of that committed analysis."""
    probe = json.loads((config.results_dir("cross_model") / "probe_qwen3_1p7b.json").read_text())
    lad = {p: cm.primary_delta(probe[p]) for p in pairs if p in probe}
    return cm.refit_s(lad, [p for p in pairs if p in lad])
