"""Cross-model probe sweep (Stage 5 Phase 0, `cross_model_probe`): the model registry + the
cross-model probe-ladder assembly.

The instrument is unchanged across models: the probe reads each model's RAW-TEXT last-token activations
(``wrap='none'``, ``representation='last'`` — no prompt, no chat template) on the SAME shipped 2x2 grids
(model-agnostic text), fits a logistic probe on the diagonal (A vs B) cells with per-model held-in layer
selection (min held-in validation cross-entropy — `probe.select_global_layer`), reads the disagreement
cells -> per-pair ``Δ_pr(model)``, and refits the additive law (`transitivity.fit_strengths`) ->
per-feature ``s_pr(model)``. We then compare RANKINGS / winner-signs vs the Qwen3-1.7B reference (never
raw nats — the logit scale differs across models and sizes; see INSTRUCTIONS).

This module holds only model-agnostic registry + assembly logic; the GPU probe call lives in
`modal_app.probe_logprob_diff` (now `model_id`/`dtype` parameterized) and the entry points in
`experiments/cross_model/`.
"""
from __future__ import annotations

import numpy as np
from attrs import frozen

import congruence_properties as cp
import probe
import rank_stats as rs
import readout_invariance as ri
import transitivity


@frozen
class ModelSpec:
    key: str            # short id used in result files / plots (no slashes)
    model_id: str       # HuggingFace identifier
    label: str          # human-readable (plots / write-up)
    family: str         # "Qwen3" | "Llama-3.1" | "OLMo-2"
    params_b: float     # parameter count in billions (the scale axis)
    post_trained: bool  # True = instruct/post-trained checkpoint, False = pretrained base
    role: str           # what axis this model probes (reference / scale / family / corpus / control)
    dtype: str = "bfloat16"


REFERENCE_KEY = "qwen3_1p7b"

# The set, chosen to span the three axes that matter — FAMILY (Qwen vs Llama vs OLMo), SCALE
# (1.7B -> 8B -> 32B), and PRETRAINING CORPUS (OLMo's open corpus). All Qwen points share a checkpoint
# type (post-trained) so the Qwen scale axis is internally clean; Llama/OLMo are BASE (pretrained). The
# base-vs-post-trained gap that otherwise sits on the family/corpus axes is BOUNDED by the
# `qwen3_1p7b_base` control (how far the probe ladder moves base->post-trained on the same 1.7B). The
# raw-text probe read is a pretrained representation, so post-training is expected to move it little —
# the control measures that directly. Qwen3-32B has no base variant on HF, which is why the 32B point
# is post-trained.
MODELS: dict[str, ModelSpec] = {
    "qwen3_1p7b": ModelSpec("qwen3_1p7b", "Qwen/Qwen3-1.7B", "Qwen3-1.7B", "Qwen3", 1.7, True,
                            "reference (the project's committed model)"),
    "qwen3_8b": ModelSpec("qwen3_8b", "Qwen/Qwen3-8B", "Qwen3-8B", "Qwen3", 8.2, True,
                          "scale axis (Qwen, mid)"),
    "qwen3_32b": ModelSpec("qwen3_32b", "Qwen/Qwen3-32B", "Qwen3-32B", "Qwen3", 32.8, True,
                           "scale axis (Qwen, large)"),
    "llama31_8b": ModelSpec("llama31_8b", "NousResearch/Meta-Llama-3.1-8B", "Llama-3.1-8B", "Llama-3.1",
                            8.0, False, "family axis (different family, ~same scale)"),
    "olmo2_7b": ModelSpec("olmo2_7b", "allenai/OLMo-2-1124-7B", "OLMo-2-7B", "OLMo-2", 7.3, False,
                          "corpus axis (different pretraining corpus — the key architecture-vs-data test)"),
    # base controls: bound the base-vs-post-training confound that otherwise sits on the family/corpus
    # axes, AND give a clean ALL-BASE 8B family comparison (qwen3_8b_base vs llama31_8b vs olmo2_7b).
    "qwen3_1p7b_base": ModelSpec("qwen3_1p7b_base", "Qwen/Qwen3-1.7B-Base", "Qwen3-1.7B-Base", "Qwen3",
                                 1.7, False, "post-training-invariance control @1.7B (vs the reference)"),
    "qwen3_8b_base": ModelSpec("qwen3_8b_base", "Qwen/Qwen3-8B-Base", "Qwen3-8B-Base", "Qwen3", 8.2, False,
                               "post-training control @8B + clean all-base family comparison"),
}

# ----------------------------------------------------------------------------- fine-tune memory config (Phase 1)
# A LoRA *fine-tune* retains activations for backprop (more than a forward), so the ≥7B models use gradient
# checkpointing (result-neutral: it only trades compute for activation memory). The 32B additionally runs at
# concurrency 1 (its bf16 forward already fills most of the B200 — the Phase-0 OOM-at-concurrency>1 gotcha).
# The 1.7B fits without checkpointing. All models fine-tune in their registry dtype (bf16) — identical to the
# cross-model PROBE sweep, where the 1.7B bf16 read tied back to the committed fp32 ranking at Spearman 0.95.
FT_GRAD_CKPT_KEYS = {"qwen3_8b", "qwen3_32b", "llama31_8b", "olmo2_7b", "qwen3_8b_base"}
FT_CONCURRENCY1_KEYS = {"qwen3_32b"}


def ft_grad_ckpt(key: str) -> bool:
    return key in FT_GRAD_CKPT_KEYS


# The models whose ladders enter the headline cross-model ranking table (every model vs the reference).
SWEEP_KEYS = ["qwen3_1p7b", "qwen3_8b", "qwen3_32b", "llama31_8b", "olmo2_7b"]
# The post-trained within-Qwen scale axis (consistent checkpoint type) — for the scale-sharpening test.
QWEN_SCALE_AXIS = ["qwen3_1p7b", "qwen3_8b", "qwen3_32b"]
# The clean ALL-BASE ~7-8B family/corpus comparison (no base-vs-post confound).
BASE_8B_FAMILY = ["qwen3_8b_base", "llama31_8b", "olmo2_7b"]
# Base-vs-post-trained pairs (the post-training-invariance check), keyed post -> base.
POST_BASE_PAIRS = {"qwen3_1p7b": "qwen3_1p7b_base", "qwen3_8b": "qwen3_8b_base"}


def spec(key: str) -> ModelSpec:
    return MODELS[key]


# Probe-scale gap strata. The probe Δ_pr scale is ~half the fine-tune scale (1.7B reference: |Δ_pr| maxes
# at ~8 vs |Δ_ft| ~18), so `ladder_compare.gap_stratum`'s fine-tune thresholds (>=8 large) would label
# almost everything mid/near. Calibrated to the natural breaks in the 1.7B reference probe distribution:
# a decisive cluster |Δ_pr| >= 5 (sigmoid(5)=0.993) and a near-tie cluster |Δ_pr| < 1.5 (sigmoid<0.82).
# Pre-registered on the 1.7B reference (the baseline), before any other model is run.
PROBE_LARGE_GAP = 5.0
PROBE_NEAR_TIE = 1.5


def probe_gap_stratum(abs_delta: float) -> str:
    """large = well-separated on the 1.7B PROBE (the should-never-flip anchors); near = probe near-tie
    (flips allowed); mid otherwise."""
    if abs_delta >= PROBE_LARGE_GAP:
        return "large"
    return "near" if abs_delta < PROBE_NEAR_TIE else "mid"


# ----------------------------------------------------------------------------- the curated pair set
# Five language pairs spanning |Δ_pr| (near-tie -> large), chosen so the additive refit covers the 21st
# feature (`language`, the cross-lingual outlier) and lifts the two otherwise degree-1 roster features
# (formal_register, parenthetical_asides) to degree 2.
CURATED_LANGUAGE_PAIRS: list[str] = [
    "language__hedging", "language__domain_jargon", "language__formal_register",
    "language__narrative_vs_expository", "language__parenthetical_asides",
]


def roster_pairs() -> list[str]:
    """The 33 ROSTER pairs that feed the additive-law refit -> per-feature `s_pr`: the 28 gap-spanning
    `readout_invariance.CURATED_PAIRS` (near-ties oversampled + mid + large-gap anchors; covers 20 of the
    21 features) plus 5 `language` pairs (the 21st feature + the cross-lingual travel test). This subset
    reproduces the committed full-402-pair `s_pr` ranking at Spearman 0.93 (in-sample additive R^2 0.95),
    so the curated refit is a faithful stand-in — and every model (incl. the 1.7B reference) is refit from
    this SAME pair set, a like-for-like comparison."""
    return list(ri.CURATED_PAIRS) + CURATED_LANGUAGE_PAIRS


def congruence_pairs() -> list[str]:
    """The 18 ASSOCIATED / prior-association pairs analyzed for winner-agreement + flip model-dependence:
    `congruence_properties.all_congruence_pairs()` (21) minus the 3 `DROP_PAIRS` that failed the
    buildability gate. These use off-roster features (gender, finance, legal, ...), so they are NOT refit
    into `s` — they are scored per-pair (does the prior-association winner travel / flip across models?)."""
    keys = [f"{a}__{b}" for a, b, _seed in cp.all_congruence_pairs()]
    return [k for k in keys if k not in cp.DROP_PAIRS]


def curated_pairs() -> list[str]:
    """The full pre-registered set: 33 roster + 18 congruence = 51 pairs (probed on every model)."""
    return roster_pairs() + congruence_pairs()


# ----------------------------------------------------------------------------- ladder assembly
# Layer rule. The diagonal is trivially separable (hidden >> texts/cell), so held-in validation CE
# saturates across the deep layers and a min-CE argmin is near-arbitrary among them (design-review C2). So
# we do NOT let argmin-CE carry the claim. PRIMARY = the FINAL hidden state (post-final-norm) — a FIXED,
# unambiguous, architecture-agnostic read position; it is where the committed 1.7B probe ladder lives, and
# the read's SIGN is stable across the whole deep band. ROBUSTNESS = the deep-band [0.75, 1.0] MEDIAN and
# the (degenerate) argmin-CE layer; the headline ranking/winner conclusions are reported under all three.
DEEP_BAND = (0.75, 1.0)


def n_hidden_states(probe_result: dict) -> int:
    """Number of hidden states in the probe result (= transformer layers + 1; valid indices 0..n-1)."""
    return int(probe_result["n_layers"])


def deep_band_layers(probe_result: dict) -> list[int]:
    """Hidden-state indices in the [0.75, 1.0] fractional-depth band (the deep, sign-stable layers)."""
    last = n_hidden_states(probe_result) - 1
    lo = int(round(DEEP_BAND[0] * last))
    return list(range(lo, last + 1))


def final_layer(probe_result: dict) -> int:
    return n_hidden_states(probe_result) - 1


def per_pair_delta(probe_result: dict, layer: int) -> float:
    """Seed-mean held-out oriented log-odds (raw nats, toward f1) at one layer = the per-pair Δ_pr."""
    raws = [s["layers"][layer]["heldout_raw"] for s in probe_result["per_seed"]]
    return float(np.mean(raws))


def primary_delta(probe_result: dict) -> float:
    """PRIMARY per-pair Δ_pr = the seed-mean held-out log-odds at the FINAL hidden state."""
    return per_pair_delta(probe_result, final_layer(probe_result))


def band_delta(probe_result: dict) -> float:
    """ROBUSTNESS per-pair Δ_pr = MEDIAN over the deep-band layers of the seed-mean held-out log-odds."""
    return float(np.median([per_pair_delta(probe_result, L) for L in deep_band_layers(probe_result)]))


def per_pair_delta_seedwise(probe_result: dict, layer: int) -> list[float]:
    """Per-seed held-out log-odds at one layer (for the seed-SD / signal-to-noise decisiveness metric)."""
    return [s["layers"][layer]["heldout_raw"] for s in probe_result["per_seed"]]


def per_pair_diagnostics(probe_result: dict, layer: int) -> dict:
    """Held-in diagonal separability + held-out read at the selected layer for one pair (the
    measurement-validity gate: a probe that can't separate the diagonal has a meaningless read)."""
    per_seed = probe_result["per_seed"]
    L = layer
    val_acc = float(np.mean([s["layers"][L]["val_acc"] for s in per_seed]))
    val_ce = float(np.mean([s["layers"][L]["val_ce"] for s in per_seed]))
    raws = [s["layers"][L]["heldout_raw"] for s in per_seed]
    bnd = [s["layers"][L]["heldout_bounded"] for s in per_seed]
    return {"layer": L, "val_acc": val_acc, "val_ce": val_ce,
            "delta_pr": float(np.mean(raws)), "delta_pr_sd": float(np.std(raws, ddof=1)) if len(raws) > 1 else 0.0,
            "bounded": float(np.mean(bnd)), "n_seeds": len(per_seed),
            "sign_consistent": bool(len(set(np.sign(raws))) == 1 and np.sign(raws[0]) != 0)}


def select_layer(results_by_pair: dict[str, dict]) -> int:
    """Per-model global probe layer = min mean held-in validation CE over the pairs (never the held-out
    value). Reuses the committed `probe.select_global_layer`."""
    return probe.select_global_layer(results_by_pair)


def assemble_delta_ladder(results_by_pair: dict[str, dict], layer: int) -> dict[str, float]:
    """{pair_key: Δ_pr(model)} at one fixed layer."""
    return {p: per_pair_delta(r, layer) for p, r in results_by_pair.items()}


def assemble_band_ladder(results_by_pair: dict[str, dict]) -> dict[str, float]:
    """{pair_key: Δ_pr(model)} under the PRIMARY deep-band-median rule."""
    return {p: band_delta(r) for p, r in results_by_pair.items()}


def is_language_pair(pair: str) -> bool:
    return "language" in pair.split("__")


# ----------------------------------------------------------------------------- s_pr comparison vs reference
def refit_s(delta_ladder: dict[str, float], pairs: list[str]) -> dict[str, float]:
    """Per-feature s_pr refit (additive law) from a Δ ladder restricted to `pairs`."""
    return transitivity.fit_strengths({p: delta_ladder[p] for p in pairs if p in delta_ladder})["s"]


def additive_diagnostics(delta_ladder: dict[str, float], pairs: list[str]) -> dict:
    """In-sample R², residual SD, and leave-one-pair-out R² of the additive fit on `pairs` (if the law
    collapses in a model, that model's s_pr ranking is ill-defined — a first-class finding)."""
    sub = {p: delta_ladder[p] for p in pairs if p in delta_ladder}
    fit = transitivity.fit_strengths(sub)
    import ladder_compare
    return {"n_pairs": len(sub), "r2_insample": round(fit["r2_insample"], 3),
            "resid_sd": round(fit["resid_std"], 3), "loo_r2": round(ladder_compare.loo_r2(sub), 3)}


def s_spearman_vs_ref(model_delta: dict[str, float], ref_delta: dict[str, float], pairs: list[str],
                      n_boot: int = 2000, seed: int = 0) -> dict:
    """Spearman + Kendall of refit s_pr (model vs reference) over shared features, with a PAIR-REFIT
    bootstrap CI: resample `pairs` with replacement, refit s for BOTH ladders, recompute over the features
    present in both refits (respects the pair-graph coupling, unlike an i.i.d. feature bootstrap)."""
    def corr(pair_list, fn):
        sm, sr = refit_s(model_delta, pair_list), refit_s(ref_delta, pair_list)
        shared = sorted(set(sm) & set(sr))
        if len(shared) < 4:
            return None
        return fn([sm[f] for f in shared], [sr[f] for f in shared])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        samp = list(rng.choice(pairs, size=len(pairs), replace=True))
        v = corr(samp, rs.spearman)
        if v is not None and np.isfinite(v):
            boots.append(v)
    boots = np.array(boots)
    sm, sr = refit_s(model_delta, pairs), refit_s(ref_delta, pairs)
    shared = sorted(set(sm) & set(sr))
    return {"spearman": round(corr(pairs, rs.spearman), 3), "kendall": round(corr(pairs, rs.kendall), 3),
            "ci_lo": round(float(np.percentile(boots, 2.5)), 3), "ci_hi": round(float(np.percentile(boots, 97.5)), 3),
            "n_features": len(shared), "n_boot": int(boots.size)}


# ----------------------------------------------------------------------------- scale-free decisiveness (P4)
def decisiveness(probe_result: dict, layer: int) -> dict:
    """Scale-free per-pair decisiveness at one layer: signal-to-noise |Δ|/seed-SD and the WINNER-side hard
    accuracy = the argmax-follows-the-winner fraction = max(p, 1-p) where p = fraction following f1 (so it
    is winner-agnostic — a pair where f2 wins decisively scores high too). Both avoid the forbidden raw-nat
    cross-model magnitude comparison."""
    raws = np.array(per_pair_delta_seedwise(probe_result, layer))
    sd = float(raws.std(ddof=1)) if len(raws) > 1 else 0.0
    snr = abs(float(raws.mean())) / sd if sd > 1e-9 else float("inf")
    p_f1 = float(np.mean([probe_result["per_seed"][i]["layers"][layer]["heldout_acc"]
                          for i in range(len(probe_result["per_seed"]))]))
    return {"snr": snr, "hard_acc": max(p_f1, 1.0 - p_f1)}


def ladders_by_rule(results_by_pair: dict[str, dict], non_lang_for_argmin: list[str] | None = None) -> dict:
    """All three per-pair Δ ladders for one model — keyed by layer rule. 'final' is the PRIMARY; 'band'
    (deep-band median) and 'argmin_ce' are robustness checks. The argmin-CE layer is chosen over the
    non-language pairs only (language's CE profile is anomalous)."""
    argmin_src = ({p: results_by_pair[p] for p in non_lang_for_argmin if p in results_by_pair}
                  if non_lang_for_argmin else results_by_pair)
    argmin_layer = select_layer(argmin_src)
    return {
        "final": {p: primary_delta(r) for p, r in results_by_pair.items()},         # PRIMARY
        "band": assemble_band_ladder(results_by_pair),                              # robustness
        "argmin_ce": assemble_delta_ladder(results_by_pair, argmin_layer),          # robustness
        "argmin_ce_layer": argmin_layer,
    }
