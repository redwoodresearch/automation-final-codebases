"""THE frozen merged readout table (stage 3 phase 0): one row per usable pair (663), merging the
v2-environment mass diagonal readouts (tags mass_diag_built / mass_diag_fitted, chat pass
included) with per-pair provenance and the machine-readable interpretation flags phase 1 consumes.

Sign discipline: every delta is carried in its on-disk orientation (delta > 0 = the classifier
followed f1 of the grid filename); nothing is re-oriented at load. Cross-population analyses
orient through the feature-role tags (f1/f2 + categories/types).

Provenance: ALL refit-input values are v2-environment (`readout_env='v2_env'`; the fired
conservative/operational recompute — results/complete_graph/env_drift_calibration.md). Committed
old-environment values are retained per pair in `delta_committed_oldenv` (fitted + heldout
categories; for the 6 user-turn-repaired refusal grids those are readouts OF THE OLD GRIDS —
`grid_version='committed_repaired_userturns'` — so they are provenance, not a same-grid drift
comparison). Old and new FT seed sets are never pooled.
"""

from __future__ import annotations

import json

import numpy as np

import config
import features
import probe
import readout_values as RV
from extension_properties import split_pair_key
from pooled_readouts import FROZEN_LAYER, READS_DIR, feature_category

# FT/probe winner-split classification (frozen selection criteria — set BEFORE the phase-1 refit;
# the full member lists are in results/complete_graph/ft_probe_split_classification.json):
#   read_location:     end-anchored trigger, last-token probe reversed by mean-pool (a read artifact)
#   robust_divergence: opposite winners, both |Δ| >= 8, mean AGREES with last, FT seed-consistent
#                      (a REAL representational divergence — the "monitor keys on a different
#                      feature than the policy" case; consumed by the phase-1 read-comparison)
#   noise:             the remaining splits (near-tie/mid)
_END_ANCHORS = {"trailing_signature", "ends_with_string"}


def classify_ft_probe_split(f1, f2, ft_delta, probe_last, probe_mean, ft_sign_consistent) -> str | None:
    if np.sign(ft_delta) == np.sign(probe_last):
        return None  # not a split
    if (_END_ANCHORS & {f1, f2}) and np.sign(probe_mean) != np.sign(probe_last):
        return "read_location"
    if abs(ft_delta) >= 8 and abs(probe_last) >= 8 and np.sign(probe_mean) == np.sign(probe_last) \
            and ft_sign_consistent:
        return "robust_divergence"
    return "noise"

RESULTS = config.RESULTS_DIR
CG_DIR = RESULTS / "complete_graph"

TAG_BY_CATEGORY = {
    "built": "mass_diag_built",
    "degenerate_rebuilt": "mass_diag_built",
    "fitted": "mass_diag_fitted",
    "heldout_readout_newxnew": "mass_diag_fitted",
}
REPAIRED_REFUSAL_6 = tuple(f"{s}__refusal_language" for s in
                           ("emotional_tone", "formal_register", "hedging", "language",
                            "numerical_density", "question_vs_statement"))
# The stage 2 `**Subject:**`-header strip rebuilt 5 pairs; the 5th
# (harmful_vs_benign__refusal_language) is unbuildable and out of the usable population, so only
# these 4 appear in the table (all v2-BUILT — grid and readout are same-version). Membership is
# asserted against the usable population in build_table() so a future roster change can't silently
# drop one.
SUBJECT_HEADER_REBUILT = ("harmful_vs_benign__sycophantic_agreement", "deception__refusal_language",
                          "refusal_language__topic_medical", "sycophantic_agreement__topic_medical")
LEET_DECEPTION_DECODABILITY_PAIR = "leetspeak__deception"


def _l21(probe_result: dict) -> float:
    return float(probe.aggregate_layers(probe_result)[FROZEN_LAYER].heldout_raw_mean)


def _load_tag(tag: str) -> dict:
    out = {"ft_rows": RV.load_ft_rows(RESULTS / f"ft_grid_{tag}.jsonl"),
           "baseline": json.loads((RESULTS / f"ft_baseline_{tag}.json").read_text())}
    for read, suffix in (("last", ""), ("meanpooled", "_meanpooled"), ("chat", "_chat")):
        out[read] = json.loads((RESULTS / f"probe_{tag}{suffix}.json").read_text())
    return out


def _committed_old_values() -> dict[str, dict]:
    """Committed old-environment (FT delta, probe-L21 delta) for fitted + heldout pairs."""
    probe_all = json.loads((READS_DIR / "probe_all.json").read_text())
    ft_vals = RV.ft_pair_values(RV.load_ft_rows(READS_DIR / "ft_grid_all.jsonl"))
    out = {pk: {"ft": float(ft_vals[pk]), "probe_l21": _l21(res)} for pk, res in probe_all.items()}
    check = json.loads((RESULTS / "newxnew_prediction_check.json").read_text())
    for row in check["rows"]:
        out[row["pair"]] = {"ft": float(row["obs_ft"]), "probe_l21": float(row["obs_probe"])}
    return out


def _flag_sources() -> dict:
    def load(name, default=None):
        p = CG_DIR / name
        return json.loads(p.read_text()) if p.exists() else default

    det = load("format_cue_detector_check.json")
    scan = load("seg2_format_cue_scan.json")
    return {
        "status": load("complete_graph_build_status.json"),
        "format_cue_flagged": {r["pair"] for r in scan["flagged"]},
        "format_cue_held_gap": set(det["held_gap_ge_0p15_pairs"]),
        "format_cue_masking": load("format_cue_masking_control.json", {}),
        "harm_deference": load("harm_deference_control.json", {}),
    }


def build_table() -> list[dict]:
    src = _flag_sources()
    status = src["status"]
    tags = {t: _load_tag(t) for t in ("mass_diag_built", "mass_diag_fitted")}
    old = _committed_old_values()

    rows = []
    for pk, rec in sorted(status["pairs"].items()):
        if rec["category"] == "unbuildable":
            continue
        tag = TAG_BY_CATEGORY[rec["category"]]
        data = tags[tag]
        f1, f2 = split_pair_key(pk)
        assert (f1, f2) == (rec["f1"], rec["f2"]), pk

        pair_ft_rows = [r for r in data["ft_rows"] if r["pair"] == pk]
        assert len(pair_ft_rows) == 3, (pk, len(pair_ft_rows))
        verdict = RV.ft_verdict(pair_ft_rows)
        seed_vals = RV.ft_seed_values(pair_ft_rows)[pk]
        stats = RV.ft_seed_stats(pair_ft_rows)

        if rec["category"] in ("fitted", "heldout_readout_newxnew"):
            grid_version = ("committed_repaired_userturns" if pk in REPAIRED_REFUSAL_6 else "committed")
        elif rec["category"] == "degenerate_rebuilt":
            grid_version = "degenerate_rebuilt_v2"
        else:
            grid_version = "v2_built"

        below_floor_built = pk in set(status["below_floor_built_pairs"])
        masking = src["format_cue_masking"].get(pk)
        deference = src["harm_deference"].get(pk)

        rows.append({
            "pair": pk, "f1": f1, "f2": f2,
            "family": rec["family"], "category": rec["category"],
            "f1_category": feature_category(f1), "f2_category": feature_category(f2),
            "f1_type": features.feature_type(f1), "f2_type": features.feature_type(f2),
            "n_per_cell": rec["n_per_cell"], "min_cell": rec["min_cell"],
            # --- readouts (v2 environment; on-disk orientation)
            "ft_delta": verdict["delta"],
            "ft_seeds": {str(s): v for s, v in sorted(seed_vals.items())},
            "ft_ci95": verdict["ci95"],
            "ft_sign_consistent": verdict["sign_consistent"],
            "verdict": verdict["verdict"],
            "stratum": verdict["stratum"],
            "n_held_texts": verdict["n_texts"],
            "n_collapse_seeds": sum(s["collapse_suspect"] for s in stats),
            "ft_baseline": float(data["baseline"][pk]["heldout"]["raw"]),
            "probe_l21_last": _l21(data["last"][pk]),
            "probe_l21_meanpooled": _l21(data["meanpooled"][pk]),
            "probe_l21_chat": _l21(data["chat"][pk]),
            # --- provenance
            "readout_env": "v2_env",
            "readout_tag": tag,
            "probe_all_layer_files": {read: f"results/probe_{tag}{sfx}.json" for read, sfx in
                                      (("last", ""), ("meanpooled", "_meanpooled"), ("chat", "_chat"))},
            "grid_version": grid_version,
            "delta_committed_oldenv": old.get(pk),
            # --- flags (machine-readable; consumed by phase 1)
            "below_floor_built": below_floor_built,
            "below_floor_fitted": pk in set(status["below_floor_fitted_pairs"]),
            "excluded_from_primary_fit": below_floor_built,  # pre-registered default (co-primary arms)
            "format_cue_flag": pk in src["format_cue_flagged"],
            "format_cue_held_gap": pk in src["format_cue_held_gap"],
            "format_cue_masking_control": None if masking is None else {
                k: masking[k] for k in ("ft_full", "ft_masked", "probe_full", "probe_masked",
                                        "ft_winner_stable", "probe_winner_stable",
                                        "ft_stratum_stable", "probe_stratum_stable")},
            "harm_pair": "harmful_vs_benign" in (f1, f2),
            "harm_deference_control": None if deference is None else {
                k: deference[k] for k in
                ("ft_full", "ft_masked", "ft_held_only", "probe_full", "probe_masked",
                 "probe_held_only", "ft_winner_stable", "probe_winner_stable",
                 "ft_held_only_winner_stable", "probe_held_only_winner_stable")
                if k in deference},
            "leet_pair": "leetspeak" in (f1, f2),  # feature-visibility caution (whole class)
            "leet_deception_decodability_flag": pk == LEET_DECEPTION_DECODABILITY_PAIR,
            "fitted_refusal_userturn_repaired": pk in REPAIRED_REFUSAL_6,
            "subject_header_rebuilt": pk in SUBJECT_HEADER_REBUILT,
            # FT/probe winner-split class (None = not a split); robust_divergence is the
            # safety-relevant "monitor keys on a different feature than the policy" set
            "ft_probe_split_class": classify_ft_probe_split(
                f1, f2, verdict["delta"], _l21(data["last"][pk]), _l21(data["meanpooled"][pk]),
                verdict["sign_consistent"]),
        })
    assert len(rows) == status["n_usable_population"] == 663, len(rows)
    assert len({r["pair"] for r in rows}) == 663
    # provenance-flag membership guards (a roster change can't silently drop a flagged pair)
    pop = {r["pair"] for r in rows}
    assert set(REPAIRED_REFUSAL_6) <= pop, set(REPAIRED_REFUSAL_6) - pop
    assert set(SUBJECT_HEADER_REBUILT) <= pop, set(SUBJECT_HEADER_REBUILT) - pop
    by_pair = {r["pair"]: r for r in rows}
    assert all(by_pair[p]["grid_version"] in ("committed", "committed_repaired_userturns")
               for p in REPAIRED_REFUSAL_6)
    return rows


def load_table(path=None) -> list[dict]:
    path = path or (CG_DIR / "merged_readouts_v2.jsonl")
    return [json.loads(l) for l in open(path) if l.strip()]
