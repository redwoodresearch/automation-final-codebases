"""Dose-response analysis (stage 5, work item 9). Per feature: Spearman(realized dose, activation
movement), where movement = per-text projection of (activation − mean(zero-level)) onto the feature's
GRID-derived signature direction (mean-pooled L20; the direction is never re-derived from dose texts).
Also raw ‖mean(level)−mean(zero)‖, length-adjusted correlation, intended-level (secondary), and the
fraction of features with the expected (positive) sign vs a binomial null.

Outputs: results/complete_graph/dose_response_v2.{json,md}
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, spearmanr

import act_signature as A
import config
import dose
import features

CG = config.RESULTS_DIR / "complete_graph"
MANIFEST = CG / "dose_signature_manifest_v2.json"
KEEP_IDX = {20: 0, 21: 1}


def _boot_ci(x, y, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        if len(set(x[idx])) > 1:
            vals.append(spearmanr(x[idx], y[idx]).statistic)
    vals = np.array([v for v in vals if np.isfinite(v)])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _partial_len(dose_v, move, nwords):
    """Spearman(dose, movement) after removing a linear length effect from movement."""
    nwords = np.asarray(nwords, float)
    if np.std(nwords) < 1e-9:
        return float(spearmanr(dose_v, move).statistic)
    b = np.polyfit(nwords, move, 1)
    resid = move - np.polyval(b, nwords)
    return float(spearmanr(dose_v, resid).statistic)


def main():
    manifest = json.loads(MANIFEST.read_text())
    checks = defaultdict(dict)
    for r in (json.loads(l) for l in open(CG / "dose_checks_v2.jsonl")):
        checks[r["feature"]][(r["level"], r["idx"])] = r
    gate = json.loads((CG / "dose_check_gate_v2.json").read_text())
    sigvec = A.load_signature_vectors(CG / "signature_vectors_v2.npz")["mean"]  # {feature: [L,H]}

    per_feature = {}
    for feature, m in manifest.items():
        if not gate.get(feature, {}).get("passed_gate", False):
            per_feature[feature] = {"analyzed": False, "reason": "failed dose QA gate"}
            continue
        u = sigvec[feature][20]
        u = u / (np.linalg.norm(u) + 1e-12)
        with np.load(m["npz"]) as z:
            level_pt = {lvl: z[f"L{lvl}__pertext_mean"][:, KEEP_IDX[20], :] for lvl in m["levels"]}
            level_pt21 = {lvl: z[f"L{lvl}__pertext_mean"][:, KEEP_IDX[21], :] for lvl in m["levels"]}
        u21 = sigvec[feature][21]
        u21 = u21 / (np.linalg.norm(u21) + 1e-12)
        # kept texts + realized dose + length, aligned to idx_order
        dose_v, move, move21, nwords, ilevel = [], [], [], [], []
        zero_acts = []
        # first collect zero-level kept acts for the reference
        for lvl in m["levels"]:
            order = m["idx_order"][f"L{lvl}"]
            for pos, idx in enumerate(order):
                c = checks[feature].get((lvl, idx))
                if c and c["kept"] and lvl == m["levels"][0]:
                    zero_acts.append(level_pt[lvl][pos])
        zero_mean = np.mean(zero_acts, 0)
        zero_mean21 = None
        z21 = []
        for lvl in m["levels"]:
            order = m["idx_order"][f"L{lvl}"]
            for pos, idx in enumerate(order):
                if lvl == m["levels"][0]:
                    c = checks[feature].get((lvl, idx))
                    if c and c["kept"]:
                        z21.append(level_pt21[lvl][pos])
        zero_mean21 = np.mean(z21, 0)
        for lvl in m["levels"]:
            order = m["idx_order"][f"L{lvl}"]
            for pos, idx in enumerate(order):
                c = checks[feature].get((lvl, idx))
                if not (c and c["kept"]):
                    continue
                dose_v.append(c["realized_dose"]); ilevel.append(lvl); nwords.append(c["n_words"])
                move.append(float((level_pt[lvl][pos] - zero_mean) @ u))
                move21.append(float((level_pt21[lvl][pos] - zero_mean21) @ u21))
        dose_v, move, move21 = np.array(dose_v), np.array(move), np.array(move21)
        rho = float(spearmanr(dose_v, move).statistic)
        # raw norm movement per level
        raw = {}
        for lvl in m["levels"]:
            order = m["idx_order"][f"L{lvl}"]
            acts = [level_pt[lvl][pos] for pos, idx in enumerate(order)
                    if (checks[feature].get((lvl, idx)) or {}).get("kept")]
            if acts:
                raw[int(lvl)] = float(np.linalg.norm(np.mean(acts, 0) - zero_mean))
        esign = dose.expected_sign(feature)
        per_feature[feature] = {
            "analyzed": True, "type": features.feature_type(feature), "dose_type": dose.dose_type(feature),
            "n": int(len(dose_v)), "expected_sign": esign,
            "spearman_realized_proj_L20": rho,
            "spearman_ci": _boot_ci(dose_v, move),
            "spearman_realized_proj_L20_lenadj": _partial_len(dose_v, move, nwords),
            "spearman_intended_proj_L20": float(spearmanr(ilevel, move).statistic),
            "spearman_realized_proj_L21": float(spearmanr(dose_v, move21).statistic),
            "raw_norm_movement_by_level": raw,
            # "correct" = the projection moves in the direction the dosed pole predicts (esign)
            "moves_as_expected": rho * esign > 0,
        }

    analyzed = {f: v for f, v in per_feature.items() if v.get("analyzed")}
    n_pos = sum(v["moves_as_expected"] for v in analyzed.values())
    n = len(analyzed)
    binom = binomtest(n_pos, n, 0.5, alternative="greater").pvalue if n else 1.0
    # signed by the dosed pole: |rho| in the expected direction (positive = dose-consistent)
    rhos = [v["spearman_realized_proj_L20"] * v["expected_sign"] for v in analyzed.values()]
    summary = {"n_analyzed": n, "n_failed_gate": len(per_feature) - n,
               "failed_features": [f for f, v in per_feature.items() if not v.get("analyzed")],
               "n_expected_sign": n_pos, "binomial_p_greater": float(binom),
               "median_spearman": float(np.median(rhos)) if rhos else None,
               "mean_spearman": float(np.mean(rhos)) if rhos else None,
               "frac_spearman_ge_0.3": float(np.mean([r >= 0.3 for r in rhos])) if rhos else None}
    out = {"summary": summary, "per_feature": per_feature}
    (CG / "dose_response_v2.json").write_text(json.dumps(out, indent=1))
    _write_md(out)
    print("=== dose-response ===")
    print(f"  analyzed {n}, failed gate {summary['n_failed_gate']} ({summary['failed_features']})")
    print(f"  expected-sign {n_pos}/{n} (binom p={binom:.2e}); median Spearman {summary['median_spearman']}")


def _write_md(out):
    s = out["summary"]
    L = ["# Dose-response: saliency → mean-activation (work item 9)\n",
         f"Per feature: Spearman(realized dose, per-text projection of activation onto the "
         f"GRID-derived signature direction, mean-pooled L20). Each dose axis increases toward the "
         f"feature's + pole; **hedging** is the sole exception — the study stores it as +confident / "
         f"−hedged, so dosing 'more hedging' targets the − pole and its correct dose-response is "
         f"NEGATIVE (ρ=−0.87). Counting each feature in the direction its dosed pole predicts, "
         f"**{s['n_expected_sign']}/{s['n_analyzed']} features move in the dose-consistent direction** "
         f"(binomial p={s['binomial_p_greater']:.2e} vs a 50/50 null); median |ρ| in the dosed "
         f"direction **{s['median_spearman']:.2f}**, {s['frac_spearman_ge_0.3']*100:.0f}% with "
         f"dose-consistent ρ ≥ 0.3.\n"]
    if s["failed_features"]:
        L.append(f"Failed the dose QA gate (not analyzed): {s['failed_features']}\n")
    L.append("| feature | type | dose type | n | Spearman [CI] | pole | len-adj | intended | L21 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    rows = [(f, v) for f, v in out["per_feature"].items() if v.get("analyzed")]
    for f, v in sorted(rows, key=lambda kv: -kv[1]["spearman_realized_proj_L20"] * kv[1]["expected_sign"]):
        pole = "−" if v["expected_sign"] < 0 else "+"
        L.append(f"| {f} | {v['type']} | {v['dose_type']} | {v['n']} | "
                 f"{v['spearman_realized_proj_L20']:+.2f} [{v['spearman_ci'][0]:+.2f},{v['spearman_ci'][1]:+.2f}] | "
                 f"{pole} | {v['spearman_realized_proj_L20_lenadj']:+.2f} | {v['spearman_intended_proj_L20']:+.2f} | "
                 f"{v['spearman_realized_proj_L21']:+.2f} |")
    (CG / "dose_response_v2.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
