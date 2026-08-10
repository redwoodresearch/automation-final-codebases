"""FULL-37 cross-model ranking transfer analysis (new deliverable, Aug 2026).

Mirrors analyze_cross_model.py / analyze_ft_cross_model.py, but over the 100-pair `xmodel` set that the
read-position-across-models check used (spans all 37 features), refitting per-feature strengths from
those 100 pairs for every model and comparing rankings to the Qwen3-1.7B reference with the SAME
pair-refit bootstrap machinery (cross_model.s_spearman_vs_ref).

  PROBE reference  = the Qwen3-1.7B probe sweep over the same 100 pairs (same read), refit over 37 feats.
  FINE-TUNE ref    = the committed 1.7B fine-tune (merged_readouts_v2.jsonl ft_delta), the canonical
                     37-feature fine-tune ranking the read-position check also uses, refit over 37 feats.

Inputs:
  results/cross_model/probe_<key>_x37.json         (last-token read; matches the blog headline 0.93-0.97)
  results/cross_model/probe_<key>_mean_x37.json     (mean-pooled read)
  results/cross_model_ft_37/ft_<key>_lora.jsonl      (LoRA fine-tune sweep, 4 models)
  ../inoc_data/merged_readouts_v2.jsonl              (1.7B fine-tune reference)
Output: results/cross_model_37/transfer_37.json + printed summary.

  PYTHONPATH=lib .venv/bin/python experiments/cross_model/analyze_x37.py
"""
from __future__ import annotations

import os as _os
_XMODEL_PAIRS = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'xmodel_pairs.json')
import json
from pathlib import Path

import numpy as np

import config
import cross_model as cm
import cross_model_ft as cmft
import rank_stats as rs

TOP = Path(config.RESULTS_DIR).parent
XMODEL = json.load(open(_XMODEL_PAIRS))
PROBE_DIR = config.results_dir("cross_model")
FT_DIR = TOP / "results" / "cross_model_ft_37"
MERGED = TOP.parent / "inoc_data" / "merged_readouts_v2.jsonl"
OUT_DIR = TOP / "results" / "cross_model_37"

REF = "qwen3_1p7b"
PROBE_MODELS = ["qwen3_8b", "qwen3_32b", "llama31_8b", "olmo2_7b", "qwen3_1p7b_base", "qwen3_8b_base"]
FT_MODELS = ["qwen3_8b", "llama31_8b", "olmo2_7b", "qwen3_32b"]


def load_probe(key, read):
    suffix = "_x37" if read == "last" else f"_{read}_x37"
    p = PROBE_DIR / f"probe_{key}{suffix}.json"
    return json.loads(p.read_text()) if p.exists() else None


def probe_ladder(probe: dict, pairs) -> dict[str, float]:
    """Per-pair Δ_pr at the FINAL hidden state (the committed PRIMARY read)."""
    return {p: cm.primary_delta(probe[p]) for p in pairs if p in probe}


def winner_agreement(model_lad, ref_lad, pairs):
    rows = [(p, model_lad[p], ref_lad[p]) for p in pairs if p in model_lad and p in ref_lad]
    mv = np.array([r[1] for r in rows]); rv = np.array([r[2] for r in rows])
    agree = (np.sign(mv) == np.sign(rv))
    # "should-never-flip" proxy: well-separated reference pairs (top tercile of |Δ_ref|).
    thr = np.quantile(np.abs(rv), 2 / 3)
    big = np.abs(rv) >= thr
    return {"n": len(rows), "overall": round(float(agree.mean()), 3),
            "n_large_gap": int(big.sum()),
            "large_gap_agreement": round(float(agree[big].mean()), 3) if big.any() else None,
            "flips": [rows[i][0] for i in range(len(rows)) if not agree[i]]}


def per_pair_spearman(model_lad, ref_lad, pairs):
    shp = [p for p in pairs if p in model_lad and p in ref_lad]
    return round(rs.spearman([model_lad[p] for p in shp], [ref_lad[p] for p in shp]), 3), len(shp)


def probe_section(read):
    ref_probe = load_probe(REF, read)
    if ref_probe is None:
        return None
    pairs = [p for p in XMODEL if p in ref_probe]
    ref_lad = probe_ladder(ref_probe, pairs)
    sec = {"read": read, "reference": REF, "n_pairs": len(pairs), "per_model": {}}
    sec["reference_additive"] = cm.additive_diagnostics(ref_lad, pairs)
    for k in PROBE_MODELS:
        pr = load_probe(k, read)
        if pr is None:
            continue
        lad = probe_ladder(pr, pairs)
        sp = cm.s_spearman_vs_ref(lad, ref_lad, pairs)
        pp, npp = per_pair_spearman(lad, ref_lad, pairs)
        s = cm.spec(k)
        sec["per_model"][k] = {
            "label": s.label, "family": s.family, "params_b": s.params_b, "post_trained": s.post_trained,
            "s_pr_spearman": sp["spearman"], "kendall": sp["kendall"], "ci": [sp["ci_lo"], sp["ci_hi"]],
            "n_features": sp["n_features"], "per_pair_delta_spearman": pp,
            "additive_law": cm.additive_diagnostics(lad, pairs),
            "winner_agreement": winner_agreement(lad, ref_lad, pairs)}
    return sec


def load_ft_ladder(key):
    p = FT_DIR / f"ft_{key}_lora.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows, cmft.assemble_ft_ladder(rows)


def ft_reference_ladder(pairs):
    merged = {json.loads(l)["pair"]: json.loads(l) for l in MERGED.read_text().splitlines() if l.strip()}
    return {p: float(merged[p]["ft_delta"]) for p in pairs if p in merged and merged[p].get("ft_delta") is not None}


def ft_section():
    ref_lad = ft_reference_ladder(XMODEL)
    sec = {"reference": "qwen3_1p7b (committed fine-tune, merged_readouts_v2 ft_delta)",
           "n_ref_pairs": len(ref_lad), "per_model": {}}
    ladders = {}
    for k in FT_MODELS:
        got = load_ft_ladder(k)
        if got is None:
            continue
        rows, asm = got
        lad = asm["ladder"]
        ladders[k] = lad
        pairs = [p for p in XMODEL if p in lad and p in ref_lad]         # like-for-like coverage
        sp = cm.s_spearman_vs_ref(lad, ref_lad, pairs)
        pp, npp = per_pair_spearman(lad, ref_lad, pairs)
        # probe<->fine-tune agreement: last-token probe ladder vs this fine-tune ladder (same model)
        pr = load_probe(k, "last")
        pft = None
        if pr is not None:
            plad = probe_ladder(pr, pairs)
            pft = cm.s_spearman_vs_ref(lad, plad, [p for p in pairs if p in plad])
        s = cm.spec(k)
        n_seeds = sorted({r["seed"] for r in rows})
        sec["per_model"][k] = {
            "label": s.label, "family": s.family, "params_b": s.params_b, "seeds": n_seeds,
            "n_pairs_used": len(pairs), "n_dropped_pairs": len(asm["dropped_pairs"]),
            "dropped_pairs": asm["dropped_pairs"],
            "s_ft_spearman": sp["spearman"], "kendall": sp["kendall"], "ci": [sp["ci_lo"], sp["ci_hi"]],
            "n_features": sp["n_features"], "per_pair_delta_spearman": pp,
            "additive_law": cm.additive_diagnostics(lad, pairs),
            "winner_agreement": winner_agreement(lad, ref_lad, pairs),
            "probe_ft_agreement": (pft["spearman"] if pft else None)}
    # pairwise cross-model FT agreement (is the ranking shared, not just vs the reference?)
    keys = sorted(ladders)
    if len(keys) >= 2:
        common = [p for p in XMODEL if all(p in ladders[k] for k in keys)]
        mat = {f"{a}__vs__{b}": cm.s_spearman_vs_ref(ladders[a], ladders[b], common)["spearman"]
               for i, a in enumerate(keys) for b in keys[i + 1:]}
        off = list(mat.values())
        sec["pairwise_ft_agreement"] = {"n_common_pairs": len(common), "matrix": mat,
                                        "mean": round(float(np.mean(off)), 3),
                                        "min": round(float(np.min(off)), 3)}
    return sec


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {"pair_set": "reads_expansion xmodel (read-position-across-models check)",
           "n_pairs": len(XMODEL),
           "probe_last": probe_section("last"),
           "probe_mean": probe_section("mean"),
           "fine_tune": ft_section()}
    (OUT_DIR / "transfer_37.json").write_text(json.dumps(out, indent=2))
    _summary(out)
    print(f"\nwrote {OUT_DIR/'transfer_37.json'}")
    return out


def _summary(out):
    for read in ("probe_last", "probe_mean"):
        sec = out[read]
        if not sec:
            print(f"\n[{read}] no data yet"); continue
        print(f"\n=== PROBE transfer ({read}, n_pairs={sec['n_pairs']}) vs Qwen3-1.7B, refit over 37 feats ===")
        print(f"{'model':16s} {'params':>6} {'s_pr rho':>9} {'CI':>16} {'nfeat':>5} {'perpairRho':>10} {'winner%':>8} {'bigGap%':>8}")
        for k, m in sec["per_model"].items():
            print(f"{m['label']:16s} {m['params_b']:6.1f} {m['s_pr_spearman']:9.3f} "
                  f"[{m['ci'][0]:+.2f},{m['ci'][1]:+.2f}] {m['n_features']:5d} "
                  f"{m['per_pair_delta_spearman']:10.3f} {m['winner_agreement']['overall']*100:7.0f}% "
                  f"{(m['winner_agreement']['large_gap_agreement'] or 0)*100:7.0f}%")
    sec = out["fine_tune"]
    print(f"\n=== FINE-TUNE transfer (n_ref_pairs={sec['n_ref_pairs']}) vs Qwen3-1.7B fine-tune, refit over 37 feats ===")
    print(f"{'model':16s} {'params':>6} {'seeds':>7} {'s_ft rho':>9} {'CI':>16} {'nfeat':>5} {'winner%':>8} {'probe<->ft':>11} {'drop':>5}")
    for k, m in sec["per_model"].items():
        print(f"{m['label']:16s} {m['params_b']:6.1f} {len(m['seeds']):7d} {m['s_ft_spearman']:9.3f} "
              f"[{m['ci'][0]:+.2f},{m['ci'][1]:+.2f}] {m['n_features']:5d} "
              f"{m['winner_agreement']['overall']*100:7.0f}% {(m['probe_ft_agreement'] or 0):11.3f} "
              f"{m['n_dropped_pairs']:5d}")
    if "pairwise_ft_agreement" in sec:
        pa = sec["pairwise_ft_agreement"]
        print(f"\npairwise cross-model FT agreement: mean {pa['mean']} (min {pa['min']}, n_common={pa['n_common_pairs']})")


if __name__ == "__main__":
    main()
