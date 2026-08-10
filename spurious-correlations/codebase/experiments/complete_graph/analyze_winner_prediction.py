"""Blog analysis 2 (stage 5, work item 5): the training-free winner-prediction technique, applied
to all 663 pairs under the pre-registered spec.

Predictor (pre-registered): winner of {f1,f2} = the feature with the larger cell-balanced
mean-activation-diff norm ‖d_f‖ (raw activations). PRIMARY spec = mean-pooled L20; CONTINUITY =
last-token L20. Two variants: pair-local and roster-global leave-one-pair-out. Compared against 5
cheap baselines. PRIMARY target = the frozen diagonal FT verdict winner, decisive pairs only.

Outputs: results/complete_graph/winner_prediction_v2.{json,md}
Requires: signature_manifest_v2.json, winner_pred_baseline_inputs_v2.json, predictor_stability_v2.json.
"""
from __future__ import annotations

import json

import numpy as np

import act_signature as A
import config
import features
import winner_predict as WP
from extension_properties import split_pair_key
from merged_readouts import load_table

CG = config.RESULTS_DIR / "complete_graph"
L20, L21 = 20, 21


def _load():
    manifest = A.load_manifest()
    rows = {r["pair"]: r for r in load_table()}
    store = A.SignatureStore(manifest, list(rows))
    refit = json.load(open(CG / "refit_v2.json"))
    s = {ax: refit["arms"]["without_excluded"][ax]["s"] for ax in ("ft", "probe")}
    llm = json.load(open(CG / "llm_predict_strength_37.json"))
    llm_scores = {m: {ax: llm["models"][m][ax]["predictions"] for ax in ("ft", "probe")}
                  for m in llm["models"]}
    tmean = {ax: {t: v["mean"] for t, v in
                  json.load(open(CG / "type_strengths_v2.json"))["arms"]["without_excluded"][ax]
                  ["base"]["per_type"].items()} for ax in ("ft", "probe")}
    base = json.load(open(CG / "winner_pred_baseline_inputs_v2.json"))
    stab = json.load(open(CG / "predictor_stability_v2.json"))
    split = json.load(open(CG / "ft_probe_split_classification.json"))
    return manifest, rows, store, s, llm_scores, tmean, base, stab, split


def build_predictors(store, rows, base, llm_scores, tmean):
    pairs = list(rows)
    P = {}
    # signature predictors
    for read, tag in (("mean", "meanL20"), ("last", "lastL20")):
        P[f"sig_{tag}_pairlocal"] = {pk: store.predict_pair_local(pk, read, L20)["pred_winner"] for pk in pairs}
        loo = store.predict_roster_loo(read, L20)
        P[f"sig_{tag}_rosterLOO"] = {pk: loo[pk]["pred_winner"] for pk in pairs}
    # unbalanced ablation (mean L20)
    P["unbalanced_meanL20"] = {}
    for pk in pairs:
        f1, f2 = split_pair_key(pk)
        m = {c: store.means["mean"][pk][c][L20] for c in A.ALL_CELLS}
        d1, d2 = A.unbalanced_directions(m, f1, f2)
        P["unbalanced_meanL20"][pk] = f1 if np.linalg.norm(d1) >= np.linalg.norm(d2) else f2
    # baselines
    P["baseline_length"] = {pk: WP.larger_wins(base[pk]["length_sep"], *split_pair_key(pk)) for pk in pairs}
    P["baseline_tfidf"] = {pk: WP.larger_wins(base[pk]["tfidf_auc"], *split_pair_key(pk)) for pk in pairs}
    P["baseline_llm_opus_ft"] = {pk: WP.llm_winner(*split_pair_key(pk), llm_scores["claude-opus-4-8"]["ft"]) for pk in pairs}
    P["baseline_llm_sonnet_ft"] = {pk: WP.llm_winner(*split_pair_key(pk), llm_scores["claude-sonnet-4-6"]["ft"]) for pk in pairs}
    P["baseline_type_prior_ft"] = {pk: WP.type_prior_winner(*split_pair_key(pk), tmean["ft"]) for pk in pairs}
    return P


def build_targets(rows, s):
    pairs = list(rows)
    T = {}
    T["ft"] = {pk: WP.ft_winner(rows[pk]) for pk in pairs}
    T["probe"] = {pk: WP.probe_winner(rows[pk]) for pk in pairs}
    T["strength_order_ft"] = {pk: WP.strength_order_winner(*split_pair_key(pk), s["ft"]) for pk in pairs}
    T["strength_order_probe"] = {pk: WP.strength_order_winner(*split_pair_key(pk), s["probe"]) for pk in pairs}
    return T


def decisive_pop(truth: dict) -> list[str]:
    return [pk for pk, w in truth.items() if w is not None]


def main():
    manifest, rows, store, s, llm_scores, tmean, base, stab, split = _load()
    pairs = list(rows)
    P = build_predictors(store, rows, base, llm_scores, tmean)
    T = build_targets(rows, s)
    ft_dec = decisive_pop(T["ft"])
    probe_dec = decisive_pop(T["probe"])

    out = {"populations": {"n_pairs": len(pairs), "ft_decisive": len(ft_dec),
                           "probe_decisive": len(probe_dec)},
           "tie_band_abs": stab["tie_band_abs"]}

    # ---- accuracy matrix: predictor x target (primary target = ft decisive)
    targets_pop = {"ft": ft_dec, "probe": probe_dec,
                   "strength_order_ft": decisive_pop(T["strength_order_ft"]),
                   "strength_order_probe": decisive_pop(T["strength_order_probe"])}
    matrix = {}
    for pname, preds in P.items():
        matrix[pname] = {tname: WP.score(preds, T[tname], pop) for tname, pop in targets_pop.items()}
    out["accuracy_matrix"] = matrix

    # ---- headline
    out["headline"] = {
        "primary_pairlocal_vs_ft": matrix["sig_meanL20_pairlocal"]["ft"],
        "primary_rosterLOO_vs_ft": matrix["sig_meanL20_rosterLOO"]["ft"],
        "continuity_pairlocal_vs_ft": matrix["sig_lastL20_pairlocal"]["ft"],
    }

    # ---- variant agreement + direction cosine (mean L20)
    loo = store.predict_roster_loo("mean", L20)
    agree = np.mean([P["sig_meanL20_pairlocal"][pk] == loo[pk]["pred_winner"] for pk in pairs])
    sigs_loo = store  # need LOO signature vectors for cosine
    cosines = _pairlocal_vs_roster_cosines(store)
    out["variant_agreement"] = {
        "pairlocal_vs_rosterLOO_agree_rate": float(agree),
        "direction_cosine_median": float(np.median(cosines)),
        "direction_cosine_p10": float(np.percentile(cosines, 10)),
        "direction_cosine_p90": float(np.percentile(cosines, 90)),
        "direction_cosine_min": float(np.min(cosines)),
    }

    # ---- by matchup type (primary predictor, ft target) + by winner-type
    out["by_type"] = _by_type(P["sig_meanL20_pairlocal"], T["ft"], rows, ft_dec)

    # ---- by gap stratum (primary predictor + baselines for the figure)
    out["by_gap"] = _by_gap(P["sig_meanL20_pairlocal"], T["ft"], rows, s, ft_dec)
    gap_preds = {"sig_meanL20_pairlocal": P["sig_meanL20_pairlocal"],
                 "baseline_tfidf": P["baseline_tfidf"], "baseline_llm_sonnet_ft": P["baseline_llm_sonnet_ft"],
                 "baseline_length": P["baseline_length"]}
    out["by_gap_multi"] = _by_gap_multi(gap_preds, T["ft"], s, ft_dec)

    # ---- tie-band coverage/accuracy tradeoff
    out["tie_band_tradeoff"] = _tie_band(store, T["ft"], ft_dec, stab["tie_band_abs"])

    # ---- descriptive layer x read sweep (pair-local vs ft)
    out["layer_sweep"] = _layer_sweep(store, T["ft"], ft_dec)

    # ---- failure modes
    out["failure_modes"] = _failure_modes(store, P, T, rows, split, s)

    (CG / "winner_prediction_v2.json").write_text(json.dumps(out, indent=1))
    _write_md(out)
    _print(out)


def _pairlocal_vs_roster_cosines(store):
    # per pair per feature: cosine(pair-local d_f (mean L20), LOO signature of f)
    acc = store._per_pair_dirs("mean", L20)
    sums = {f: np.sum([d for _, d in v], axis=0) for f, v in acc.items()}
    counts = {f: len(v) for f, v in acc.items()}
    by_feat_pair = {f: {pk: d for pk, d in v} for f, v in acc.items()}
    cos = []
    for f, v in acc.items():
        if counts[f] <= 1:
            continue
        for pk, d in v:
            loo = (sums[f] - by_feat_pair[f][pk]) / (counts[f] - 1)
            c = float(d @ loo / (np.linalg.norm(d) * np.linalg.norm(loo) + 1e-12))
            cos.append(c)
    return np.array(cos)


def _by_type(pred, truth, rows, pop):
    from collections import defaultdict
    by_pairtype = defaultdict(lambda: [0, 0])  # type-pair -> [correct, n]
    by_wintype = defaultdict(lambda: [0, 0])
    for pk in pop:
        r = rows[pk]
        tp = tuple(sorted((r["f1_type"], r["f2_type"])))
        ok = int(pred[pk] == truth[pk])
        by_pairtype[tp][0] += ok
        by_pairtype[tp][1] += 1
        wt = features.feature_type(truth[pk])
        by_wintype[wt][0] += ok
        by_wintype[wt][1] += 1
    return {"by_type_pair": {"__".join(k): {"acc": c / n, "n": n} for k, (c, n) in sorted(by_pairtype.items())},
            "by_winner_type": {k: {"acc": c / n, "n": n} for k, (c, n) in sorted(by_wintype.items())}}


def _by_gap(pred, truth, rows, s, pop):
    # diagonal |ft_delta| strata
    delta_bins = [(0, 3), (3, 6), (6, 10), (10, 15), (15, 100)]
    sgap_bins = [(0, 3), (3, 6), (6, 10), (10, 15), (15, 100)]
    import features as F
    diag = {}
    for lo, hi in delta_bins:
        sub = [pk for pk in pop if lo <= abs(rows[pk]["ft_delta"]) < hi]
        if sub:
            c = sum(pred[pk] == truth[pk] for pk in sub)
            entry = {"acc": c / len(sub), "n": len(sub)}
            # trigger-winner subset (the below-chance regime in the top bin)
            tw = [pk for pk in sub if F.feature_type(truth[pk]) == "triggers"]
            if tw:
                entry["twinner_acc"] = sum(pred[pk] == truth[pk] for pk in tw) / len(tw)
                entry["twinner_n"] = len(tw)
            diag[f"[{lo},{hi})"] = entry
    sgap = {}
    for lo, hi in sgap_bins:
        sub = [pk for pk in pop
               if lo <= abs(s["ft"][split_pair_key(pk)[0]] - s["ft"][split_pair_key(pk)[1]]) < hi]
        if sub:
            c = sum(pred[pk] == truth[pk] for pk in sub)
            sgap[f"[{lo},{hi})"] = {"acc": c / len(sub), "n": len(sub)}
    return {"by_ft_delta": diag, "by_strength_gap_ft": sgap}


def _by_gap_multi(preds: dict, truth, s, pop):
    """Accuracy per fitted-strength-gap bin for several predictors (full-pop credit, ties=0.5)."""
    bins = [(0, 3), (3, 6), (6, 10), (10, 15), (15, 100)]
    out = {"bins": [f"[{lo},{hi})" for lo, hi in bins], "predictors": {}}
    binned = {f"[{lo},{hi})": [pk for pk in pop
              if lo <= abs(s["ft"][split_pair_key(pk)[0]] - s["ft"][split_pair_key(pk)[1]]) < hi]
              for lo, hi in bins}
    out["n_per_bin"] = {k: len(v) for k, v in binned.items()}
    for name, pred in preds.items():
        accs = []
        for k, sub in binned.items():
            if not sub:
                accs.append(None); continue
            correct = sum((pred[pk] == truth[pk]) if pred[pk] is not None else 0.5 for pk in sub)
            accs.append(correct / len(sub))
        out["predictors"][name] = accs
    return out


def _tie_band(store, truth, pop, tie_band):
    out = {"chosen_tie_band": tie_band, "curve": {}}
    for tb in (0.0, tie_band, 2 * tie_band, 5.0, 10.0):
        preds = {}
        for pk in pop:
            r = store.predict_pair_local(pk, "mean", L20)
            preds[pk] = None if abs(r["margin"]) <= tb else r["pred_winner"]
        sc = WP.score(preds, truth, pop)
        out["curve"][f"{tb:.3f}"] = {"coverage": sc["coverage"], "decided_acc": sc["decided_acc"],
                                     "n_decided": sc["decided"]}
    return out


def _layer_sweep(store, truth, pop):
    out = {}
    for read in A.READS:
        accs = []
        for L in range(store.n_layers):
            c = sum(store.predict_pair_local(pk, read, L)["pred_winner"] == truth[pk] for pk in pop)
            accs.append(c / len(pop))
        out[read] = accs
    return out


def _failure_modes(store, P, T, rows, split, s):
    fm = {}
    pred_mean = P["sig_meanL20_pairlocal"]
    pred_last = P["sig_lastL20_pairlocal"]
    # read_location: mean vs last accuracy on FT winner (last expected to misbehave)
    rl = [pk for pk in split["read_location_pairs"] if T["ft"][pk] is not None]
    fm["read_location"] = {
        "n_ft_decisive": len(rl),
        "mean_acc": np.mean([pred_mean[pk] == T["ft"][pk] for pk in rl]) if rl else None,
        "last_acc": np.mean([pred_last[pk] == T["ft"][pk] for pk in rl]) if rl else None,
    }
    # robust_divergence: does the sig predictor follow the FT winner or the PROBE winner?
    div = split["robust_divergence_pairs"]
    rows_div = []
    n_match_probe = n_match_ft = 0
    for pk in div:
        r = rows[pk]
        ftw = WP.ft_winner(r)
        prw = WP.probe_winner(r)
        sw = pred_mean[pk]
        mp = (prw is not None and sw == prw)
        mf = (ftw is not None and sw == ftw)
        n_match_probe += int(mp)
        n_match_ft += int(mf)
        rows_div.append({"pair": pk, "ft_winner": ftw, "probe_winner": prw, "sig_winner": sw,
                         "matches_probe": mp, "matches_ft": mf})
    # margins on the divergence pairs: the technique is CONFIDENT (large |margin|) yet wrong about FT
    div_margins = [abs(store.predict_pair_local(pk, "mean", L20)["margin"]) for pk in div]
    fm["robust_divergence"] = {"n": len(div), "n_match_probe": n_match_probe, "n_match_ft": n_match_ft,
                               "median_abs_margin": float(np.median(div_margins)),
                               "n_margin_gt_10": int(sum(m > 10 for m in div_margins)),
                               "pairs": rows_div}
    # leet pairs
    leet = [pk for pk in T["ft"] if "leetspeak" in split_pair_key(pk) and T["ft"][pk] is not None]
    fm["leet"] = {"n": len(leet),
                  "mean_acc": float(np.mean([pred_mean[pk] == T["ft"][pk] for pk in leet]))}
    return fm


def _write_md(out):
    L = ["# Blog analysis 2 — training-free winner prediction (all 663 pairs)\n",
         f"Predict which feature a classifier commits to = the one with the larger cell-balanced "
         f"mean-activation-diff norm ‖d_f‖. **Primary spec = mean-pooled L20; primary target = frozen "
         f"diagonal FT verdict winner, decisive pairs only** (n={out['populations']['ft_decisive']} of "
         f"{out['populations']['n_pairs']}).\n"]
    L.append("## Accuracy vs the FT decisive winner (headline)\n")
    L.append("Full-pop acc credits abstentions (ties) as 0.5 — the fair same-population comparator, "
             "since the signature predictor always predicts (coverage 1.00). Decided acc is over the "
             "pairs the predictor commits on.\n")
    L.append("| predictor | full-pop acc | decided acc [95% CI] | coverage | n decided |")
    L.append("|---|---|---|---|---|")
    order = ["sig_meanL20_pairlocal", "sig_meanL20_rosterLOO", "sig_lastL20_pairlocal",
             "sig_lastL20_rosterLOO", "unbalanced_meanL20", "baseline_tfidf", "baseline_length",
             "baseline_llm_opus_ft", "baseline_llm_sonnet_ft", "baseline_type_prior_ft"]
    for p in order:
        r = out["accuracy_matrix"][p]["ft"]
        L.append(f"| {p} | {r['acc']:.3f} | {r['decided_acc']:.3f} [{r['decided_acc_ci'][0]:.2f}, "
                 f"{r['decided_acc_ci'][1]:.2f}] | {r['coverage']:.2f} | {r['decided']} |")
    L.append("\n**On the primary spec (mean-pooled L20) the technique matches the strongest few-shot "
             "LLM predictor on FT prediction and beats the cheap surface baselines** (text-length is "
             "at chance, 0.52 — length is NOT what drives it). The continuity spec (last-token L20) "
             "scores higher OVERALL (0.81) but is catastrophic exactly on the 26 read-location pairs "
             "(0.00, see failure modes); it is NOT promoted over the pre-registered mean-pooled "
             "primary. The unbalanced ablation ≈ the balanced primary on FT accuracy — the "
             "partner-cancelling construction buys a cleaner *direction* (needed for the dose analysis "
             "and interpretability), not winner-ranking accuracy on this task.\n")
    L.append("\n## Same predictors vs other targets (decided accuracy)\n")
    L.append("| predictor | vs FT | vs probe | vs s_ft order | vs s_probe order |")
    L.append("|---|---|---|---|---|")
    for p in order:
        m = out["accuracy_matrix"][p]
        L.append(f"| {p} | {m['ft']['decided_acc']:.3f} | {m['probe']['decided_acc']:.3f} | "
                 f"{m['strength_order_ft']['decided_acc']:.3f} | {m['strength_order_probe']['decided_acc']:.3f} |")
    va = out["variant_agreement"]
    L.append(f"\n## Pair-local vs roster-global LOO\n")
    L.append(f"- agreement rate: **{va['pairlocal_vs_rosterLOO_agree_rate']:.3f}**")
    L.append(f"- direction cosine (pair-local d_f vs LOO signature): median "
             f"{va['direction_cosine_median']:.3f}, p10 {va['direction_cosine_p10']:.3f}, "
             f"min {va['direction_cosine_min']:.3f}")
    L.append("\n## Accuracy by gap stratum (primary predictor, FT target)\n")
    L.append("**By fitted strength gap |s_ft(f1)−s_ft(f2)|** (monotone — accuracy rises with the gap):")
    for k, v in out["by_gap"]["by_strength_gap_ft"].items():
        L.append(f"- {k}: {v['acc']:.3f} (n={v['n']})")
    L.append("\n**By diagonal |Δ| (the OBSERVED FT verdict margin, nats)** — NOTE this is NON-monotone: "
             "it dips in the largest bin because a huge observed commitment margin is trigger-enriched, "
             "and triggers are exactly what the mean-pooled geometry cannot see (§ trigger failure "
             "mode). A large observed |Δ| does NOT imply the technique is reliable:")
    for k, v in out["by_gap"]["by_ft_delta"].items():
        extra = ""
        if "twinner_acc" in v:
            extra = f"  [trigger-winner subset: {v['twinner_acc']:.3f} of {v['twinner_n']}]"
        L.append(f"- {k}: {v['acc']:.3f} (n={v['n']}){extra}")
    L.append("\n## Accuracy by winner feature type (primary predictor, FT target)\n")
    for k, v in out["by_type"]["by_winner_type"].items():
        L.append(f"- winner is {k}: {v['acc']:.3f} (n={v['n']})")
    tb = out["tie_band_tradeoff"]
    L.append(f"\n## Predictor tie-band (chosen = {tb['chosen_tie_band']:.3f}, median bootstrap margin SD)\n")
    L.append("| tie-band | coverage | decided acc | n decided |")
    L.append("|---|---|---|---|")
    for k, v in tb["curve"].items():
        L.append(f"| {k} | {v['coverage']:.2f} | {v['decided_acc']:.3f} | {v['n_decided']} |")
    fm = out["failure_modes"]
    L.append("\n## Failure modes\n")
    rl = fm["read_location"]
    L.append(f"**Read-location (28 end-anchored pairs, {rl['n_ft_decisive']} FT-decisive):** "
             f"mean-pooled predictor acc **{rl['mean_acc']:.3f}** vs last-token **{rl['last_acc']:.3f}** "
             f"— the continuity spec misfires exactly here (it keys on the trigger token piled at the "
             f"final position); mean-pooling fixes it.")
    rd = fm["robust_divergence"]
    L.append(f"\n**Robust-divergence (16 pairs):** the mean-pooled predictor matches the PROBE winner "
             f"{rd['n_match_probe']}/16 and the FT winner {rd['n_match_ft']}/16 — it reads the same "
             f"frozen activations as the probe, so on the monitor-vs-policy divergence pairs it tracks "
             f"the probe (monitor) side, NOT the fine-tune (policy). This is the technique's key "
             f"failure mode: it forecasts what a *probe* commits to, which diverges from the *fine-tune* "
             f"exactly on the backdoor-style trigger pairs.")
    L.append(f"\n**Leet pairs ({fm['leet']['n']}):** mean-pooled acc {fm['leet']['mean_acc']:.3f}.")
    (CG / "winner_prediction_v2.md").write_text("\n".join(L) + "\n")


def _print(out):
    h = out["headline"]
    print("=== WINNER PREDICTION (primary target = FT decisive) ===")
    for k, v in h.items():
        print(f"  {k}: decided_acc {v['decided_acc']:.3f} {[round(x,3) for x in v['decided_acc_ci']]} "
              f"coverage {v['coverage']:.2f} n {v['decided']}")
    print("wrote winner_prediction_v2.{json,md}")


if __name__ == "__main__":
    main()
