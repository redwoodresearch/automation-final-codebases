#!/usr/bin/env python3
"""Re-derive the near-tie collapse / commitment numbers for the full-37 roster
from committed data only (no GPU/Modal/new fine-tunes).

Feeds the blog paragraph "When strengths tie, fine-tuning gives up on both."

Three products:
  1. VALIDATION - reproduce commitment_stability_census.json's 10 unstable +
     13 high-spread-same-winner pairs from merged_readouts_v2.jsonl `ft_seeds`
     (criterion: cross-run spread max-min >= 15 nats; unstable = signs disagree).
  2. COMMIT RATE BY STRATUM (per-pair, all 663 pairs, full-37) - a pair "commits"
     when its 3 fine-tune runs agree on the winning feature (all same sign of the
     signed run value); rate by gap stratum + Spearman(commit, gap).
  3. CONSTANT-LABEL COLLAPSE (per-example, from ft_grid_mass_diag_fitted.jsonl
     d_values) - fraction of near-tie SEEDS that confidently emit a fixed token
     (at-chance accuracy + confident + one-sided predictions) vs mid/well-sep.

Sources (all committed, read-only):
  results/complete_graph/merged_readouts_v2.jsonl  -> ft_seeds (per-run signed vals,
      == census ft_run_values), stratum (|ft_delta| binned at 3 and 8 nats),
      ft_sign_consistent.
  results/complete_graph/refit_v2.json             -> arms.without_excluded.ft.s
      (per-feature FT strength; additive-s gap = |s[f1]-s[f2]|).
  results/ft_grid_mass_diag_fitted.jsonl           -> per-example d_values (336
      confounded-diagonal pairs x 3 seeds, single lr 1e-5).
  results/complete_graph/commitment_stability_census.json -> ground truth to match.
"""
import json
import os
import statistics as S
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(BASE, "results")
CG = os.path.join(R, "complete_graph")

SPREAD_NATS = 15.0          # census cross-run spread threshold
COLLAPSE_CONF_NATS = 4.0    # mean|d_values| for "confident" emission
COLLAPSE_SIDEDNESS = 0.25   # |frac_positive - 0.5| for "one-sided" (constant-label)


def spearman(x, y):
    n = len(x)

    def rank(a):
        order = sorted(range(n), key=lambda i: a[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and a[order[j + 1]] == a[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    mx, my = S.mean(rx), S.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main():
    merged = [json.loads(l) for l in open(os.path.join(CG, "merged_readouts_v2.jsonl"))]
    census = json.load(open(os.path.join(CG, "commitment_stability_census.json")))
    refit = json.load(open(os.path.join(CG, "refit_v2.json")))
    s_ft = refit["arms"]["without_excluded"]["ft"]["s"]
    strat = {r["pair"]: r["stratum"] for r in merged}

    def seeds(r):
        return [r["ft_seeds"][k] for k in ("0", "1", "2")]

    # ---- 1. VALIDATION -----------------------------------------------------
    my_unstable, my_highspread = [], []
    for r in merged:
        v = seeds(r)
        spread = max(v) - min(v)
        if spread >= SPREAD_NATS:
            signs = set(1 if x > 0 else -1 for x in v)
            (my_unstable if len(signs) > 1 else my_highspread).append(r["pair"])
    cu = set(p["pair"] for p in census["unstable_commitment_pairs"])
    ch = set(p["pair"] for p in census["high_spread_same_winner_pairs"])
    validation = {
        "criterion": "spread(max-min of 3 ft_seeds) >= 15 nats; unstable = signs disagree",
        "my_unstable_n": len(my_unstable),
        "census_unstable_n": len(cu),
        "unstable_exact_match": set(my_unstable) == cu,
        "my_highspread_n": len(my_highspread),
        "census_highspread_n": len(ch),
        "highspread_exact_match": set(my_highspread) == ch,
    }

    # ---- 2. COMMIT RATE BY STRATUM (per-pair, all 663) ---------------------
    by = defaultdict(lambda: [0, 0])
    commit, gap_delta, gap_s = [], [], []
    for r in merged:
        st = r["stratum"]
        by[st][1] += 1
        if r["ft_sign_consistent"]:
            by[st][0] += 1
        commit.append(1 if r["ft_sign_consistent"] else 0)
        gap_delta.append(abs(r["ft_delta"]))
        gap_s.append(abs(s_ft[r["f1"]] - s_ft[r["f2"]]))
    commit_rate = {
        st: {"commit": by[st][0], "n": by[st][1], "rate": round(by[st][0] / by[st][1], 3)}
        for st in ("near_tie", "mid_gap", "well_separated")
    }
    commit_rate["overall"] = {
        "commit": sum(v[0] for v in by.values()),
        "n": sum(v[1] for v in by.values()),
        "rate": round(sum(v[0] for v in by.values()) / sum(v[1] for v in by.values()), 3),
    }
    spearmans = {
        "per_pair_commit_vs_abs_ft_delta": round(spearman(commit, gap_delta), 3),
        "per_pair_commit_vs_additive_s_gap": round(spearman(commit, gap_s), 3),
        "stratum_level_rate_vs_mean_gap_3pts": round(
            spearman(
                [commit_rate[st]["rate"] for st in ("near_tie", "mid_gap", "well_separated")],
                [
                    S.mean(abs(r["ft_delta"]) for r in merged if r["stratum"] == st)
                    for st in ("near_tie", "mid_gap", "well_separated")
                ],
            ),
            3,
        ),
    }

    # ---- 3. CONSTANT-LABEL COLLAPSE (per-example d_values, mass-diag) -------
    mass = [json.loads(l) for l in open(os.path.join(R, "ft_grid_mass_diag_fitted.jsonl"))]

    def is_collapse(fs):
        dv = fs["d_values"]
        n = len(dv)
        mabs = S.mean(abs(x) for x in dv)
        fpos = sum(1 for x in dv if x > 0) / n
        # confident (large |d|), at-chance accuracy, one-sided emission ->
        # model confidently pushes ~every input to a FIXED token, uncorrelated
        # with the true (diagonal) label. On the confounded diagonal both
        # features predict the label, so at-chance acc rules out "learned a
        # feature"; it is the constant-label signature.
        return (0.4 <= fs["heldout_acc"] <= 0.6) and (mabs >= COLLAPSE_CONF_NATS) and (
            abs(fpos - 0.5) >= COLLAPSE_SIDEDNESS
        )

    cby = defaultdict(lambda: [0, 0])
    npairs = defaultdict(set)
    for r in mass:
        st = strat[r["pair"]]
        npairs[st].add(r["pair"])
        cby[st][1] += 1
        if is_collapse(r["curve"][-1]):
            cby[st][0] += 1
    collapse = {
        "detector": {
            "confident_mean_abs_d_nats": COLLAPSE_CONF_NATS,
            "at_chance_acc_band": [0.4, 0.6],
            "one_sided_min_abs_fracpos_minus_half": COLLAPSE_SIDEDNESS,
            "eval": "confounded-diagonal held-out; d_values per-example, single lr 1e-5",
        },
        "by_stratum": {
            st: {
                "collapse_seeds": cby[st][0],
                "n_seeds": cby[st][1],
                "n_pairs": len(npairs[st]),
                "rate": round(cby[st][0] / cby[st][1], 3),
            }
            for st in ("near_tie", "mid_gap", "well_separated")
        },
    }

    out = {
        "note": "Full-37 near-tie collapse / commitment numbers for the blog, committed data only.",
        "validation": validation,
        "commit_rate_by_stratum": commit_rate,
        "commit_spearmans": spearmans,
        "constant_label_collapse": collapse,
        "coverage": {
            "commit_rate": "full-37, all 663 pairs x 3 fine-tune seeds (merged_readouts_v2)",
            "collapse": "mass-diag subset: 336 pairs x 3 seeds, single lr 1e-5 "
            "(near_tie 38 pairs/114 seeds, mid_gap 77/231, well_sep 221/663)",
        },
    }
    outdir = CG
    outpath = os.path.join(outdir, "near_tie_collapse_commit_full37.json")
    json.dump(out, open(outpath, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print("\nwrote", outpath)


if __name__ == "__main__":
    main()
