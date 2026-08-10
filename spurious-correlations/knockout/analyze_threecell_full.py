"""Analysis for the full-pair-graph three-cell experiment (results/threecell_full/runs.jsonl).

Per run, "rescue" = the weaker (label) feature wins the held-out cell: out_logodds_true > 0
(mean log-odds toward the true weak-feature label; the partner shortcut predicts the wrong label
on every held-out cell, so this is a clean two-way fight). Sections:

  1. Transition curve: P(rescue) vs the pair's strength gap (s_partner - s_label), logistic fit,
     critical gap (50% point) with a pair-level cluster-bootstrap CI. Weaker-PRESENT holdouts only
     (the variant set every pair has).
  2. Behaviors special? At matched gap, rescue rates by weaker-feature type (logistic with type
     offsets, converted to gap-equivalent nats, plus a binned table).
  3. Memorize-vs-generalize map: per-run mechanism signature (train-acc shelf at the partner-rule
     ceiling + no plateau + depressed val acc on the shortcut-contradicting cell = memorization;
     early plateau + clean val = generalization), related to rescue outcome and feature type.
  4. Asymmetry on the 4-variant subset: weaker-present vs weaker-absent holdouts, paired by pair.
  5. Seed consistency + the bimodal near-transition band.

Usage: PYTHONPATH=../codebase/lib:. ../codebase/.venv/bin/python analyze_threecell_full.py
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

import threecell_full_common as C

RUNS = C.RESULTS_DIR / "runs.jsonl"
OUT_JSON = C.RESULTS_DIR / "analysis.json"

TYPES = ["content_behavior", "style_voice", "discourse_structure", "triggers", "surface_mechanical"]
GAP_BINS = [0, 4, 8, 12, 16, 21, 26]
N_BOOT = 2000
SHELF_TOL = 0.04          # |train_acc - shortcut ceiling| at a mid eval step
VAL_CLEAN = 0.97          # in-dist val gate for "generalization"
VAL_CONTRA_DEPRESSED = 0.85


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-4, iters: int = 200) -> np.ndarray:
    """IRLS logistic regression with intercept prepended. Returns beta (len = 1 + X.shape[1])."""
    Xd = np.column_stack([np.ones(len(y)), X])
    beta = np.zeros(Xd.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xd @ beta))
        W = p * (1 - p) + 1e-9
        H = Xd.T @ (Xd * W[:, None]) + l2 * np.eye(Xd.shape[1])
        g = Xd.T @ (y - p) - l2 * beta
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def mechanism(r: dict) -> dict:
    ceiling = r["shortcut_ceiling"]
    mid = [c for c in r["curve_no_d"] if 25 <= c["step"] <= 150]
    shelf = any(abs(c["train_acc"] - ceiling) <= SHELF_TOL for c in mid)
    no_plateau = r["plateau_step"] is None
    val_contra = r["val_acc_by_cell"].get(r["contradicting_cell"], float("nan"))
    contra_depressed = val_contra < VAL_CONTRA_DEPRESSED
    if no_plateau and (shelf or contra_depressed):
        mech = "memorization"
    elif not no_plateau and r["val_acc"] >= VAL_CLEAN and not contra_depressed:
        mech = "generalization"
    else:
        mech = "mixed"
    return {"mech": mech, "shelf": shelf, "no_plateau": no_plateau,
            "val_contra": val_contra, "contra_depressed": contra_depressed}


def critical_gap(beta: np.ndarray) -> float:
    return float(-beta[0] / beta[1]) if beta[1] != 0 else float("nan")


def main() -> None:
    runs = list(C.load_done(RUNS).values())
    for r in runs:
        r["win"] = r["out_logodds_true"] > 0
        r.update(mechanism(r))
    present = [r for r in runs if r["holdout_is_label_present"]]
    absent = [r for r in runs if not r["holdout_is_label_present"]]
    report: dict = {"n_runs": len(runs), "n_present": len(present), "n_absent": len(absent)}
    print(f"{len(runs)} runs loaded ({len(present)} weaker-present holdouts, {len(absent)} weaker-absent)")
    npairs = len({r["pair"] for r in runs})
    print(f"{npairs} pairs; mean val_acc {np.mean([r['val_acc'] for r in runs]):.3f}; "
          f"val_acc<0.9 in {np.mean([r['val_acc'] < 0.9 for r in runs]) * 100:.1f}% of runs")

    # ---------- 1. transition curve ----------
    gaps = np.array([r["gap"] for r in present])
    wins = np.array([float(r["win"]) for r in present])
    pair_ids = np.array([r["pair"] for r in present])
    beta = fit_logistic(gaps[:, None], wins)
    cg = critical_gap(beta)
    uniq = np.unique(pair_ids)
    rng = np.random.default_rng(0)
    boots = []
    idx_by_pair = {p: np.flatnonzero(pair_ids == p) for p in uniq}
    for _ in range(N_BOOT):
        take = np.concatenate([idx_by_pair[p] for p in rng.choice(uniq, size=len(uniq))])
        boots.append(critical_gap(fit_logistic(gaps[take, None], wins[take])))
    boots = np.array(boots)
    ci = np.percentile(boots, [2.5, 97.5])
    print("\n=== 1. TRANSITION CURVE (weaker-present holdouts) ===")
    print(f"P(rescue) overall: {wins.mean():.3f}   logistic: b0={beta[0]:+.3f}, b_gap={beta[1]:+.4f}/nat")
    print(f"critical gap (50% rescue): {cg:.2f} nats  [95% pair-bootstrap CI {ci[0]:.2f}, {ci[1]:.2f}]")
    binrows = []
    for lo, hi in zip(GAP_BINS[:-1], GAP_BINS[1:]):
        m = (gaps >= lo) & (gaps < hi)
        if m.sum():
            binrows.append({"bin": f"[{lo},{hi})", "n": int(m.sum()), "rescue": float(wins[m].mean())})
            print(f"  gap [{lo:2d},{hi:2d}): n={m.sum():4d}  P(rescue)={wins[m].mean():.3f}")
    report["transition"] = {"beta": beta.tolist(), "critical_gap": cg, "ci95": ci.tolist(),
                            "overall_rescue": float(wins.mean()), "bins": binrows}

    # ---------- 2. behaviors special? ----------
    print("\n=== 2. BEHAVIORS SPECIAL? (weaker-feature type at matched gap; present holdouts) ===")
    types = np.array([r["label_type"] for r in present])
    X = np.column_stack([gaps] + [(types == t).astype(float) for t in TYPES[1:]])  # base=content_behavior
    bt = fit_logistic(X, wins)
    b_gap = bt[1]
    offsets = {TYPES[0]: 0.0}
    for i, t in enumerate(TYPES[1:]):
        offsets[t] = float(bt[2 + i])
    print(f"logistic: win ~ gap + type (baseline content_behavior); b_gap={b_gap:+.4f}/nat")
    boot_off = {t: [] for t in TYPES[1:]}
    for _ in range(N_BOOT // 2):
        take = np.concatenate([idx_by_pair[p] for p in rng.choice(uniq, size=len(uniq))])
        try:
            bb = fit_logistic(X[take], wins[take])
        except np.linalg.LinAlgError:
            continue
        for i, t in enumerate(TYPES[1:]):
            boot_off[t].append(bb[2 + i])
    typerows = []
    for t in TYPES:
        m = types == t
        row = {"type": t, "n": int(m.sum()), "raw_rescue": float(wins[m].mean()),
               "mean_gap": float(gaps[m].mean()), "logit_offset_vs_behavior": offsets[t],
               "gap_equiv_nats": float(offsets[t] / -b_gap) if b_gap else float("nan")}
        if t in boot_off and boot_off[t]:
            row["offset_ci95"] = np.percentile(boot_off[t], [2.5, 97.5]).tolist()
        typerows.append(row)
        ci_s = (f"  [CI {row['offset_ci95'][0]:+.2f},{row['offset_ci95'][1]:+.2f}]"
                if "offset_ci95" in row else "")
        print(f"  {t:20s} n={row['n']:4d} raw_rescue={row['raw_rescue']:.3f} mean_gap={row['mean_gap']:5.2f} "
              f"offset={offsets[t]:+.3f} logits (={row['gap_equiv_nats']:+.2f} gap-nats){ci_s}")
    print("  binned rescue rates (type x gap bin):")
    hdr = "    " + f"{'':22s}" + "".join(f"[{lo},{hi}) ".rjust(10) for lo, hi in zip(GAP_BINS[:-1], GAP_BINS[1:]))
    print(hdr)
    binned = {}
    for t in TYPES:
        cells = []
        for lo, hi in zip(GAP_BINS[:-1], GAP_BINS[1:]):
            m = (types == t) & (gaps >= lo) & (gaps < hi)
            cells.append(f"{wins[m].mean():.2f}({m.sum()})".rjust(10) if m.sum() else " -".rjust(10))
            binned[f"{t}|[{lo},{hi})"] = {"n": int(m.sum()), "rescue": float(wins[m].mean()) if m.sum() else None}
        print(f"    {t:22s}" + "".join(cells))
    report["behaviors"] = {"type_rows": typerows, "binned": binned, "b_gap": float(b_gap)}

    # ---------- 3. memorize-vs-generalize map ----------
    print("\n=== 3. MEMORIZE-VS-GENERALIZE MAP (all runs) ===")
    mech_rows = {}
    for mech in ["generalization", "mixed", "memorization"]:
        rs = [r for r in runs if r["mech"] == mech]
        if not rs:
            continue
        row = {"n": len(rs), "frac": len(rs) / len(runs),
               "rescue_rate": float(np.mean([r["win"] for r in rs])),
               "mean_gap": float(np.mean([r["gap"] for r in rs])),
               "mean_val_contra": float(np.nanmean([r["val_contra"] for r in rs])),
               "shelf_frac": float(np.mean([r["shelf"] for r in rs]))}
        mech_rows[mech] = row
        print(f"  {mech:15s} n={row['n']:4d} ({row['frac'] * 100:4.1f}%)  rescue={row['rescue_rate']:.3f}  "
              f"mean_gap={row['mean_gap']:5.2f}  val_contra={row['mean_val_contra']:.3f}  "
              f"shelf={row['shelf_frac']:.2f}")
    print("  mechanism by weaker-feature type (present holdouts):")
    mech_by_type = {}
    for t in TYPES:
        rs = [r for r in present if r["label_type"] == t]
        if not rs:
            continue
        d = {m: float(np.mean([r["mech"] == m for r in rs])) for m in ["generalization", "mixed", "memorization"]}
        mech_by_type[t] = {"n": len(rs), **d}
        print(f"    {t:20s} n={len(rs):4d}  gen={d['generalization']:.2f} mixed={d['mixed']:.2f} "
              f"mem={d['memorization']:.2f}")
    report["mechanism"] = {"rows": mech_rows, "by_type": mech_by_type}

    # ---------- 4. asymmetry (4-variant subset) ----------
    print("\n=== 4. ASYMMETRY: weaker-present vs weaker-absent holdouts (4-variant pairs) ===")
    by_pair = defaultdict(lambda: {"present": [], "absent": []})
    for r in runs:
        if r["four_variant"]:
            by_pair[r["pair"]]["present" if r["holdout_is_label_present"] else "absent"].append(r)
    pres_r, abs_r, d_lo, d_win = [], [], [], []
    for pair, d in by_pair.items():
        if not d["present"] or not d["absent"]:
            continue
        pw = np.mean([r["win"] for r in d["present"]])
        aw = np.mean([r["win"] for r in d["absent"]])
        pl = np.mean([r["out_logodds_true"] for r in d["present"]])
        al = np.mean([r["out_logodds_true"] for r in d["absent"]])
        pres_r.append(pw); abs_r.append(aw); d_win.append(pw - aw); d_lo.append(pl - al)
    d_lo, d_win = np.array(d_lo), np.array(d_win)
    boot_dlo = [np.mean(rng.choice(d_lo, size=len(d_lo))) for _ in range(N_BOOT)]
    ci_dlo = np.percentile(boot_dlo, [2.5, 97.5])
    print(f"  {len(d_lo)} pairs with both variant families")
    print(f"  rescue rate: present {np.mean(pres_r):.3f} vs absent {np.mean(abs_r):.3f} "
          f"(paired mean diff {d_win.mean():+.3f})")
    print(f"  mean out log-odds diff (present - absent): {d_lo.mean():+.2f} nats "
          f"[95% CI {ci_dlo[0]:+.2f}, {ci_dlo[1]:+.2f}]")
    report["asymmetry"] = {"n_pairs": len(d_lo), "present_rescue": float(np.mean(pres_r)),
                           "absent_rescue": float(np.mean(abs_r)), "paired_win_diff": float(d_win.mean()),
                           "paired_logodds_diff": float(d_lo.mean()), "logodds_diff_ci95": ci_dlo.tolist()}

    # ---------- 5. seed consistency / near-transition bimodality ----------
    print("\n=== 5. SEED CONSISTENCY ===")
    cond = defaultdict(list)
    for r in runs:
        cond[(r["pair"], r["holdout_cell"])].append(r)
    both = {k: v for k, v in cond.items() if len(v) == 2}
    agree = np.array([v[0]["win"] == v[1]["win"] for v in both.values()])
    cgaps = np.array([v[0]["gap"] for v in both.values()])
    print(f"  {len(both)} (pair x variant) conditions with 2 seeds; seed win/loss agreement {agree.mean():.3f}")
    disrows = []
    for lo, hi in zip(GAP_BINS[:-1], GAP_BINS[1:]):
        m = (cgaps >= lo) & (cgaps < hi)
        if m.sum():
            disrows.append({"bin": f"[{lo},{hi})", "n": int(m.sum()), "disagree": float(1 - agree[m].mean())})
            print(f"    gap [{lo:2d},{hi:2d}): n={m.sum():4d}  seed-disagreement {1 - agree[m].mean():.3f}")
    band_lo, band_hi = (max(0.0, cg - 4), cg + 4) if np.isfinite(cg) else (4, 12)
    band = [v for v in both.values() if band_lo <= v[0]["gap"] < band_hi]
    blo = np.array([r["out_logodds_true"] for v in band for r in v])
    frac_mid = float(np.mean(np.abs(blo) < 2)) if len(blo) else float("nan")
    frac_far = float(np.mean(np.abs(blo) > 5)) if len(blo) else float("nan")
    print(f"  near-transition band gap [{band_lo:.1f},{band_hi:.1f}): {len(band)} conditions; "
          f"per-run |log-odds|<2 in {frac_mid * 100:.0f}%, >5 in {frac_far * 100:.0f}% "
          f"({'bimodal (decisive-but-random)' if frac_far > frac_mid else 'graded'})")
    dis_band = [v for v in band if v[0]["win"] != v[1]["win"]]
    print(f"  band seed-disagreement: {len(dis_band)}/{len(band)}")
    report["seeds"] = {"n_conditions": len(both), "agreement": float(agree.mean()),
                       "disagreement_by_bin": disrows,
                       "band": {"lo": band_lo, "hi": band_hi, "n": len(band),
                                "frac_absloggodds_lt2": frac_mid, "frac_gt5": frac_far,
                                "seed_disagree": len(dis_band)}}

    OUT_JSON.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
