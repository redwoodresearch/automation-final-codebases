"""Held-cell asymmetry predictability analysis (footnote backing).

For each pair's 2x2 grid, the diagonal-trained classifier is read out on the two antidiagonal
held cells: held_f1pos=(f1+,f2-) and held_f1neg=(f1-,f2+). Delta = mean of the two cells'
log-odds toward following f1. We ask: can we predict WHICH held cell the WINNER is followed in
more decisively, and by HOW MUCH, from per-feature strengths / activation overlap?

Instruments:
  * PROBE  : committed antidiagonal-probe per-cell reads (results/adp_<pair>__last.json), layer 21,
             uncontrolled C=0.1, 5 seeds (the frozen shipped probe). Per-cell margin toward f1 =
             logit(bounded_heldcell) where bounded_heldcell = mean_over_cell sigmoid(d*sign) =
             P(follow f1) in that cell. (adp stores per-cell BOUNDED and pooled raw; per-cell raw
             log-odds is not stored, so we use the logit of the per-cell follow-probability as the
             per-cell log-odds-scale margin. Monotone in the true per-cell margin.)
  * FINETUNE: committed per-cell RAW log-odds (results/finetune_full_values.jsonl:
             raw_held_f1pos / raw_held_f1neg, oriented toward f1). True log-odds, n=32.

Winner (per instrument) = feature the instrument follows = sign of pooled Delta.
Winner-relative cells:
  winner f1 -> own cell = held_f1pos (winner feature present), other cell = held_f1neg.
  winner f2 -> own cell = held_f1neg (winner feature present), other cell = held_f1pos.
  m_own   = margin toward winner on winner's OWN cell
  m_other = margin toward winner on the OTHER cell
  asym    = m_own - m_other      (signed held-cell asymmetry, toward the winner)
  which_cell_wins = 1 if m_own > m_other else 0  (own cell more decisive)

Self-compression claim: winner is LESS decisive on its OWN cell -> mean(asym) < 0.

Predictors: s_w, s_l (winner/loser fitted strengths, instrument-matched arm), gap=s_w-s_l,
cos = cos(d_f1,d_f2) activation-direction overlap (decision-token signature means, layer 21).
"""
from __future__ import annotations
import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.exists(_os.path.join(_ROOT, 'REPORT.md')):
    _ROOT = _os.path.dirname(_ROOT)
import json, math, glob, os
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

ROOT = _ROOT + ""
RESULTS = f"{ROOT}/codebase/results"
INOC = f"{ROOT}/inoc_data"
L = 21

def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))

# ---- metadata
mr = {json.loads(l)["pair"]: json.loads(l) for l in open(f"{INOC}/merged_readouts_v2.jsonl")}

# ---- strengths (last-token probe arm; ft arm)
refit = json.load(open(f"{INOC}/refit_v2.json"))
S_PROBE = refit["arms"]["without_excluded"]["probe"]["s"]
S_FT = refit["arms"]["without_excluded"]["ft"]["s"]

# ---- activation overlap cos(d_f1,d_f2) at layer 21 from decision-token signature means
zsig = np.load(f"{RESULTS}/reads_expansion/decision_token_signature_means_v2.npz")
def cos_overlap(pair):
    try:
        m = {c: zsig[f"{pair}__{c}"][1] for c in ["A", "B", "held_f1pos", "held_f1neg"]}  # idx1 = layer21
    except KeyError:
        return None
    d1 = (m["A"] + m["held_f1pos"]) / 2 - (m["B"] + m["held_f1neg"]) / 2
    d2 = (m["A"] + m["held_f1neg"]) / 2 - (m["B"] + m["held_f1pos"]) / 2
    den = np.linalg.norm(d1) * np.linalg.norm(d2)
    return float((d1 @ d2) / den) if den > 0 else None

def feat_names(pair):
    r = mr[pair]
    return r["f1"], r["f2"]

def build_row(pair, m_pos_f1, m_neg_f1, delta_pooled, Sarm):
    """m_pos_f1/m_neg_f1 = per-cell margin toward f1 (log-odds scale). delta_pooled = pooled Delta."""
    f1, f2 = feat_names(pair)
    if f1 not in Sarm or f2 not in Sarm:
        return None
    winner_is_f1 = delta_pooled > 0
    if winner_is_f1:
        m_own, m_other = m_pos_f1, m_neg_f1     # own cell = held_f1pos
        s_w, s_l = Sarm[f1], Sarm[f2]
    else:
        m_own, m_other = -m_neg_f1, -m_pos_f1   # winner f2; own cell = held_f1neg; toward-winner = -toward-f1
        s_w, s_l = Sarm[f2], Sarm[f1]
    c = cos_overlap(pair)
    if c is None:
        return None
    return {"pair": pair, "m_own": m_own, "m_other": m_other, "asym": m_own - m_other,
            "which_own_wins": 1 if m_own > m_other else 0,
            "s_w": s_w, "s_l": s_l, "gap": s_w - s_l, "abs_s_w": abs(s_w), "abs_s_l": abs(s_l),
            "cos": c, "delta": delta_pooled, "winner_is_f1": winner_is_f1,
            "verdict": mr[pair]["verdict"], "stratum": mr[pair]["stratum"]}

# ---------------- PROBE rows (committed adp last-token, layer 21) ----------------
def probe_rows(exclude_near_tie=True):
    rows = []
    for f in sorted(glob.glob(f"{RESULTS}/adp_*__last.json")):
        pair = os.path.basename(f)[4:-11]
        if pair not in mr:
            continue
        if exclude_near_tie and mr[pair]["verdict"] == "near_tie":
            continue
        d = json.load(open(f))
        try:
            lay = d["grid"]["uncontrolled"]["C=0.1"]["layers"][str(L)]
        except KeyError:
            continue
        m_pos = logit(lay["bounded_heldpos"])   # toward f1 on held_f1pos
        m_neg = logit(lay["bounded_heldneg"])   # toward f1 on held_f1neg
        delta = lay["raw_mean"]                  # pooled log-odds toward f1
        r = build_row(pair, m_pos, m_neg, delta, S_PROBE)
        if r:
            rows.append(r)
    return rows

# ---------------- FINETUNE rows (committed raw per-cell) ----------------
def ft_rows():
    rows = []
    ft = [json.loads(l) for l in open(f"{RESULTS}/finetune_full_values.jsonl")]
    for r in ft:
        pair = r["pair"]
        if pair not in mr:
            continue
        if r.get("raw_held_f1pos") is None:
            continue
        m_pos = r["raw_held_f1pos"]   # already toward f1
        m_neg = r["raw_held_f1neg"]   # already toward f1
        delta = r["ft_raw_mean"]
        row = build_row(pair, m_pos, m_neg, delta, S_FT)
        if row:
            rows.append(row)
    return rows

# ---------------- predictive evaluation ----------------
def loo_r2(X, y):
    X = np.asarray(X, float); y = np.asarray(y, float)
    n = len(y)
    pred = np.empty(n)
    for i in range(n):
        tr = np.ones(n, bool); tr[i] = False
        mdl = LinearRegression().fit(X[tr], y[tr])
        pred[i] = mdl.predict(X[i:i+1])[0]
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot, pred

def loo_sign_acc(X, ybin):
    X = np.asarray(X, float); ybin = np.asarray(ybin, int)
    n = len(ybin)
    if len(set(ybin)) < 2:
        return None, None
    pred = np.empty(n, int)
    for i in range(n):
        tr = np.ones(n, bool); tr[i] = False
        if len(set(ybin[tr])) < 2:
            pred[i] = int(round(ybin[tr].mean()))
            continue
        mdl = LogisticRegression(max_iter=1000).fit(X[tr], ybin[tr])
        pred[i] = int(mdl.predict(X[i:i+1])[0])
    acc = float((pred == ybin).mean())
    base = float(max(ybin.mean(), 1 - ybin.mean()))
    return acc, base

def standardize(cols):
    A = np.array(cols, float).T  # (n, k)
    mu = A.mean(0); sd = A.std(0); sd[sd == 0] = 1
    return (A - mu) / sd

def analyze(rows, label):
    n = len(rows)
    asym = [r["asym"] for r in rows]
    which = [r["which_own_wins"] for r in rows]
    s_w = [r["s_w"] for r in rows]; s_l = [r["s_l"] for r in rows]
    gap = [r["gap"] for r in rows]; cos = [r["cos"] for r in rows]
    absw = [r["abs_s_w"] for r in rows]; absl = [r["abs_s_l"] for r in rows]
    m_own = np.array([r["m_own"] for r in rows]); m_other = np.array([r["m_other"] for r in rows])

    predictor_sets = {
        "gap_only": [gap],
        "cos_only": [cos],
        "s_w+s_l": [s_w, s_l],
        "gap+cos": [gap, cos],
        "s_w+s_l+gap+cos": [s_w, s_l, gap, cos],
        "abs_s_w+abs_s_l+gap+cos": [absw, absl, gap, cos],
    }
    out = {"label": label, "n": n}
    # base rate for which-cell
    wm = np.mean(which)
    out["which_base_rate"] = float(max(wm, 1 - wm))
    out["frac_own_wins"] = float(wm)

    absasym = list(np.abs(asym))
    r2 = {}; r2abs = {}; signacc = {}
    for name, cols in predictor_sets.items():
        X = standardize(cols)
        r2[name] = float(loo_r2(X, asym)[0])
        r2abs[name] = float(loo_r2(X, absasym)[0])
        acc, base = loo_sign_acc(X, which)
        signacc[name] = None if acc is None else float(acc)
    out["loo_r2_predict_signed_asym"] = r2
    out["loo_r2_predict_abs_asym"] = r2abs
    out["loo_sign_acc_which_cell"] = signacc

    # correlations (Pearson) of each predictor with signed asym and |asym|
    out["pearson_asym"] = {k: float(np.corrcoef(v, asym)[0, 1]) for k, v in
                           {"gap": gap, "cos": cos, "s_w": s_w, "s_l": s_l}.items()}

    # self-compression: m_own vs m_other
    diff = m_own - m_other
    out["self_compression"] = {
        "mean_m_own": float(m_own.mean()), "mean_m_other": float(m_other.mean()),
        "mean_asym(m_own-m_other)": float(diff.mean()),
        "median_asym": float(np.median(diff)),
        "sd_asym": float(diff.std(ddof=1)),
        "cohens_dz": float(diff.mean() / diff.std(ddof=1)),
        "frac_m_own_less(self_compression)": float((diff < 0).mean()),
        "t_stat": float(diff.mean() / (diff.std(ddof=1) / np.sqrt(n))),
    }
    return out, rows

if __name__ == "__main__":
    results = {}
    pr = probe_rows(exclude_near_tie=True)
    prb, pr_rows = analyze(pr, "PROBE_last_L21_noNearTie")
    results["probe"] = prb

    pr_all = probe_rows(exclude_near_tie=False)
    prb_all, _ = analyze(pr_all, "PROBE_last_L21_allVerdicts")
    results["probe_all"] = prb_all

    fr = ft_rows()
    if len(fr) >= 8:
        ftb, _ = analyze(fr, "FINETUNE_rawpercell")
        results["finetune"] = ftb

    os.makedirs(f"{ROOT}/heldcell_asym", exist_ok=True)
    json.dump(results, open(f"{ROOT}/heldcell_asym/results.json", "w"), indent=2)
    # dump per-pair rows too
    json.dump({"probe": pr_rows, "finetune": fr}, open(f"{ROOT}/heldcell_asym/rows.json", "w"), indent=1)
    print(json.dumps(results, indent=2))
