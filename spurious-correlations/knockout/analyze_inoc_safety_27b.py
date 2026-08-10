"""Analyze the 27B inoculation sweep for the four non-deception safety behaviors and write
results/qwen36_27b/inoculation_safety_27b.json.

Mirrors analyze_inoc_lying_27b.py, generalized over the pair's safety behavior (orientation per
pair via inoc_safety_27b_common.toward_behavior; neutral_appended reused from the safety run's
ft/). Reports, per pair and aggregated PER FEATURE:
  * mean commitment TOWARD the behavior under the neutral prompt vs the named prompt (nats),
  * the naming shift (named - neutral) in matched layouts and vs the bare used prompt,
  * how many contests flip from behavior-loses (neutral toward_behavior < 0) to behavior-wins
    (named toward_behavior > 0).
Also carries the committed 1.7B shifts (both layouts) for the same pair/behavior for context.

Usage:
  cd knockout && PYTHONPATH=../codebase/lib:. ../codebase/.venv/bin/python analyze_inoc_safety_27b.py
"""

from __future__ import annotations

import json
import statistics

import inoc_safety_27b_common as IS
import qwen36_common as QC


def _seed_means(pair: str):
    """Per-condition toward_behavior, per seed, from the on-disk fine-tunes."""
    out: dict[str, dict] = {}
    na = {}
    for s in QC.SEEDS:
        d = QC.load_json(IS.neutral_appended_path(pair, s))
        na[str(s)] = {"toward_behavior": IS.toward_behavior(pair, d["heldout_raw"]),
                      "heldout_raw": d["heldout_raw"], "saved_step": d["saved_step"],
                      "train_acc": d["train_acc"]}
    out["neutral_appended"] = na
    for cond in IS.CONDITIONS:
        c = {}
        for s in QC.SEEDS:
            d = QC.load_json(IS.ft_path(pair, cond, s))
            c[str(s)] = {"toward_behavior": d["toward_behavior"], "heldout_raw": d["heldout_raw"],
                         "saved_step": d["saved_step"], "train_acc": d["train_acc"]}
        out[cond] = c
    return out


def _mean(cond: dict) -> float:
    return statistics.mean(v["toward_behavior"] for v in cond.values())


def _load_17b(pairs):
    """Committed 1.7B shifts (both layouts) from the inoc_first records, keyed by (pair, behavior)."""
    ifp = QC.KNOCKOUT_DIR / "results" / "inoc_first" / "inoc_first_records.jsonl"
    byk = {}
    for line in open(ifp):
        if line.strip():
            r = json.loads(line)
            byk[(r["pair"], r["named_feature"])] = r
    out = {}
    for p in pairs:
        r = byk[(p, IS.behavior_of(p))]
        out[p] = {
            "first_neutral_toward_behavior": r["first_neutral_toward_named"],
            "first_named_toward_behavior": r["first_named_delta"],
            "first_shift": r["first_shift"],
            "appended_neutral_toward_behavior": r["appended_neutral_toward_named"],
            "appended_named_toward_behavior": r["appended_named_delta"],
            "appended_shift": r["appended_shift"],
        }
    return out


def _pair_row(p: str, c17: dict) -> dict:
    sm = _seed_means(p)
    na = _mean(sm["neutral_appended"])
    nf = _mean(sm["neutral_first"])
    mf = _mean(sm["named_first"])
    ma = _mean(sm["named_appended"])
    return {
        "pair": p,
        "behavior": IS.behavior_of(p),
        "neutral_appended_toward_behavior_27b": na,
        "neutral_first_toward_behavior_27b": nf,
        "named_first_toward_behavior_27b": mf,
        "named_appended_toward_behavior_27b": ma,
        "shift_first_27b": mf - nf,                 # instruction-first, matched (primary)
        "shift_appended_27b": ma - na,              # appended, matched (~ 1.7B appended_shift)
        "shift_named_first_vs_bare_27b": mf - na,   # named-before-passage vs the bare used prompt
        # flip: neutral loses (bare prompt < 0) -> named-first wins (> 0)
        "neutral_loses_27b": bool(na < 0),
        "named_first_wins_27b": bool(mf > 0),
        "flips_toward_behavior_27b": bool(na < 0 and mf > 0),
        "seed_detail_27b": sm,
        "b1p7": c17[p],
    }


def _feature_summary(rows: list[dict]) -> dict:
    def mm(key):
        return statistics.mean(r[key] for r in rows)
    return {
        "n_pairs": len(rows),
        "neutral_appended_toward_behavior": mm("neutral_appended_toward_behavior_27b"),
        "neutral_first_toward_behavior": mm("neutral_first_toward_behavior_27b"),
        "named_first_toward_behavior": mm("named_first_toward_behavior_27b"),
        "named_appended_toward_behavior": mm("named_appended_toward_behavior_27b"),
        "shift_first": mm("shift_first_27b"),
        "shift_appended": mm("shift_appended_27b"),
        "shift_named_first_vs_bare": mm("shift_named_first_vs_bare_27b"),
        "n_named_first_wins": sum(r["named_first_wins_27b"] for r in rows),
        "n_neutral_loses": sum(r["neutral_loses_27b"] for r in rows),
        "n_flips_toward_behavior": sum(r["flips_toward_behavior_27b"] for r in rows),
        # 1.7B reference (appended layout, the committed arm)
        "b1p7_appended_shift": mm_key(rows, "appended_shift"),
        "b1p7_first_shift": mm_key(rows, "first_shift"),
    }


def mm_key(rows, key):
    return statistics.mean(r["b1p7"][key] for r in rows)


def main() -> None:
    all_rows = []
    per_feature = {}
    for feat in IS.FEATURES:
        pairs = [p for p in IS.PAIRS_BY_FEATURE[feat]
                 if all(IS.ft_path(p, c, s).exists() for c in IS.CONDITIONS for s in QC.SEEDS)]
        c17 = _load_17b(pairs)
        rows = [_pair_row(p, c17) for p in pairs]
        all_rows.extend(rows)
        per_feature[feat] = {"summary": _feature_summary(rows), "rows": rows} if rows else {
            "summary": {"n_pairs": 0}, "rows": []}

    overall = {
        "note": ("toward_behavior = commitment of the held-cell read toward the pair's safety "
                 "behavior, in nats; >0 = the fine-tune committed to the behavior, <0 = it followed "
                 "the style/partner. Orientation per pair (all 22 have behavior as f2, so "
                 "toward_behavior = -heldout_raw). shift = named - neutral (matched layout). "
                 "flip = bare-neutral loses (<0) AND named_first wins (>0). seeds [0,1]."),
        "conditions": ["neutral_appended (reused)", "neutral_first", "named_first", "named_appended"],
        "features": IS.FEATURES,
        "n_pairs_total": len(all_rows),
        "per_feature": {f: per_feature[f]["summary"] for f in IS.FEATURES},
    }
    out = {"overall": overall, "per_feature": per_feature}
    QC.save_json(IS.OUT_FILE, out)

    # console table
    print(json.dumps(overall, indent=1))
    print("\n=== per-feature table (nats, seed means) ===")
    hdr = f"{'feature':22s} {'n':>2s} {'neut_bare':>9s} {'named_1st':>9s} {'shift':>7s} {'flips':>7s}"
    print(hdr)
    for f in IS.FEATURES:
        s = per_feature[f]["summary"]
        if not s.get("n_pairs"):
            print(f"{f:22s}  0   (no complete pairs)")
            continue
        print(f"{f:22s} {s['n_pairs']:>2d} {s['neutral_appended_toward_behavior']:>+9.2f} "
              f"{s['named_first_toward_behavior']:>+9.2f} {s['shift_named_first_vs_bare']:>+7.2f} "
              f"{s['n_flips_toward_behavior']:>3d}/{s['n_pairs']:<3d}")
    print(f"\nwrote {IS.OUT_FILE}")


if __name__ == "__main__":
    main()
