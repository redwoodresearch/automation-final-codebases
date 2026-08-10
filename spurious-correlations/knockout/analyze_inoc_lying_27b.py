"""Analyze the 27B inoculation-for-lying sweep and write results/qwen36_27b/inoculation_lying_27b.json.

For each deception pair, reads the four fine-tune conditions (neutral_appended reused from the safety
run; neutral_first / named_first / named_appended from inoc_lying/), orients every held-cell read
TOWARD lying (deception = f2, so toward_lying = -heldout_raw), and reports the inoculation shift
(named - neutral) in three matched forms, plus the committed 1.7B shifts for the same pairs.

Usage:
  cd knockout && PYTHONPATH=../codebase/lib:. ../codebase/.venv/bin/python analyze_inoc_lying_27b.py
"""

from __future__ import annotations

import json
import statistics

import inoc_lying_27b_common as IL
import qwen36_common as QC


def _seed_means(pair: str):
    """Per-condition toward_lying, per seed and mean, from the on-disk fine-tunes."""
    out: dict[str, dict] = {}
    # neutral_appended: reuse the safety run's bare-classify fine-tune
    na = {}
    for s in QC.SEEDS:
        d = QC.load_json(IL.neutral_appended_path(pair, s))
        na[str(s)] = {"toward_lying": IL.toward_lying(d["heldout_raw"]),
                      "heldout_raw": d["heldout_raw"], "saved_step": d["saved_step"],
                      "train_acc": d["train_acc"]}
    out["neutral_appended"] = na
    for cond in IL.NEW_CONDITIONS:
        c = {}
        for s in QC.SEEDS:
            d = QC.load_json(IL.ft_path(pair, cond, s))
            c[str(s)] = {"toward_lying": d["toward_lying"], "heldout_raw": d["heldout_raw"],
                         "saved_step": d["saved_step"], "train_acc": d["train_acc"]}
        out[cond] = c
    return out


def _mean(cond: dict) -> float:
    return statistics.mean(v["toward_lying"] for v in cond.values())


def _load_17b(pairs):
    """Committed 1.7B shifts (both layouts) from the inoc_first records, keyed by (pair, named)."""
    ifp = QC.KNOCKOUT_DIR / "results" / "inoc_first" / "inoc_first_records.jsonl"
    byk = {}
    for line in open(ifp):
        if line.strip():
            r = json.loads(line)
            byk[(r["pair"], r["named_feature"])] = r
    out = {}
    for p in pairs:
        r = byk[(p, IL.NAMED_FEATURE)]
        out[p] = {
            "first_neutral_toward_lying": r["first_neutral_toward_named"],
            "first_named_toward_lying": r["first_named_delta"],
            "first_shift": r["first_shift"],
            "appended_neutral_toward_lying": r["appended_neutral_toward_named"],
            "appended_named_toward_lying": r["appended_named_delta"],
            "appended_shift": r["appended_shift"],
        }
    return out


def main() -> None:
    pairs = [p for p in IL.LYING_PAIRS
             if all(IL.ft_path(p, c, s).exists() for c in IL.NEW_CONDITIONS for s in QC.SEEDS)]
    c17 = _load_17b(pairs)
    rows = []
    for p in pairs:
        IL.assert_deception_is_f2(p)
        sm = _seed_means(p)
        na = _mean(sm["neutral_appended"])
        nf = _mean(sm["neutral_first"])
        mf = _mean(sm["named_first"])
        ma = _mean(sm["named_appended"])
        rows.append({
            "pair": p,
            # 27B toward-lying commitment (nats) per condition (seed mean)
            "neutral_appended_toward_lying_27b": na,
            "neutral_first_toward_lying_27b": nf,
            "named_first_toward_lying_27b": mf,
            "named_appended_toward_lying_27b": ma,
            # inoculation shifts at 27B (named - neutral), matched layouts + vs the bare prompt
            "shift_first_27b": mf - nf,          # instruction-first, matched (primary; ~ 1.7B first_shift)
            "shift_appended_27b": ma - na,       # appended, matched (~ 1.7B appended_shift)
            "shift_named_first_vs_bare_27b": mf - na,  # requested: named-before-passage vs bare used prompt
            "named_first_rescues_lying_27b": bool(mf > 0),
            # per-seed detail
            "seed_detail_27b": sm,
            # committed 1.7B (both layouts) for the same pair, deception named
            "b1p7": c17[p],
        })

    def mm(key):
        return statistics.mean(r[key] for r in rows)

    summary = {
        "n_pairs": len(rows),
        "conditions": ["neutral_appended (reused)", "neutral_first", "named_first", "named_appended"],
        "note": ("toward_lying = commitment of the held-cell read toward deception (f2), in nats; "
                 ">0 = fine-tune committed toward lying. shift = named - neutral (matched layout). "
                 "seeds [0,1]."),
        "mean_27b": {
            "neutral_first_toward_lying": mm("neutral_first_toward_lying_27b"),
            "named_first_toward_lying": mm("named_first_toward_lying_27b"),
            "neutral_appended_toward_lying": mm("neutral_appended_toward_lying_27b"),
            "named_appended_toward_lying": mm("named_appended_toward_lying_27b"),
            "shift_first": mm("shift_first_27b"),
            "shift_appended": mm("shift_appended_27b"),
            "shift_named_first_vs_bare": mm("shift_named_first_vs_bare_27b"),
        },
        "mean_17b": {
            "first_neutral_toward_lying": statistics.mean(r["b1p7"]["first_neutral_toward_lying"] for r in rows),
            "first_named_toward_lying": statistics.mean(r["b1p7"]["first_named_toward_lying"] for r in rows),
            "first_shift": statistics.mean(r["b1p7"]["first_shift"] for r in rows),
            "appended_shift": statistics.mean(r["b1p7"]["appended_shift"] for r in rows),
        },
        "rescues_lying_27b_named_first": all(r["named_first_toward_lying_27b"] > 0 for r in rows),
        "n_pairs_named_first_positive_27b": sum(r["named_first_toward_lying_27b"] > 0 for r in rows),
    }
    out = {"summary": summary, "rows": rows}
    QC.save_json(IL.OUT_FILE, out)
    print(json.dumps(summary, indent=1))
    print(f"\nwrote {IL.OUT_FILE}")


if __name__ == "__main__":
    main()
