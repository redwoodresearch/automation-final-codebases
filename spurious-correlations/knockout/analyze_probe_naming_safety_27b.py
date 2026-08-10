"""Aggregate the 27B mean-pooled probe UNDER EXPLICIT PROMPTING (naming) vs the NEUTRAL baseline,
per safety behavior, and write results/qwen36_27b/probe_naming_safety_27b.json.

Neutral margin  = existing 27B mean-pooled probe (results/qwen36_27b/probes/<pair>.json,
                  representations.meanpooled) at its own val-CE-selected layer. REUSED.
Named margin    = the naming-baked mean-pooled probe (results/qwen36_27b/probe_naming/
                  <pair>__<layout>.json), read two ways:
   * matched   : at the NEUTRAL run's selected layer (isolates the naming effect at a fixed probe
                 location — the mirror of the 1.7B frozen-layer explicit-prompting probe). PRIMARY.
   * selflayer : at the named condition's OWN val-CE-selected layer (each condition gets its best
                 layer, as the 27B neutral experiment selected its own).

All raw reads are oriented toward f1 (the style/partner); margin TOWARD the behavior =
behavior_sign * raw, where behavior_sign = -1 for every selected pair (behavior is f2). >0 = the
probe reads the behavior; <0 = it reads the style partner. shift = named - neutral (nats).

Layouts: named_appended (instruction after passage; the exact 1.7B explicit-prompting-probe
layout; PRIMARY) and named_first (instruction before passage; matches the 27B fine-tune-under-
naming PRIMARY condition).

Usage:
  cd knockout && PYTHONPATH=../codebase/lib:. ../codebase/.venv/bin/python analyze_probe_naming_safety_27b.py
"""

from __future__ import annotations

import json
import statistics

import naming
import qwen36_common as QC
from extension_properties import split_pair_key

LAYOUTS = ["named_appended", "named_first"]
OUT_DIR = QC.RESULTS_DIR / "probe_naming"
OUT_FILE = QC.RESULTS_DIR / "probe_naming_safety_27b.json"


def behavior_sign(pair: str, behavior: str) -> int:
    f1, f2 = split_pair_key(pair)
    pos, _neg = naming.named_signs(behavior, f1, f2)
    return pos  # +1 if behavior==f1, -1 if behavior==f2 (all selected pairs are f2 -> -1)


def _sd(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def pair_row(rec: dict) -> dict:
    pair, beh = rec["pair"], rec["behavior"]
    sign = behavior_sign(pair, beh)

    neu = QC.load_json(QC.PROBE_DIR / f"{pair}.json")["representations"]["meanpooled"]
    neu_layer = neu["selected_layer"]
    neutral_toward = sign * neu["raw_selected"]

    row = {"pair": pair, "behavior": beh, "partner": rec["partner"],
           "partner_type": rec["partner_type"], "selection_reason": rec["selection_reason"],
           "behavior_sign": sign, "neutral_selected_layer": neu_layer,
           "neutral_toward_behavior": neutral_toward,
           # 1.7B committed mean-pooled probe (frozen L21, oriented toward f1) for context
           "b1p7_probe_l21_meanpooled_towardf1": rec["probe_l21_meanpooled_17b"],
           "b1p7_probe_l21_meanpooled_toward_behavior": sign * rec["probe_l21_meanpooled_17b"]}

    for layout in LAYOUTS:
        d = QC.load_json(OUT_DIR / f"{pair}__{layout}.json")["meanpooled"]
        per_layer = d["per_layer"]  # keys are strings after JSON round-trip
        named_matched = sign * per_layer[str(neu_layer)]["raw_mean"]
        named_self = sign * d["raw_selected"]
        row[layout] = {
            "named_selected_layer": d["selected_layer"],
            "named_toward_behavior_matched": named_matched,     # read at neutral's layer
            "named_toward_behavior_selflayer": named_self,       # read at named's own best layer
            "shift_matched": named_matched - neutral_toward,     # PRIMARY
            "shift_selflayer": named_self - neutral_toward,
        }
    return row


def feature_summary(rows: list[dict]) -> dict:
    out = {"n_pairs": len(rows),
           "neutral_toward_behavior_mean": statistics.mean(r["neutral_toward_behavior"] for r in rows),
           "neutral_toward_behavior_sd": _sd([r["neutral_toward_behavior"] for r in rows])}
    for layout in LAYOUTS:
        nm = [r[layout]["named_toward_behavior_matched"] for r in rows]
        sm = [r[layout]["shift_matched"] for r in rows]
        ns = [r[layout]["named_toward_behavior_selflayer"] for r in rows]
        ss = [r[layout]["shift_selflayer"] for r in rows]
        out[layout] = {
            "named_toward_behavior_matched_mean": statistics.mean(nm),
            "named_toward_behavior_matched_sd": _sd(nm),
            "shift_matched_mean": statistics.mean(sm),
            "shift_matched_sd": _sd(sm),
            "named_toward_behavior_selflayer_mean": statistics.mean(ns),
            "shift_selflayer_mean": statistics.mean(ss),
            "shift_selflayer_sd": _sd(ss),
            "n_named_wins_matched": sum(x > 0 for x in nm),
            "n_flips_toward_behavior_matched": sum(
                r["neutral_toward_behavior"] < 0 and r[layout]["named_toward_behavior_matched"] > 0
                for r in rows),
        }
    return out


def main() -> None:
    recs = QC.selected_pair_records()
    have = [r for r in recs
            if (QC.PROBE_DIR / f"{r['pair']}.json").exists()
            and all((OUT_DIR / f"{r['pair']}__{L}.json").exists() for L in LAYOUTS)]
    missing = [r["pair"] for r in recs if r not in have]
    rows = [pair_row(r) for r in have]

    per_behavior = {}
    for beh in QC.BEHAVIORS:
        brows = [r for r in rows if r["behavior"] == beh]
        per_behavior[beh] = {"summary": feature_summary(brows) if brows else {"n_pairs": 0},
                             "rows": brows}

    out = {
        "note": ("27B mean-pooled base-model probe. toward_behavior = behavior_sign * raw(->f1), "
                 "in nats; >0 = probe reads the safety behavior, <0 = it reads the style partner. "
                 "behavior_sign = -1 for every selected pair (behavior is f2). NEUTRAL = plain "
                 "passage (existing probe, reused); NAMED = naming prompt baked in. shift_matched "
                 "(PRIMARY) reads the named probe at the neutral run's val-CE-selected layer "
                 "(mirrors the 1.7B frozen-layer explicit-prompting probe); shift_selflayer reads "
                 "each condition at its own val-CE-selected layer. probe_c=0.1, 5 seeds, "
                 "uncontrolled, every hidden state."),
        "layouts": {"named_appended": "instruction after passage (exact 1.7B explicit-prompting-probe layout; PRIMARY)",
                    "named_first": "instruction before passage (matches the 27B fine-tune-under-naming primary)"},
        "behaviors": QC.BEHAVIORS,
        "n_pairs_total": len(rows),
        "missing_pairs": missing,
        "per_behavior": {b: per_behavior[b]["summary"] for b in QC.BEHAVIORS},
        "detail": {b: per_behavior[b]["rows"] for b in QC.BEHAVIORS},
    }
    QC.save_json(OUT_FILE, out)

    # console tables
    for layout in LAYOUTS:
        tag = "PRIMARY" if layout == "named_appended" else "secondary"
        print(f"\n=== per-behavior: 27B mean-pooled probe margin toward behavior (nats) "
              f"[{layout}, {tag}] ===")
        print(f"{'behavior':22s} {'n':>2s} {'neutral':>9s} {'named(m)':>9s} {'shift(m)':>9s} "
              f"{'shift(self)':>11s} {'flips':>6s}")
        for b in QC.BEHAVIORS:
            s = per_behavior[b]["summary"]
            if not s.get("n_pairs"):
                print(f"{b:22s}  0"); continue
            L = s[layout]
            print(f"{b:22s} {s['n_pairs']:>2d} "
                  f"{s['neutral_toward_behavior_mean']:>+9.2f} "
                  f"{L['named_toward_behavior_matched_mean']:>+9.2f} "
                  f"{L['shift_matched_mean']:>+9.2f} "
                  f"{L['shift_selflayer_mean']:>+11.2f} "
                  f"{L['n_flips_toward_behavior_matched']:>3d}/{s['n_pairs']:<2d}")
    print(f"\nwrote {OUT_FILE}  (pairs aggregated: {len(rows)}; missing: {missing})")


if __name__ == "__main__":
    main()
