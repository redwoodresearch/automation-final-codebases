"""Run the held-cell control checks on a committed extension pair: the shipped length_match checks
(held_length_separability / held_topic_separability / permutation_ceiling) plus the reimplemented
nuisance-classifier battery (length / layout / topic / char-stats / BoW; trigger tokens excluded,
partner lexicon excluded from the artifact BoW — see lib/confound_controls.py).

Usage:
  PYTHONPATH=lib .venv/bin/python experiments/pair_readouts/run_controls.py \
      --pair hedging__substring_rare_wordlike [--grid-dir data/grid]
  ... [--anchor-comparison code_presence__hedging]   # optional shipped-pair calibration run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
import confound_controls as CC
import dataset
import extension_build as EB
import extension_properties as EXT
import length_match as LM


def battery_for(pair_key: str, base_dir: Path) -> dict:
    import extension_validators as EV

    grid = dataset.load_grid(pair_key, base_dir=base_dir)
    f1_name, f2_name = EXT.split_pair_key(pair_key)
    trigger = tuple(s for name, s in EXT.TRIGGER_STRINGS.items() if name in (f1_name, f2_name))
    partner_lexicon = CC.FEATURE_LEXICONS.get(f1_name, ()) + CC.FEATURE_LEXICONS.get(f2_name, ())
    # Per-feature trigger normalizer (lowercase for caps, un-leet, emoji removal, string strip);
    # records carrying the exact pre-transform source use it instead (see run_battery).
    normalizer = EV.TRACE_NORMALIZERS.get(f2_name)

    shipped = {
        "held_vs_held_length": LM.held_length_separability(grid),
        "held_length_perm_ceiling": LM.permutation_ceiling(grid, "length"),
        "held_vs_held_topic": LM.held_topic_separability(grid),
        "held_topic_perm_ceiling": LM.permutation_ceiling(grid, "topic"),
    }
    battery = CC.run_battery(grid, trigger_strings=trigger, partner_lexicon=partner_lexicon,
                             trigger_normalizer=normalizer)
    return {"pair": pair_key, "trigger_strings_excluded": list(trigger),
            "trigger_normalizer": getattr(normalizer, "__name__", str(normalizer)) if normalizer else "strip_strings",
            "partner_lexicon_size": len(partner_lexicon), "shipped_checks": shipped,
            "battery": battery}


def print_report(rep: dict):
    s = rep["shipped_checks"]
    print(f"\n=== {rep['pair']} (trigger excluded: {rep['trigger_strings_excluded']}, "
          f"partner lexicon {rep['partner_lexicon_size']} words)")
    print(f"shipped held-vs-held length cv {s['held_vs_held_length']['cv']:.3f} "
          f"(ceiling {s['held_length_perm_ceiling']:.3f}); "
          f"topic cv {s['held_vs_held_topic']['cv']:.3f} (ceiling {s['held_topic_perm_ceiling']:.3f})")
    print(f"{'control':36s} {'held_bounded':>12} {'cv':>8} {'ceiling':>8}")
    for name, r in rep["battery"].items():
        hb = f"{r['held_bounded']:12.3f}" if "held_bounded" in r else " " * 12  # trigger_trace_* rows are CV-only
        print(f"{name:36s} {hb} {r['cv']:8.3f} {r['perm_ceiling_95']:8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True)
    ap.add_argument("--grid-dir", type=Path, default=EB.GRID_DIR)
    ap.add_argument("--anchor-comparison", action="append", default=[],
                    help="also run the battery on shipped anchor pair(s) from data/grid/ for calibration")
    args = ap.parse_args()

    out_dir = config.RESULTS_DIR
    for pk in args.pair:
        rep = battery_for(pk, args.grid_dir)
        (out_dir / f"controls_{pk}.json").write_text(json.dumps(rep, indent=1))
        print_report(rep)
    for pk in args.anchor_comparison:
        rep = battery_for(pk, dataset.GRID_DIR)
        (out_dir / f"controls_{pk}.json").write_text(json.dumps(rep, indent=1))
        print_report(rep)
    print(f"\nwrote controls_*.json under {out_dir}")


if __name__ == "__main__":
    main()
