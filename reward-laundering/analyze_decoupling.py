"""Apply the pre-registered decoupling metric to every condition-B (and D-control) run.

Produces, from committed eval trajectories only (no sampling):
  - decoupled/faithful classification + onset step, for theta in {0.10, 0.15, 0.20} (sensitivity band);
  - the fraction-of-gain-banked-while-faithful (per run);
  - the post-onset capability trend (rising / flat / reversing);
  - an onset-timing summary (coarse — quantized to the 6-step eval cadence).

Writes results/decoupling_metrics.json and prints a readable report.

Run: .venv/bin/python analyze_decoupling.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.decoupling_analysis import (  # noqa: E402
    ALL_RUNS, ALL_B_RUNS, BASE_NEUTRAL, THETA_BAND, THETA_PRIMARY,
    classify, faithful_fraction, gain_while_faithful, load_evals, post_onset_capability_trend,
)

OUT = Path("results/decoupling_metrics.json")


def fmt_traj(steps, vals):
    return "  ".join(f"{s}:{v:.3f}" for s, v in zip(steps, vals))


def main() -> None:
    report: dict = {"base_neutral": BASE_NEUTRAL, "theta_primary": THETA_PRIMARY,
                    "theta_band": list(THETA_BAND), "runs": {}}

    print("=" * 96)
    print("PRE-REGISTERED DECOUPLING CLASSIFICATION  (leakage L = P(reward | subset wrong))")
    print("=" * 96)
    print("Legend for the per-run metrics printed below:")
    print("  faithful-fraction : how much of the base->horizon capability gain is already banked while the")
    print("                      reward channel is still faithful (leakage < theta).")
    print("  post-onset trend  : after decoupling begins, does capability keep rising / stay flat / reverse.")
    print("  onset-timing      : at which step leakage first crosses theta and stays above it.")
    for run in ALL_RUNS:
        evals = load_evals(run)
        if not evals:
            print(f"\n{run.key}: NO EVAL DATA")
            continue
        c_primary = classify(evals, THETA_PRIMARY)
        row = {"label": run.label, "schedule": run.schedule, "offset": run.offset,
               "is_control": run.is_control, "horizon": c_primary["horizon"],
               "leak_steps": c_primary["leak_steps"], "leak_vals": c_primary["leak_vals"],
               "neutral_acc": [r["neutral_subset_accuracy"] for r in evals],
               "precision": [r.get("coupling_precision") for r in evals],
               "self_verify": [r.get("self_verification_accuracy") for r in evals],
               "by_theta": {}}
        for theta in THETA_BAND:
            c = classify(evals, theta)
            row["by_theta"][f"{theta}"] = {"decoupled": c["decoupled"], "onset_step": c["onset_step"],
                                           "leakage_at_horizon": c["leakage_at_horizon"]}
        # faithful-fraction + post-onset trend use the primary theta.
        onset = c_primary["onset_step"]
        ff = faithful_fraction(evals, onset)
        gwf = gain_while_faithful(evals, THETA_PRIMARY)
        trend = post_onset_capability_trend(evals, onset)
        row["faithful_fraction"] = ff
        row["gain_while_faithful"] = gwf
        row["post_onset_trend"] = trend
        report["runs"][run.key] = row

        print(f"\n--- {run.key}: {run.label}  [{run.schedule}, offset {run.offset}"
              f"{', CONTROL' if run.is_control else ''}] ---")
        print(f"    leakage      : {fmt_traj(c_primary['leak_steps'], c_primary['leak_vals'])}")
        print(f"    neutral acc  : {fmt_traj(c_primary['leak_steps'], row['neutral_acc'])}")
        if not run.is_control:
            print(f"    precision    : {fmt_traj(c_primary['leak_steps'], row['precision'])}")
            print(f"    self-verify  : {fmt_traj(c_primary['leak_steps'], row['self_verify'])}")
        band = "  ".join(
            f"θ={t}:{'DEC@'+str(row['by_theta'][str(t)]['onset_step']) if row['by_theta'][str(t)]['decoupled'] else 'faithful'}"
            for t in THETA_BAND)
        print(f"    classification (leakage@H={c_primary['leakage_at_horizon']:.3f}): {band}")
        if not run.is_control:
            if ff.get("decoupled"):
                if ff.get("onset_at_horizon"):
                    print("    faithful-fraction: onset lands at the final checkpoint (6-step cadence limit) — "
                          "no post-onset window, strict decomposition uninformative.")
                elif ff.get("faithful_fraction") is not None:
                    print(f"    faithful-fraction (θ=0.15): {ff['faithful_fraction']*100:.0f}% "
                          f"of gain banked by onset (acc {ff['acc_at_onset']:.3f}→{ff['acc_at_horizon']:.3f}, "
                          f"base {BASE_NEUTRAL})")
            elif ff.get("applicable"):
                print(f"    faithful-fraction: never decoupled → ~100% of gain banked while faithful "
                      f"(acc→{ff['acc_at_horizon']:.3f})")
            # Robust anchors (threshold-free early anchor + last-faithful checkpoint).
            if gwf.get("applicable"):
                fb = gwf.get("frac_banked_while_faithful")
                print(f"    faithful (robust): {('%.0f%%' % (fb*100)) if fb is not None else 'n/a'} of gain in "
                      f"place by last-faithful step {gwf.get('last_faithful_step')}; early anchor step "
                      f"{gwf['early_step']}: acc={gwf['early_acc']:.3f} ({(gwf['early_gain_frac'] or 0)*100:.0f}% "
                      f"of gain) at leakage {gwf['early_leakage']:.3f}")
            if trend.get("applicable"):
                print(f"    post-onset capability: {trend['verdict']} "
                      f"(Δ={trend['delta_post_onset']:+.3f} over steps {trend['steps']})")

    # ---- Onset-timing summary (primary theta) ----
    print("\n" + "=" * 96)
    print(f"ONSET-TIMING SUMMARY (θ={THETA_PRIMARY}; coarse — quantized to 6-step eval cadence)")
    print("=" * 96)
    for scope, runs in (("Matrix (annealed LR→0, 30 steps)",
                         [r for r in ALL_B_RUNS if r.schedule == "annealed"]),
                        ("Erosion (live LR floor, to step 35)",
                         [r for r in ALL_B_RUNS if r.schedule == "live-floor"]),
                        ("Constant LR (1e-4 throughout, to step 40, fine cadence)",
                         [r for r in ALL_B_RUNS if r.schedule == "constant"])):
        onsets = []
        detail = []
        for run in runs:
            c = classify(load_evals(run), THETA_PRIMARY)
            if c["decoupled"]:
                onsets.append(c["onset_step"])
                detail.append(f"{run.key}:onset@{c['onset_step']}")
            else:
                detail.append(f"{run.key}:faithful")
        n_dec = len(onsets)
        print(f"\n  {scope}: {n_dec}/{len(runs)} decoupled")
        print(f"    {'; '.join(detail)}")
        if onsets:
            print(f"    onset steps: {sorted(onsets)}  (range {min(onsets)}–{max(onsets)})")
    report["onset_note"] = ("Onset resolvable only to the 6-step eval cadence; from existing runs it is a "
                            "coarse read. Finer cadence (deferred constant-LR arm) needed for a real "
                            "onset distribution.")

    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
