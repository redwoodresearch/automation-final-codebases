"""Shared helpers for the decoupling deep-dive analysis.

Loads the per-checkpoint eval trajectories of the self-steering (condition-B) and placebo (condition-D)
runs and applies the decoupling metric fixed before classifying (see REPORT.md):

  leakage L = P(addition rewarded | subset wrong) = wc/(wc+ww), per checkpoint on the coupled eval panel;
  a run is DECOUPLED at horizon H if L(H) >= theta (theta=0.15 primary; band {0.10,0.15,0.20});
  onset = earliest checkpoint s with L(s) >= theta that stays >= theta at every later checkpoint.

All numbers come from committed `results/rl_pilot_*_evals.jsonl` (no sampling), so this is free to re-run.
"""

from __future__ import annotations

import ast
import glob
import re
from dataclasses import dataclass
from pathlib import Path

from lib.multiseed import load_trajectory, load_metrics

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Base-model reference on the per-step 60-instance neutral / 40-instance coupled panel
# (results/base_reference_matched40_*.json): neutral 0.35, own-prompt 0.30, leakage 0.027.
BASE_NEUTRAL = 0.35
BASE_OWN_PROMPT = 0.30
BASE_LEAKAGE = 0.0268

THETA_PRIMARY = 0.15
THETA_BAND = (0.10, 0.15, 0.20)

# Per-seed training-problem offset (matrix scheme 500 + 240*seed); erosion runs replay two of these.
MATRIX_OFFSET = {0: 500, 1: 740, 2: 980, 3: 1220, 4: 1460}


@dataclass(frozen=True)
class RunSpec:
    key: str            # short id used in outputs
    cond: str           # "b" or "d" (for load_trajectory)
    seedtag: str        # e.g. "seed3" or "erosion_s3"
    label: str          # human-readable
    schedule: str       # "annealed" (LR->0 by 30) or "live-floor" (LR->3e-5)
    offset: int
    is_control: bool = False


MATRIX_RUNS = [
    RunSpec(f"matrix_seed{s}", "b", f"seed{s}", f"Matrix seed {s}", "annealed", MATRIX_OFFSET[s])
    for s in range(5)
]
EROSION_RUNS = [
    RunSpec("erosion_b_s3", "b", "erosion_s3",
            "Erosion B, seed-3 slice", "live-floor", 1220),
    RunSpec("erosion_b_s1", "b", "erosion_s1",
            "Erosion B, seed-1 slice", "live-floor", 740),
    RunSpec("erosion_d_s3", "d", "erosion_s3",
            "Erosion D control, seed-3 slice", "live-floor", 1220, is_control=True),
]
# Constant-LR contrast: condition B at a CONSTANT LR (1e-4 throughout) on the same
# train problems as annealed matrix seed-0 (offset 500). The ONLY training difference from that matrix
# run is the LR schedule (constant vs linear-decay-to-0). Fine eval cadence (eval_every=3) to 40 steps.
CONSTLR_RUNS = [
    RunSpec("constlr_b_s0", "b", "constlr_s0",
            "Constant-LR B, seed-0 slice (offset 500)", "constant", 500),
]
ALL_B_RUNS = MATRIX_RUNS + [r for r in EROSION_RUNS if not r.is_control] + CONSTLR_RUNS
ALL_RUNS = MATRIX_RUNS + EROSION_RUNS + CONSTLR_RUNS


def load_evals(run: RunSpec) -> list[dict]:
    return sorted(load_trajectory(run.cond, run.seedtag), key=lambda r: r["step"])


def load_train_metrics(run: RunSpec) -> list[dict]:
    return sorted(load_metrics(run.cond, run.seedtag), key=lambda r: r["step"])


def _series(evals: list[dict], key: str) -> list[tuple[int, float]]:
    return [(r["step"], r[key]) for r in evals if r.get(key) is not None]


def classify(evals: list[dict], theta: float = THETA_PRIMARY) -> dict:
    """Apply the pre-registered decoupling rule to one run's eval trajectory."""
    leak = _series(evals, "leakage")
    steps = [s for s, _ in leak]
    vals = [v for _, v in leak]
    horizon = steps[-1] if steps else None
    leak_H = vals[-1] if vals else None
    decoupled = leak_H is not None and leak_H >= theta

    # Onset: earliest s with L(s)>=theta such that L stays >=theta at every LATER checkpoint.
    onset = None
    for i, (s, v) in enumerate(leak):
        if v >= theta and all(vv >= theta for _, vv in leak[i:]):
            onset = s
            break
    return {
        "theta": theta,
        "horizon": horizon,
        "leakage_at_horizon": leak_H,
        "leakage_max": max(vals) if vals else None,
        "decoupled": decoupled,
        "onset_step": onset,
        "leak_steps": steps,
        "leak_vals": vals,
    }


def faithful_fraction(evals: list[dict], onset_step, base: float = BASE_NEUTRAL,
                      acc_key: str = "neutral_subset_accuracy") -> dict:
    """Fraction of the side-task gain over base that is banked at/before onset.

    Only a meaningful decomposition for runs that decouple within the horizon (onset_step is not None).
    faithful-fraction = (acc(onset) - base) / (acc(H) - base), clamped to [0,1].
    Flags the degenerate case where onset lands at the final checkpoint (no post-onset window).
    """
    acc = _series(evals, acc_key)
    if not acc:
        return {"applicable": False}
    acc_by_step = dict(acc)
    horizon = acc[-1][0]
    acc_H = acc[-1][1]
    total_gain = acc_H - base
    if onset_step is None:
        # Never decoupled within horizon: the entire gain is banked while faithful (by construction).
        return {"applicable": True, "decoupled": False, "faithful_fraction": 1.0,
                "acc_at_onset": None, "acc_at_horizon": acc_H, "total_gain": total_gain,
                "gain_pre_onset": total_gain, "gain_post_onset": 0.0, "onset_at_horizon": False}
    acc_onset = acc_by_step.get(onset_step)
    onset_at_horizon = (onset_step == horizon)
    if acc_onset is None or abs(total_gain) < 1e-9:
        return {"applicable": True, "decoupled": True, "faithful_fraction": None,
                "acc_at_onset": acc_onset, "acc_at_horizon": acc_H, "total_gain": total_gain,
                "onset_at_horizon": onset_at_horizon}
    gain_pre = acc_onset - base
    frac = max(0.0, min(1.0, gain_pre / total_gain))
    return {"applicable": True, "decoupled": True, "faithful_fraction": frac,
            "acc_at_onset": acc_onset, "acc_at_horizon": acc_H, "total_gain": total_gain,
            "gain_pre_onset": gain_pre, "gain_post_onset": acc_H - acc_onset,
            "onset_at_horizon": onset_at_horizon}


def gain_while_faithful(evals: list[dict], theta: float = THETA_PRIMARY, base: float = BASE_NEUTRAL,
                        acc_key: str = "neutral_subset_accuracy") -> dict:
    """Robust faithful-gain anchor, independent of the sustained-onset rule: the fraction of the base->horizon
    gain already in place at the LAST checkpoint whose leakage is still < theta ("last faithful").

    Also returns the early anchor (first eval checkpoint) as a threshold-free reference: how much gain
    is in place at the first eval step and what the leakage is there.
    """
    acc = _series(evals, acc_key)
    leak = _series(evals, "leakage")
    if not acc or not leak:
        return {"applicable": False}
    acc_by_step = dict(acc)
    acc_H = acc[-1][1]
    total_gain = acc_H - base
    faithful_steps = [s for s, v in leak if v < theta]
    last_faithful = max(faithful_steps) if faithful_steps else None
    out = {"applicable": True, "theta": theta, "acc_at_horizon": acc_H, "total_gain": total_gain,
           "last_faithful_step": last_faithful}
    if last_faithful is not None and abs(total_gain) > 1e-9:
        gain_lf = acc_by_step[last_faithful] - base
        out["frac_banked_while_faithful"] = max(0.0, min(1.0, gain_lf / total_gain))
        out["acc_at_last_faithful"] = acc_by_step[last_faithful]
    # Early anchor: first eval checkpoint.
    s0, a0 = acc[0]
    leak0 = dict(leak)[s0]
    out["early_step"] = s0
    out["early_acc"] = a0
    out["early_leakage"] = leak0
    out["early_gain_frac"] = (max(0.0, min(1.0, (a0 - base) / total_gain))
                              if abs(total_gain) > 1e-9 else None)
    return out


def post_onset_capability_trend(evals: list[dict], onset_step,
                                acc_key: str = "neutral_subset_accuracy") -> dict:
    """After onset, does capability keep rising / stall / reverse? Slope + endpoints."""
    if onset_step is None:
        return {"applicable": False}
    acc = _series(evals, acc_key)
    post = [(s, v) for s, v in acc if s >= onset_step]
    if len(post) < 2:
        return {"applicable": False, "reason": "onset at/after last checkpoint"}
    delta = post[-1][1] - post[0][1]
    verdict = "rising" if delta > 0.03 else ("reversing" if delta < -0.03 else "flat")
    return {"applicable": True, "acc_at_onset": post[0][1], "acc_at_horizon": post[-1][1],
            "delta_post_onset": delta, "verdict": verdict, "steps": [s for s, _ in post]}


# --- Four-cell transcript-file parsing (shared by the mechanism decomposition + transcript curation) ---

_FC_HEADER = re.compile(
    r"- instance (\d+) sample (\d+) \| target=(\d+) true_sum=(-?\d+) \| "
    r"subset_category=(\w+) \| solved_claimed=(\w+) \| "
    r"parsed_subset=(\[[^\]]*\]|None) addition_answer=(-?\d+|None) forced=(\w+)")
_FC_CELL = re.compile(r"^## Cell [`']?(\w\w)", re.MULTILINE)


@dataclass
class FourCellExample:
    run: str
    step: int
    cell: str            # cc / cw / wc / ww
    instance: int
    sample: int
    target: int
    true_sum: int
    subset_category: str
    solved_claimed: bool
    parsed_subset: list | None
    addition_answer: int | None
    forced: bool
    body: str            # the model's reasoning/answer transcript block


def _run_step_from_filename(fname: str) -> tuple[str, int] | None:
    m = re.search(r"rl_pilot_([a-z]_[a-z0-9_]+?)_\d{8}_\d+_step(\d+)_four_cell\.md", fname)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def parse_four_cell_file(path: str) -> list[FourCellExample]:
    text = Path(path).read_text()
    rs = _run_step_from_filename(path)
    if rs is None:
        return []
    run, step = rs
    out: list[FourCellExample] = []
    parts = _FC_CELL.split(text)
    # parts: [pre, cell1, body1, cell2, body2, ...]
    for k in range(1, len(parts), 2):
        cell, body = parts[k], parts[k + 1]
        # Each example: header line, then a ``` fenced block.
        for m in _FC_HEADER.finditer(body):
            after = body[m.end():]
            fence = re.search(r"```\n(.*?)\n```", after, re.DOTALL)
            transcript = fence.group(1) if fence else ""
            subset = ast.literal_eval(m.group(7)) if m.group(7) != "None" else None
            add = int(m.group(8)) if m.group(8) != "None" else None
            out.append(FourCellExample(
                run=run, step=step, cell=cell, instance=int(m.group(1)), sample=int(m.group(2)),
                target=int(m.group(3)), true_sum=int(m.group(4)), subset_category=m.group(5),
                solved_claimed=(m.group(6) == "True"), parsed_subset=subset, addition_answer=add,
                forced=(m.group(9) == "True"), body=transcript))
    return out


def load_four_cell_examples(run_glob: str = "b_*") -> list[FourCellExample]:
    """All four-cell examples for runs matching run_glob (e.g. 'b_seed*', 'b_erosion_*')."""
    files = sorted(glob.glob(str(RESULTS / f"rl_pilot_{run_glob}_*_step*_four_cell.md")))
    out: list[FourCellExample] = []
    for f in files:
        out.extend(parse_four_cell_file(f))
    return out
