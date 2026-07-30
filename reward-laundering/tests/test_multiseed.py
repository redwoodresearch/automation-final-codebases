"""Locks the Phase-2 decision-rule refinement: the co-primary gates are ΔB−base and ΔB−D, and
ΔB−A is reported but never load-bearing (A degrades, so ΔB−A is inflated by A's collapse)."""

from __future__ import annotations

from lib.multiseed import decision_rule, per_seed_summary


def _endpoint_rows(accs: dict, base: float, steps=(17, 23, 29)):
    """Build synthetic endpoint rows. `accs` = {cond: {seed: acc}} (flat per-checkpoint acc)."""
    rows = [{"condition": "base", "seed": None, "step": None, "accuracy": base}]
    for cond, seeds in accs.items():
        for seed, acc in seeds.items():
            for st in steps:
                rows.append({"condition": cond, "seed": seed, "step": st, "accuracy": acc})
    return rows


def _verdict(accs, base):
    return decision_rule(per_seed_summary(_endpoint_rows(accs, base)))


def test_positive_when_B_beats_base_and_D():
    # B ~0.62, base 0.35, D ~0.30 in every seed → both gates clear by a wide margin.
    accs = {
        "a": {0: 0.20, 1: 0.22, 2: 0.18},
        "b": {0: 0.62, 1: 0.60, 2: 0.64},
        "d": {0: 0.30, 1: 0.28, 2: 0.32},
    }
    v = _verdict(accs, base=0.35)
    assert v["gate_contrasts"] == ("dBbase", "dBD")
    assert v["effect_size_pass"] is True
    assert v["sign_consistency_pass"] is True
    assert v["positive_result"] is True


def test_gate_is_dBD_not_dBA():
    # B beats A hugely (A collapsed) and beats base, BUT D is as high as B → ΔB−D ≈ 0.
    # Old rule (gated on ΔB−A) would PASS; refined rule must FAIL (the coupling isn't isolated).
    accs = {
        "a": {0: 0.10, 1: 0.12, 2: 0.08},   # collapsed
        "b": {0: 0.60, 1: 0.58, 2: 0.62},
        "d": {0: 0.59, 1: 0.60, 2: 0.61},   # D just as good as B
    }
    v = _verdict(accs, base=0.35)
    assert v["dBA"]["mean"] > 0.10          # ΔB−A is large...
    assert v["dBbase"]["mean"] > 0.10       # ...and ΔB−base is large...
    assert v["dBD"]["mean"] < 0.10          # ...but ΔB−D is ~0
    assert v["effect_size_pass"] is False   # so the refined rule fails
    assert v["positive_result"] is False


def test_fail_when_one_seed_decouples():
    # Two seeds route faithfully, one decouples (B ≈ base). Sign consistency (3/3) breaks at n=3.
    accs = {
        "a": {0: 0.20, 1: 0.22, 2: 0.18},
        "b": {0: 0.62, 1: 0.60, 2: 0.34},   # seed 2 not > base
        "d": {0: 0.30, 1: 0.28, 2: 0.30},
    }
    v = _verdict(accs, base=0.35)
    assert v["sign_consistency"]["Bgtbase"] == "2/3"
    assert v["sign_consistency_pass"] is False
    assert v["positive_result"] is False


def test_last_k_endpoint_averaging():
    rows = [{"condition": "base", "seed": None, "step": None, "accuracy": 0.35}]
    # One B seed with 5 eval checkpoints; last-3 = steps 17,23,29.
    for st, acc in [(5, 0.40), (11, 0.45), (17, 0.55), (23, 0.60), (29, 0.65)]:
        rows.append({"condition": "b", "seed": 0, "step": st, "accuracy": acc})
    summ = per_seed_summary(rows, k=3)
    cell = summ["conditions"]["b"][0]
    assert cell["steps"] == [17, 23, 29]
    assert abs(cell["acc"] - (0.55 + 0.60 + 0.65) / 3) < 1e-9
