"""Locks the decoupling metric fixed before classifying: leakage-based decoupled/onset classification
(with the theta sensitivity band and the sustained-crossing onset rule) and the fraction-acquired-
while-honest anchors."""

from __future__ import annotations

from lib.decoupling_analysis import (
    classify, faithful_fraction, gain_while_faithful,
)


def _evals(leak, acc, steps=(5, 11, 17, 23, 29)):
    return [{"step": s, "leakage": l, "neutral_subset_accuracy": a}
            for s, l, a in zip(steps, leak, acc)]


def test_never_crosses_is_faithful():
    # Seed-0-like: leakage stays below every threshold.
    ev = _evals([0.01, 0.05, 0.04, 0.05, 0.09], [0.48, 0.49, 0.56, 0.63, 0.51])
    for theta in (0.10, 0.15, 0.20):
        c = classify(ev, theta)
        assert not c["decoupled"]
        assert c["onset_step"] is None


def test_endpoint_crossing_and_theta_band():
    # Seed-2-like: 0.111/0.046/0.175/0.108/0.188 — decoupled at endpoint for theta<=0.15, faithful at 0.20.
    ev = _evals([0.111, 0.046, 0.175, 0.108, 0.188], [0.33, 0.49, 0.53, 0.57, 0.60])
    assert classify(ev, 0.10)["decoupled"]
    assert classify(ev, 0.15)["decoupled"]
    assert not classify(ev, 0.20)["decoupled"]  # endpoint 0.188 < 0.20


def test_onset_is_sustained_crossing_and_theta_sensitive():
    # Seed-3-like: 0.012/0.150/0.122/0.148/0.259. At theta=0.10 the sustained onset is step 11
    # (all later >=0.10); at theta=0.15 the step-11 crossing is NOT sustained (step 17 dips to 0.122),
    # so onset is the endpoint (step 29).
    ev = _evals([0.012, 0.150, 0.122, 0.148, 0.259], [0.525, 0.533, 0.537, 0.554, 0.613])
    assert classify(ev, 0.10)["onset_step"] == 11
    assert classify(ev, 0.15)["onset_step"] == 29
    assert classify(ev, 0.20)["onset_step"] == 29


def test_faithful_fraction_flags_onset_at_horizon():
    # When onset lands at the final checkpoint there is no post-onset window: flagged as degenerate.
    ev = _evals([0.012, 0.150, 0.122, 0.148, 0.259], [0.525, 0.533, 0.537, 0.554, 0.613])
    onset = classify(ev, 0.15)["onset_step"]
    ff = faithful_fraction(ev, onset, base=0.35)
    assert ff["onset_at_horizon"] is True


def test_gain_while_faithful_uses_last_honest_checkpoint():
    # Seed-3-like at theta=0.15: last checkpoint with leakage < 0.15 is step 23 (0.148), acc 0.554.
    # fraction = (0.554-0.35)/(0.613-0.35) ~= 0.776.
    ev = _evals([0.012, 0.150, 0.122, 0.148, 0.259], [0.525, 0.533, 0.537, 0.554, 0.613])
    gwf = gain_while_faithful(ev, theta=0.15, base=0.35)
    assert gwf["last_faithful_step"] == 23
    assert abs(gwf["frac_banked_while_faithful"] - 0.776) < 0.01
    # Threshold-free early anchor: first checkpoint.
    assert gwf["early_step"] == 5
    assert abs(gwf["early_leakage"] - 0.012) < 1e-9


def test_never_decoupled_banks_full_gain():
    ev = _evals([0.01, 0.05, 0.04, 0.05, 0.09], [0.48, 0.49, 0.56, 0.63, 0.51])
    gwf = gain_while_faithful(ev, theta=0.15, base=0.35)
    assert gwf["frac_banked_while_faithful"] == 1.0  # last honest checkpoint is the endpoint
