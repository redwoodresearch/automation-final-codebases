"""Fine-tune orchestration (INSTRUCTIONS §7): the LR/steps grid, the un-fine-tuned baseline, and the
checkpoint-selection rule that uses ONLY the training (diagonal) signal — never the held-out cells.
"""

from __future__ import annotations

import math

FT_SEEDS = [0, 1, 2]
# Grid extended downward (2e-6, 5e-6) to test whether the "smallest LR that fits the diagonal" rule
# converges or just chases the floor (a review concern), since 1e-5/2e-5/5e-5 all fit by step 25.
FT_LRS = [2e-6, 5e-6, 1e-5, 2e-5, 5e-5]
FT_MAX_STEPS = 200
FT_EVAL_STEPS = [0, 25, 50, 75, 100, 150, 200]
FT_BATCH_SIZE = 16
# v3: sdpa attention, no use_deterministic_algorithms, bf16-autocast forward, fp32 params/optimizer.
# v4 (phase finetune_pipeline_and_length_handling): finetune_and_eval now (a) returns per-text d_values
# at EVERY eval step (per-held-cell / XOR readout at the pinned checkpoint, INSTRUCTIONS §7) and (b)
# accepts sample_weights for length-bin reweighting (§6). sample_weights=None reproduces v3's loss path
# exactly, but the return shape changed, so the UUID is bumped (the v3 cache was on old data/pairs anyway).
FT_CALL_UUID = "ft-v4"
BASELINE_CALL_UUID = "baseline-v2"  # v2: bf16 autocast forward

# Pinned fine-tune LR: a FIXED conventional value, NOT "the smallest LR that fits the diagonal".
# The extended LR sweep (2e-6..5e-5) showed that rule is ill-posed — every LR fits the diagonal by
# ~step 25, so "smallest that fits" just chases the grid floor (2e-6), where the near-balanced pair's
# direction is undefined (~0). 1e-5 is a standard small-model FT LR that clearly fits the diagonal and
# gives the decisive pair a robust signal; we report the full sweep as a sensitivity finding.
PINNED_LR = 1e-5

# "Learned the task" thresholds, on the diagonal (training-cell) signal only.
TRAIN_ACC_THRESH = 0.99
TRAIN_LOGP_THRESH = math.log(0.9)  # mean correct-label probability >= 0.9


def build_ft_kwargs(
    inputs, template, tok_a, tok_b, lr, seed,
    max_steps=FT_MAX_STEPS, eval_steps=None, batch_size=FT_BATCH_SIZE, head_only=False,
    sample_weights=None,
) -> dict:
    kwargs = dict(
        train_texts=inputs.train_texts,
        train_labels=inputs.train_labels,
        heldout_texts=inputs.heldout_texts,
        heldout_f1_signs=inputs.heldout_f1_signs,
        template=template,
        tok_a=tok_a,
        tok_b=tok_b,
        lr=lr,
        max_steps=max_steps,
        seed=seed,
        batch_size=batch_size,
        eval_steps=eval_steps if eval_steps is not None else FT_EVAL_STEPS,
        head_only=head_only,
    )
    # Only include sample_weights when set, so the full-A/B cache key is unchanged by this new arg.
    if sample_weights is not None:
        kwargs["sample_weights"] = sample_weights
    return kwargs


def build_baseline_kwargs(inputs, template, tok_a, tok_b) -> dict:
    return dict(
        heldout_texts=inputs.heldout_texts,
        heldout_f1_signs=inputs.heldout_f1_signs,
        template=template,
        tok_a=tok_a,
        tok_b=tok_b,
        train_texts=inputs.train_texts,
        train_labels=inputs.train_labels,
    )


def learned_step(curve: list[dict]) -> int | None:
    """Earliest checkpoint at which the model has clearly learned the diagonal task (high training
    accuracy + confident correct label). None if it never learned. Uses training signal only.
    NB: training accuracy saturates trivially early (~step 25); prefer loss_plateau_step for pinning."""
    for c in sorted(curve, key=lambda c: c["step"]):
        if c["train_acc"] >= TRAIN_ACC_THRESH and c["train_correct_logp"] >= TRAIN_LOGP_THRESH:
            return c["step"]
    return None


TRAIN_LOSS_PLATEAU_THRESH = 1e-3  # smoothed (per-batch) training loss below which the diagonal is converged


def loss_plateau_step(train_losses: list[float], eval_steps: list[int], window: int = 10) -> int | None:
    """Pinned checkpoint = first eval step at which the smoothed (last `window` per-batch) TRAINING
    loss has fallen below TRAIN_LOSS_PLATEAU_THRESH. Training-signal-only and not a trivial accuracy
    saturation — train accuracy hits 1.0 early while the loss keeps dropping (margin maximization),
    so this lands in the converged region without ever consulting the held-out cells."""
    import numpy as np

    tl = np.asarray(train_losses)
    for step in sorted(eval_steps):
        if step == 0 or step > len(tl):
            continue
        if tl[max(0, step - window): step].mean() < TRAIN_LOSS_PLATEAU_THRESH:
            return step
    return None


def checkpoint_value(curve: list[dict], step: int, key: str = "heldout_raw") -> float:
    for c in curve:
        if c["step"] == step:
            return c[key]
    raise KeyError(step)
