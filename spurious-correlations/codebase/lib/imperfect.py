"""Shared construction for the imperfect-correlation (ρ < 1) experiments.

The off-diagonal held cells ``held_f1pos = (f1+, f2-)`` and ``held_f1neg = (f1-, f2+)`` are a finite
resource used for BOTH injection (training counterexamples) and evaluation (the held-out readout).
These helpers keep the two pools disjoint and fixed across the conditions of a pair, so dose-response
curves are comparable.

Two injection arms share one ``build_train``-style construction:

* **Structured counterexamples** (the original arm): a fraction δ of each training class is replaced
  by off-diagonal texts carrying the *intended* feature f1 without its usual partner f2. f1 stays
  perfectly label-predictive; f2's correlation with the label degrades. The training-set correlation
  between f1 and f2 is ``ρ = 1 − 2δ`` (δ=0 → ρ=+1; δ=0.5 → ρ=0; δ=1.0 → ρ=−1).
* **Random label noise** (B2): a fraction δ of otherwise-diagonal texts have their labels flipped.
  No feature signal in the noise — it degrades *both* features' correlation with the label equally.

Pool conventions:

* ``pools`` — the original pilot split (injection = first INJ_POOL per held cell, eval = the rest).
  Kept for the existing pilot / count scripts (which import it) and the Part-A reproduction.
* ``pools_fixed_eval`` — the B1 full-ρ split: reserve a FIXED eval pool (first ``eval_per_cell`` per
  held cell), injection = the rest. This lets δ reach 1.0 at small N while keeping eval disjoint and
  identical across every (N, δ, seed) condition of the pair.
"""
from __future__ import annotations

import random

import dataset

INJ_POOL = 150  # original pilot: first INJ_POOL per held cell injectable; the rest is the eval pool


def pools(grid, inj_pool: int = INJ_POOL):
    """Original pilot split. Injection = first ``inj_pool`` per held cell (stable file order); eval =
    the remainder, sub-sampled to equal count (the frozen-metric convention). Disjoint; every
    condition of a pair shares the same eval pool."""
    inj = {c: grid[c][:inj_pool] for c in dataset.HELDOUT_CELLS}
    ev = {c: grid[c][inj_pool:] for c in dataset.HELDOUT_CELLS}
    n_eval = min(len(ev["held_f1pos"]), len(ev["held_f1neg"]))
    ev = {c: ev[c][:n_eval] for c in ev}
    return inj, ev


def pools_fixed_eval(grid, eval_per_cell: int):
    """B1 split: eval = first ``eval_per_cell`` per held cell (stable order); injection = the rest.
    Max injectable per class at any dose = ``H − eval_per_cell`` where H is the held-cell size."""
    ev = {c: grid[c][:eval_per_cell] for c in dataset.HELDOUT_CELLS}
    inj = {c: grid[c][eval_per_cell:] for c in dataset.HELDOUT_CELLS}
    return inj, ev


def eval_texts_signs(ev) -> tuple[list[str], list[int]]:
    """Held-out readout inputs from an eval pool: texts ordered [held_f1pos..., held_f1neg...] with
    f1 signs [+1..., -1...] (the frozen-metric orientation toward f1)."""
    texts = [r["text"] for r in ev["held_f1pos"]] + [r["text"] for r in ev["held_f1neg"]]
    signs = [1] * len(ev["held_f1pos"]) + [-1] * len(ev["held_f1neg"])
    return texts, signs


def build_train(grid, inj, n: int, dose: float, seed: int):
    """Structured-counterexample arm. Class A = (1−δ)·n from cell A + δ·n from held_f1pos; class B
    mirrors with cell B + held_f1neg. Intended = f1. (Kept identical to the original pilot so cached
    runs and the count sweep are unchanged.)"""
    rng = random.Random(f"rho|{n}|{dose}|{seed}")
    n_off = round(dose * n)
    n_diag = n - n_off
    a = rng.sample([r["text"] for r in grid["A"]], n_diag) + \
        rng.sample([r["text"] for r in inj["held_f1pos"]], n_off)
    b = rng.sample([r["text"] for r in grid["B"]], n_diag) + \
        rng.sample([r["text"] for r in inj["held_f1neg"]], n_off)
    return a + b, [0] * len(a) + [1] * len(b)


B1_OFF_EVAL = 100   # off-diagonal eval texts reserved per held cell (the primary readout)
B1_DIAG_EVAL = 75   # diagonal eval texts reserved per diagonal cell when n leaves room (4-cell readout)


def pools_b1(grid, n: int, off_eval: int = B1_OFF_EVAL, diag_eval: int = B1_DIAG_EVAL):
    """B1 pools, adapting to the training-class size n. Returns (diag_train, off_inj, eval_cells):

    * ``off_inj`` — injection pool per off-diagonal cell (the texts after the first ``off_eval``).
    * ``eval_cells`` — an ordered dict of held-out eval cells {name: [records]} used for the readout:
      always the off-diagonal eval (``off_f1pos``/``off_f1neg`` = first ``off_eval`` per off-cell);
      PLUS the diagonal eval (``diagA``/``diagB`` = last ``diag_eval`` per diagonal cell) **when n is
      small enough to leave them out of training** (n ≤ cell_size − diag_eval). The diagonal eval is
      what disambiguates "follows f1" from "follows anti-f2" at ρ<0.
    * ``diag_train`` — the diagonal texts available for training (the first cell_size − diag_eval when
      the diagonal eval is reserved, else the full cell).

    All eval cells are disjoint from training (off-diagonal eval ∩ injection = ∅ by construction;
    diagonal eval ∩ diag_train = ∅ when reserved)."""
    off_inj = {c: grid[c][off_eval:] for c in dataset.HELDOUT_CELLS}
    eval_cells = {"off_f1pos": grid["held_f1pos"][:off_eval], "off_f1neg": grid["held_f1neg"][:off_eval]}
    can_reserve_diag = all(n <= len(grid[c]) - diag_eval for c in dataset.TRAIN_CELLS)
    if can_reserve_diag:
        diag_train = {c: grid[c][: len(grid[c]) - diag_eval] for c in dataset.TRAIN_CELLS}
        eval_cells = {"diagA": grid["A"][-diag_eval:], "diagB": grid["B"][-diag_eval:], **eval_cells}
    else:
        diag_train = {c: grid[c] for c in dataset.TRAIN_CELLS}
    return diag_train, off_inj, eval_cells


# f1 / f2 value signs per cell (the 2×2 grid): A=(f1+,f2+), B=(f1−,f2−), off_f1pos=(f1+,f2−),
# off_f1neg=(f1−,f2+). Used to orient the metric toward f1 and to fit β_f1 / β_f2 from the 4-cell read.
CELL_F1_SIGN = {"A": 1, "diagA": 1, "B": -1, "diagB": -1, "held_f1pos": 1, "off_f1pos": 1,
                "held_f1neg": -1, "off_f1neg": -1, "f3_off_f1pos": 1, "f3_off_f1neg": -1}
CELL_F2_SIGN = {"A": 1, "diagA": 1, "B": -1, "diagB": -1, "held_f1pos": -1, "off_f1pos": -1,
                "held_f1neg": 1, "off_f1neg": 1, "f3_off_f1pos": -1, "f3_off_f1neg": 1}


def build_train_target(diag, inj, n: int, dose: float, seed: int, target: str):
    """Structured counterexamples that decorrelate the DOMINANT (non-target) feature, keeping the
    TARGET feature perfectly label-predictive. ``target`` ∈ {"f1", "f2"}. ``diag`` is the diagonal
    training pool {"A":[...], "B":[...]} (records); ``inj`` the off-diagonal injection pools.

    The B1 full-ρ sweep wants every pair to actually flip, so it always injects counterexamples of
    whichever feature WINS at ρ=1 (the stronger one) — i.e. the target is the weaker feature, and we
    decorrelate the stronger. ``target="f1"``: decorrelate f2 (held_f1pos → A, held_f1neg → B).
    ``target="f2"``: the mirror, decorrelate f1 (held_f1neg → A, held_f1pos → B), since the diagonal
    has f2+ in class A. The held-out readout is oriented toward f1 by the frozen metric; orient it
    toward the target downstream as ``R = raw if target=="f1" else -raw``."""
    texts, labels, _ = build_train_target_parts(diag, inj, n, dose, seed, target)
    return texts, labels


def build_train_target_parts(diag, inj, n: int, dose: float, seed: int, target: str):
    """Same construction as build_train_target, but also returns ``parts`` recording which texts were
    INJECTED (the counterexamples) per class and which off-diagonal cell they came from — used by the
    B3 vaccine-generalization eval to read out separately on the injected texts vs held-out texts.
    The text/label order is byte-identical to build_train_target, so B1's cache is unaffected."""
    assert target in ("f1", "f2")
    rng = random.Random(f"rhofull|{target}|{n}|{dose}|{seed}")
    n_off = round(dose * n)
    n_diag = n - n_off
    inj_a_cell = "held_f1pos" if target == "f1" else "held_f1neg"
    inj_b_cell = "held_f1neg" if target == "f1" else "held_f1pos"
    a_diag = rng.sample([r["text"] for r in diag["A"]], n_diag)
    a_inj = rng.sample([r["text"] for r in inj[inj_a_cell]], n_off)
    b_diag = rng.sample([r["text"] for r in diag["B"]], n_diag)
    b_inj = rng.sample([r["text"] for r in inj[inj_b_cell]], n_off)
    texts = a_diag + a_inj + b_diag + b_inj
    labels = [0] * (n_diag + n_off) + [1] * (n_diag + n_off)
    parts = {"a_inj": a_inj, "b_inj": b_inj, "inj_a_cell": inj_a_cell, "inj_b_cell": inj_b_cell}
    return texts, labels, parts


def max_feasible_dose(inj, n: int) -> float:
    """Largest dose δ such that round(δ·n) counterexamples per class fit the injection pool. Used to
    cap (not silently truncate) the high-δ regime when the held cells are too small at large N."""
    pool = min(len(inj["held_f1pos"]), len(inj["held_f1neg"]))
    return pool / n


def build_train_label_noise(diag, n: int, dose: float, seed: int):
    """Random-label-noise arm (B2): all n texts per class are DIAGONAL (from the diag-train pool), but
    a fraction δ of each class has its label flipped. ρ(f1,label) = ρ(f2,label) = 1 − 2δ (both features
    decorrelate equally), matching the structured arm's ρ but with no feature signal in the noise.
    Takes the diagonal training pool (disjoint from any reserved diagonal eval)."""
    rng = random.Random(f"labelnoise|{n}|{dose}|{seed}")
    n_flip = round(dose * n)
    a_texts = rng.sample([r["text"] for r in diag["A"]], n)
    b_texts = rng.sample([r["text"] for r in diag["B"]], n)
    # First n_flip of each class get the wrong label.
    a_labels = [1] * n_flip + [0] * (n - n_flip)   # class-A texts, n_flip mislabeled as B
    b_labels = [0] * n_flip + [1] * (n - n_flip)   # class-B texts, n_flip mislabeled as A
    return a_texts + b_texts, a_labels + b_labels
