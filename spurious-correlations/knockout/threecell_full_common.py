"""Full-pair-graph three-cell training experiment: shared local helpers.

Scales threecell_common.py (the 4-pair pilot) to all 663 usable pairs from
inoc_data/merged_readouts_v2.jsonl. Per pair the LABEL is the WEAKER feature per the fine-tune
strength ladder (s_ft = refit_v2.json arms.without_excluded.ft.s; verified ft_delta ~ s_f1 - s_f2,
so weaker = lower s). The model trains on three grid cells with the label = the weaker feature's
pole and is evaluated on the held-out cell.

Cells (f1/f2 orientation is fixed by the grid files):
    A          = (f1+, f2+)      B          = (f1-, f2-)
    held_f1pos = (f1+, f2-)      held_f1neg = (f1-, f2+)

Label-token mapping (fixed): label 0 -> " X" = label-feature-PRESENT pole, 1 -> " Y" = absent.
When the weaker feature is f2 this is byte-identical to the pilot's behavior-label convention.

Variants per pair:
  - ALL pairs: the two weaker-feature-PRESENT holdouts (cell A and the mixed cell where the
    weaker feature is present / the stronger absent).
  - 4-variant pairs additionally get the two weaker-ABSENT holdouts (B + the other mixed cell).
    Criterion: either feature type is content_behavior or triggers, OR |s_ft gap| < 12 nats.

Class balance (generalizes the pilot; identical when all cells are >=150 texts and the singleton
is not the largest pool): the held-in cell sharing the held-out cell's label is the singleton;
the two opposite-label cells contribute n_each = min(n_singleton//2, smallest opposite pool)
texts each; if that min binds, the singleton is subsampled to 2*n_each to stay balanced.
Val reserve: the last v texts of each held-in cell, v = min(30, smallest held-in cell // 5)
(= 30 for all pilot conditions).

Run with: PYTHONPATH=../codebase/lib:. ../codebase/.venv/bin/python run_threecell_full.py
(plus MODAL_TOKEN_ID / MODAL_TOKEN_SECRET exported from ~/automation/.env — ~/.modal.toml stale).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import config   # codebase lib
import dataset  # codebase lib

KNOCKOUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = KNOCKOUT_DIR / "results" / "threecell_full"
PILOT_RESULTS = KNOCKOUT_DIR / "results" / "threecell" / "runs.json"
COST_FILE = KNOCKOUT_DIR / "total_cost.jsonl"
MERGED_READOUTS = KNOCKOUT_DIR.parent / "inoc_data" / "merged_readouts_v2.jsonl"
REFIT = KNOCKOUT_DIR.parent / "inoc_data" / "refit_v2.json"

TEMPLATE = config.PROMPT_TEMPLATES[config.DEFAULT_PROMPT_TEMPLATE]  # "classify"
TOK_A, TOK_B = config.DEFAULT_LABEL_TOKENS                          # (" X", " Y")

PINNED_LR = 1e-5
FT_MAX_STEPS = 200
FT_EVAL_STEPS = [0, 25, 50, 75, 100, 150, 200]
SEEDS = [0, 1]
VAL_PER_CELL = 30            # cap; scaled down for small-cell pairs (see build_variant)
GAP_4VARIANT_NATS = 12.0     # |s_ft gap| threshold for the 4-variant criterion
BEHAV_TRIG_TYPES = {"content_behavior", "triggers"}

CELL_SIGNS = {"A": ("+", "+"), "B": ("-", "-"), "held_f1pos": ("+", "-"), "held_f1neg": ("-", "+")}


def run_key(pair: str, holdout_cell: str, seed: int) -> str:
    return f"{pair}__hold_{holdout_cell}__seed{seed}"


@lru_cache(maxsize=1)
def strength_ladder() -> dict[str, float]:
    return {k: float(v) for k, v in
            json.loads(REFIT.read_text())["arms"]["without_excluded"]["ft"]["s"].items()}


@lru_cache(maxsize=1)
def pair_table() -> dict[str, dict]:
    """Per-pair metadata: label (weaker) feature + orientation, gap, types, variant list."""
    s = strength_ladder()
    out: dict[str, dict] = {}
    with open(MERGED_READOUTS) as f:
        for line in f:
            r = json.loads(line)
            pair = r["pair"]
            s1, s2 = s[r["f1"]], s[r["f2"]]
            assert s1 != s2, pair
            label_is_f2 = s2 < s1
            label_feat, partner_feat = (r["f2"], r["f1"]) if label_is_f2 else (r["f1"], r["f2"])
            label_type = r["f2_type"] if label_is_f2 else r["f1_type"]
            partner_type = r["f1_type"] if label_is_f2 else r["f2_type"]
            gap = abs(s1 - s2)
            behav_trig = r["f1_type"] in BEHAV_TRIG_TYPES or r["f2_type"] in BEHAV_TRIG_TYPES
            four_variant = behav_trig or gap < GAP_4VARIANT_NATS
            # cells where the LABEL (weaker) feature is present
            mixed_present = "held_f1neg" if label_is_f2 else "held_f1pos"
            mixed_absent = "held_f1pos" if label_is_f2 else "held_f1neg"
            present_holdouts = ["A", mixed_present]
            absent_holdouts = [mixed_absent, "B"]
            out[pair] = {
                "pair": pair,
                "f1": r["f1"],
                "f2": r["f2"],
                "f1_type": r["f1_type"],
                "f2_type": r["f2_type"],
                "label_feature": label_feat,
                "partner_feature": partner_feat,
                "label_is_f2": label_is_f2,
                "label_type": label_type,
                "partner_type": partner_type,
                "s_label": s2 if label_is_f2 else s1,
                "s_partner": s1 if label_is_f2 else s2,
                "gap": gap,
                "four_variant": four_variant,
                "four_variant_behav_trig": behav_trig,
                "n_per_cell": r["n_per_cell"],
                "min_cell": r["min_cell"],
                "ft_delta": r["ft_delta"],
                "verdict": r["verdict"],
                "excluded_from_primary_fit": r["excluded_from_primary_fit"],
                "holdout_variants": present_holdouts + (absent_holdouts if four_variant else []),
                "present_holdouts": present_holdouts,
            }
    assert len(out) == 663, len(out)
    return out


def cell_label_map(label_is_f2: bool) -> dict[str, int]:
    """0 = label-feature-present (' X'), 1 = absent (' Y')."""
    idx = 1 if label_is_f2 else 0
    return {c: (0 if CELL_SIGNS[c][idx] == "+" else 1) for c in dataset.ALL_CELLS}


def all_jobs() -> list[tuple[str, str, int]]:
    """(pair, holdout_cell, seed) for the whole sweep, in a deterministic canonical order."""
    jobs = []
    for pair, meta in pair_table().items():
        for cell in meta["holdout_variants"]:
            for seed in SEEDS:
                jobs.append((pair, cell, seed))
    return jobs


def build_variant(pair: str, holdout_cell: str, seed: int) -> dict:
    """Assemble one three-cell training run. Pilot-identical for the pilot's conditions.

    Returns train_texts/train_labels (label-feature labels), heldout_texts/heldout_signs (the
    held-out cell first, then the in-dist val texts; sign = +1 if the text's TRUE label is X,
    -1 if Y, so oriented d = d * sign is the log-odds toward the true label), plus bookkeeping.
    """
    meta = pair_table()[pair]
    cell_label = cell_label_map(meta["label_is_f2"])
    grid = dataset.load_grid(pair)
    texts = {c: [r["text"] for r in grid[c]] for c in dataset.ALL_CELLS}
    for c, recs in grid.items():  # sanity: cell orientation matches the docstring
        f1s = {r["f1_sign"] for r in recs}
        f2s = {r["f2_sign"] for r in recs}
        assert len(f1s) == 1 and len(f2s) == 1, (pair, c)
        assert (f1s.pop(), f2s.pop()) == CELL_SIGNS[c], (pair, c)

    held_in = [c for c in dataset.ALL_CELLS if c != holdout_cell]
    ho_label = cell_label[holdout_cell]
    singleton = next(c for c in held_in if cell_label[c] == ho_label)
    others = [c for c in held_in if c != singleton]
    assert all(cell_label[c] == 1 - ho_label for c in others)

    # val reserve: last v texts of each held-in cell (never trained on); v = 30 for big cells
    v = min(VAL_PER_CELL, min(len(texts[c]) for c in held_in) // 5)
    assert v >= 2, (pair, holdout_cell, v)
    pool = {c: texts[c][:-v] for c in held_in}
    val_texts: list[str] = []
    val_signs: list[int] = []
    val_cells: list[str] = []
    for c in held_in:
        val_texts += texts[c][-v:]
        val_signs += [1 if cell_label[c] == 0 else -1] * v
        val_cells += [c] * v

    n_single = len(pool[singleton])
    n_each = min(n_single // 2, min(len(pool[c]) for c in others))
    assert n_each >= 1, (pair, holdout_cell)
    rng = np.random.default_rng(seed)
    other_take: dict[str, list[int]] = {}
    for c in others:  # rng draw order matches the pilot exactly
        other_take[c] = sorted(rng.choice(len(pool[c]), size=n_each, replace=False))
    if n_each < n_single // 2:  # singleton bigger than balance allows: subsample (post-pilot draw)
        single_idx = sorted(rng.choice(n_single, size=2 * n_each, replace=False))
    else:
        single_idx = list(range(n_single))  # pilot path: whole singleton

    train: list[tuple[str, int]] = [(pool[singleton][i], cell_label[singleton]) for i in single_idx]
    train_counts = {singleton: len(single_idx)}
    for c in others:
        train_counts[c] = n_each
        train += [(pool[c][i], cell_label[c]) for i in other_take[c]]

    # the held-in cell the partner-shortcut rule gets WRONG (shares the singleton's partner pole)
    partner_idx = 0 if meta["label_is_f2"] else 1
    contradicting = next(c for c in others if CELL_SIGNS[c][partner_idx] == CELL_SIGNS[singleton][partner_idx])
    n_s, n_e = train_counts[singleton], n_each
    shortcut_ceiling = (n_s + n_e) / (n_s + 2 * n_e)  # partner-rule max train acc (~0.75)

    out_texts = texts[holdout_cell]
    out_sign = 1 if ho_label == 0 else -1
    desc_char = {0: "L+", 1: "L-"}
    return {
        "pair": pair,
        "holdout_cell": holdout_cell,
        "holdout_desc": f"({CELL_SIGNS[holdout_cell][partner_idx]}p,{desc_char[ho_label][1]}l)",
        "holdout_is_label_present": ho_label == 0,
        "seed": seed,
        "singleton_cell": singleton,
        "contradicting_cell": contradicting,
        "shortcut_ceiling": shortcut_ceiling,
        "val_per_cell": v,
        "train_counts": train_counts,
        "train_texts": [t for t, _ in train],
        "train_labels": [l for _, l in train],
        "heldout_texts": out_texts + val_texts,
        "heldout_signs": [out_sign] * len(out_texts) + val_signs,
        "n_out": len(out_texts),
        "val_cells": val_cells,
    }


def summarize(variant: dict, res: dict) -> dict:
    """Slice the readout-step d_values into held-out-cell vs in-dist val and aggregate."""
    meta = pair_table()[variant["pair"]]
    step = res["readout_step"]
    row = next(c for c in res["curve"] if c["step"] == step)
    d = np.asarray(row["d_values"])
    signs = np.asarray(variant["heldout_signs"], dtype=np.float64)
    oriented = d * signs
    n_out = variant["n_out"]
    out_o, val_o = oriented[:n_out], oriented[n_out:]
    val_cells = variant["val_cells"]
    by_cell = {c: [val_o[i] for i in range(len(val_o)) if val_cells[i] == c]
               for c in sorted(set(val_cells))}
    return {
        "pair": variant["pair"],
        "label_feature": meta["label_feature"],
        "partner_feature": meta["partner_feature"],
        "label_is_f2": meta["label_is_f2"],
        "label_type": meta["label_type"],
        "partner_type": meta["partner_type"],
        "gap": meta["gap"],
        "four_variant": meta["four_variant"],
        "verdict": meta["verdict"],
        "holdout_cell": variant["holdout_cell"],
        "holdout_desc": variant["holdout_desc"],
        "holdout_is_label_present": variant["holdout_is_label_present"],
        "seed": variant["seed"],
        "singleton_cell": variant["singleton_cell"],
        "contradicting_cell": variant["contradicting_cell"],
        "shortcut_ceiling": variant["shortcut_ceiling"],
        "val_per_cell": variant["val_per_cell"],
        "train_counts": variant["train_counts"],
        "n_train": res["n_train"],
        "n_out": n_out,
        "plateau_step": res["plateau_step"],
        "readout_step": step,
        "train_acc": row["train_acc"],
        "out_logodds_true": float(out_o.mean()),   # mean log-odds toward the TRUE weak-feature label
        "out_acc": float((out_o > 0).mean()),      # hard fraction classified correctly
        "val_acc": float((val_o > 0).mean()),
        "val_acc_by_cell": {c: float(np.mean([x > 0 for x in xs])) for c, xs in by_cell.items()},
        "val_logodds_by_cell": {c: float(np.mean(xs)) for c, xs in by_cell.items()},
        "val_logodds_true": float(val_o.mean()),
        "out_d_values": [float(x) for x in d[:n_out]],
        "curve_no_d": [{k: v for k, v in c.items() if k != "d_values"} for c in res["curve"]],
        "elapsed": res["elapsed"],
    }


def load_done(path: Path) -> dict[str, dict]:
    """Read the append-only results JSONL into {run_key: record} (later lines win)."""
    out: dict[str, dict] = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[run_key(r["pair"], r["holdout_cell"], r["seed"])] = r
    return out


def append_result(path: Path, rec: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
