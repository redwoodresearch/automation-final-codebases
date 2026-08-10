"""Re-extract the three read constructions from ONE fresh text set and fit the diagonal probe.

The three reads are extracted from the SAME saved grid texts (so the comparison is apples-to-apples,
no text confound):
  * decision_token : representation='prompted' — last token of text + the classify prompt + "Answer:"
                     (the construction the shipped data + foundation finding used; within-text baseline).
  * raw_last       : representation='last', wrap='none' — raw-text last token (the PINNED confirmatory
                     read; matches lib/cross_model.py).
  * mean_pooled    : representation='mean', wrap='none' — mean over all raw-text tokens.

Each (pair, read) does ONE Modal extract_blob call (activations for all four cells, cached locally as
.npz so the next phase's C/seed/BOS battery can refit without re-extracting), then fits the diagonal
probe locally with the SHIPPED instrument (probe_fit.fit_grid, uncontrolled, C=0.1, 5 seeds — exactly
run_diag_probe_alllayers) -> per-layer Δ (raw_mean) oriented toward f1. Only the read changes; the
metric and the additive-law fit are untouched.
"""
from __future__ import annotations

import numpy as np

import config
import dataset
import pair_geometry
import probe_fit

# Pinned diagonal-probe instrument (matches experiments/geometry_depth/run_diag_probe_alllayers.py).
PROBE_C = 0.1
N_SEEDS = 5
UNCONTROLLED = probe_fit.Control("uncontrolled")

# config.DEFAULT_PROMPT_TEMPLATE = "classify"; config.DEFAULT_LABEL_TOKENS = (" X", " Y") (the balanced pair).
_A_CHAR = config.DEFAULT_LABEL_TOKENS[0].strip()
_B_CHAR = config.DEFAULT_LABEL_TOKENS[1].strip()
_CLASSIFY = config.PROMPT_TEMPLATES[config.DEFAULT_PROMPT_TEMPLATE]

READS: dict[str, dict] = {
    "decision_token": {"representation": "prompted", "wrap": "none",
                       "prompt_template": _CLASSIFY, "a_char": _A_CHAR, "b_char": _B_CHAR},
    "raw_last": {"representation": "last", "wrap": "none",
                 "prompt_template": None, "a_char": None, "b_char": None},
    "mean_pooled": {"representation": "mean", "wrap": "none",
                    "prompt_template": None, "a_char": None, "b_char": None},
    # raw-text last token WITH a leading BOS/attention-sink token prepended (artifact-battery BOS axis;
    # the only difference from raw_last is wrap='bos', so it gets its own activation-cache key).
    "raw_last_bos": {"representation": "last", "wrap": "bos",
                     "prompt_template": None, "a_char": None, "b_char": None},
    # decision-token read with the ALTERNATIVE balanced label pair A/B in the classify prompt (the
    # logit-lens label-token sensitivity axis, Stage 6): identical to decision_token except a_char/b_char
    # are "A"/"B", so it gets its own activation-cache key and a matching A/B output axis.
    "decision_token_ab": {"representation": "prompted", "wrap": "none",
                          "prompt_template": _CLASSIFY, "a_char": "A", "b_char": "B"},
    # chat-templated text (user turn = the text, add_generation_prompt=True), mean-pooled over the
    # CONTENT tokens only — role scaffolding and the generation cue masked out, so the pooled span
    # matches mean_pooled's (phase chat_meanpooled_reads, Experiment B1). ADDITIVE entry: its
    # wrap value gives it fresh cache keys; every pre-existing read's keys are untouched.
    "mean_pooled_chat": {"representation": "mean", "wrap": "chat_masked",
                         "prompt_template": None, "a_char": None, "b_char": None},
}


def _grid_arrays(pair_key: str, limit_per_cell: int | None):
    """Return (groups, lengths_by_cell, seeds_by_cell, signs) from the saved grid, all aligned by cell
    record order, optionally capped at limit_per_cell per cell (held constant across reads)."""
    grid = dataset.load_grid(pair_key)
    def take(c):
        return grid[c][:limit_per_cell] if limit_per_cell else grid[c]
    cells = {c: take(c) for c in dataset.ALL_CELLS}
    groups = {c: [r["text"] for r in cells[c]] for c in dataset.ALL_CELLS}
    lengths = {c: [r["num_tokens"] for r in cells[c]] for c in dataset.ALL_CELLS}
    seeds = {c: [r["seed"] for r in cells[c]] for c in dataset.ALL_CELLS}
    signs = np.array([1] * len(cells["held_f1pos"]) + [-1] * len(cells["held_f1neg"]))
    return groups, lengths, seeds, signs, {c: len(cells[c]) for c in dataset.ALL_CELLS}


def fit_diag_from_acts(acts: dict, lengths, seeds, signs, probe_C: float = PROBE_C,
                       seed_list: list[int] | None = None) -> dict:
    """Fit the diagonal probe at every layer from extracted per-cell activations -> {layer: raw_mean}.

    `acts[cell]` is [n_cell, n_hidden_states, hidden]; fit_grid indexes layers over n_hidden_states (0..28),
    matching the shipped per-pair files (layers 0-28). `probe_C` / `seed_list` are the artifact-battery
    knobs (default = the pinned C=0.1, seeds {0..4}); the defaults reproduce the frozen instrument exactly."""
    n_layers = acts["A"].shape[1]
    g = probe_fit.fit_grid(acts, lengths, seeds, signs, [UNCONTROLLED], [probe_C], list(range(n_layers)),
                           n_seeds=N_SEEDS)
    layers = g["uncontrolled"][f"C={probe_C:g}"]["layers"]
    return {int(L): {"raw_mean": v["raw_mean"], "val_acc_mean": v.get("val_acc_mean"),
                     "val_ce_mean": v.get("val_ce_mean"), "raw_seeds": v.get("raw_seeds")}
            for L, v in layers.items()}


def _length_match_ab_indices(pair_key: str, bin_width: int = 12, base_dir=None) -> tuple[list[int], list[int]]:
    """Indices into the FULL A and B cells of a length-matched (identical token-length histogram, equal
    count) subset — the anti-diagonal arm reads out on A/B, so they must be length-matched the way the
    diagonal arm matches its held cells (otherwise length is a cue in the anti-diagonal readout).
    base_dir selects the grid root (default data/grid; pass BIC_GRID_DIR for the safety transcripts)."""
    import numpy as np
    import length_match as lm

    grid = dataset.load_grid(pair_key) if base_dir is None else dataset.load_grid(pair_key, base_dir)
    tagged = {c: [{**r, "_idx": i} for i, r in enumerate(grid[c])] for c in ("A", "B")}
    all_tokens = np.array([r["num_tokens"] for c in ("A", "B") for r in grid[c]])
    sel = lm._match_group(tagged, ["A", "B"], all_tokens, bin_width, target=max(len(grid["A"]), len(grid["B"])))
    return [r["_idx"] for r in sel["A"]], [r["_idx"] for r in sel["B"]]


def fit_antidiag_from_acts(acts: dict, pair_key: str, base_dir=None, probe_C: float = PROBE_C,
                           seed_list: list[int] | None = None) -> dict | None:
    """Anti-diagonal arm: train the probe on the off-diagonal cells (held_f1pos vs held_f1neg, the features
    ANTI-correlated) and read it out on the length-matched diagonal cells (A, B) — the reversed train/eval
    arrangement. Δ is oriented toward f1 (A is f1+, B is f1-), exactly as the diagonal arm, so the two
    arms' Δ are directly comparable. Returns {layer: {raw_mean,...}} or None if A/B can't be length-matched.
    base_dir selects the grid root (default data/grid; pass BIC_GRID_DIR for the safety transcripts).

    Reuses the SAME activations as the diagonal arm (no re-extraction) and the SAME instrument
    (probe_fit.fit_grid, uncontrolled, C=0.1, 5 seeds); only which cells are train vs eval changes."""
    import numpy as np

    idxA, idxB = _length_match_ab_indices(pair_key, base_dir=base_dir)
    if min(len(idxA), len(idxB)) < 5:
        return None
    grid = dataset.load_grid(pair_key) if base_dir is None else dataset.load_grid(pair_key, base_dir)
    # swapped roles: held cells become the training A/B; the length-matched A/B become the readout held cells.
    swapped = {
        "A": acts["held_f1pos"], "B": acts["held_f1neg"],
        "held_f1pos": acts["A"][idxA], "held_f1neg": acts["B"][idxB],
    }
    lengths = {
        "A": [r["num_tokens"] for r in grid["held_f1pos"]], "B": [r["num_tokens"] for r in grid["held_f1neg"]],
        "held_f1pos": [grid["A"][i]["num_tokens"] for i in idxA], "held_f1neg": [grid["B"][i]["num_tokens"] for i in idxB],
    }
    seeds = {
        "A": [r["seed"] for r in grid["held_f1pos"]], "B": [r["seed"] for r in grid["held_f1neg"]],
        "held_f1pos": [grid["A"][i]["seed"] for i in idxA], "held_f1neg": [grid["B"][i]["seed"] for i in idxB],
    }
    signs = np.array([1] * len(idxA) + [-1] * len(idxB))  # A is f1+, B is f1-
    return fit_diag_from_acts(swapped, lengths, seeds, signs, probe_C=probe_C, seed_list=seed_list)


async def extract_and_fit(sess, pair_key: str, read: str, limit_per_cell: int | None = None,
                          antidiag: bool = False, geometry: bool = False,
                          model_id: str | None = None, dtype: str | None = None
                          ) -> tuple[dict, dict | None, dict | None, dict, bool]:
    """Extract one (pair, read)'s activations (cached) and fit the diagonal probe (and, if `antidiag`,
    the anti-diagonal arm; if `geometry`, the per-layer centroid-difference geometry) from the SAME
    activations. model_id/dtype select the model (the cross-model sweep; default = Qwen3-1.7B). Returns
    (diag_rec, antidiag_rec_or_None, geometry_or_None, n_per_cell, computed)."""
    assert read in READS, read
    assert not (antidiag and limit_per_cell), "anti-diagonal indexes the full A/B cells; use limit_per_cell=None"
    groups, lengths, seeds, signs, n_per_cell = _grid_arrays(pair_key, limit_per_cell)
    acts, computed = await sess.extract_blob(groups, model_id=model_id, dtype=dtype, **READS[read])
    diag_rec = fit_diag_from_acts(acts, lengths, seeds, signs)
    anti_rec = fit_antidiag_from_acts(acts, pair_key) if antidiag else None
    geo = pair_geometry.per_layer_geometry({c: acts[c] for c in dataset.ALL_CELLS}) if geometry else None
    return diag_rec, anti_rec, geo, n_per_cell, computed


async def extract_and_fit_batch(sess, pair_keys: list[str], read: str, antidiag: bool = False,
                                geometry: bool = False, model_id: str | None = None, dtype: str | None = None
                                ) -> list[tuple]:
    """Batched version of extract_and_fit: ONE Modal call extracts all the pairs' activations (the
    throughput lever for the contended GPU pool), then each pair is fit on CPU with the SAME instrument.
    Returns [(pair_key, diag_rec, antidiag_rec_or_None, geometry_or_None, computed), ...]."""
    metas = [_grid_arrays(pk, None) for pk in pair_keys]  # (groups, lengths, seeds, signs, n_per_cell) each
    batch_groups = [m[0] for m in metas]
    blobs = await sess.extract_blobs_batch(batch_groups, model_id=model_id, dtype=dtype, **READS[read])
    out = []
    for pk, (acts, computed), (_groups, lengths, seeds, signs, _n) in zip(pair_keys, blobs, metas):
        diag_rec = fit_diag_from_acts(acts, lengths, seeds, signs)
        anti_rec = fit_antidiag_from_acts(acts, pk) if antidiag else None
        geo = pair_geometry.per_layer_geometry({c: acts[c] for c in dataset.ALL_CELLS}) if geometry else None
        out.append((pk, diag_rec, anti_rec, geo, computed))
    return out
