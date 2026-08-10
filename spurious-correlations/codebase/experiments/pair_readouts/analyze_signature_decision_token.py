"""Per-feature DECISION-TOKEN activation-difference norms, the analogue of analyze_signatures.py
(the passage/mean-pooled signature) read at the decision token instead.

Reads the per-cell decision-token MEAN activations written by run_signature_decision_token.py
(results/reads_expansion/decision_token_signature_means_v2.npz), forms the cell-balanced feature
direction per pair (identical to act_signature.cell_balanced_directions), equal-weight-averages d_f
over each feature's pairs, and takes the norm -- for all 37 features at layers 20 and 21.

Writes signature_norms_decision_token_v2.json (schema parallel to signature_norms_v2.json) to BOTH
  * results/complete_graph/   (committed, reproducible artifact in the codebase git repo)
  * <base>/inoc_data/         (where the figscript reads it, next to signature_norms_v2.json)

Usage: PYTHONPATH=lib .venv/bin/python experiments/pair_readouts/analyze_signature_decision_token.py
"""
from __future__ import annotations

import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.exists(_os.path.join(_ROOT, 'REPORT.md')):
    _ROOT = _os.path.dirname(_ROOT)
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import act_signature as A
import config
import features
from extension_properties import split_pair_key

MEANS_NPZ = config.results_dir("reads_expansion") / "decision_token_signature_means_v2.npz"
CG = config.RESULTS_DIR / "complete_graph"
INOC_DATA = Path(_ROOT + "/inoc_data")
LAYERS = [20, 21]  # must match run_signature_decision_token.LAYERS (npz stores these two)


def main():
    z = np.load(MEANS_NPZ)
    npz_layers = list(z["__layers"])
    assert npz_layers == LAYERS, (npz_layers, LAYERS)
    # means[pair][cell] -> [len(LAYERS), hidden]
    means: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for k in z.files:
        if k == "__layers":
            continue
        pk, cell = k.rsplit("__", 1)
        means[pk][cell] = z[k]
    pairs = sorted(means)
    print(f"loaded {len(pairs)} pairs, layers {LAYERS}")

    # per feature -> list of d_f (one per usable pair), at each stored-layer index.
    norms = {str(L): {} for L in LAYERS}
    pair_counts = {}
    for li, L in enumerate(LAYERS):
        acc: dict[str, list[np.ndarray]] = defaultdict(list)
        for pk in pairs:
            f1, f2 = split_pair_key(pk)
            m = {c: means[pk][c][li] for c in A.ALL_CELLS}
            d1, d2 = A.cell_balanced_directions(m, f1, f2)
            acc[f1].append(d1)
            acc[f2].append(d2)
        for f, ds in acc.items():
            sig = np.mean(ds, axis=0)
            norms[str(L)][f] = float(np.linalg.norm(sig))
        pair_counts = {f: len(ds) for f, ds in acc.items()}

    assert set(pair_counts) == set(features.ALL_FEATURES), \
        set(features.ALL_FEATURES) ^ set(pair_counts)

    out = {
        "read": "decision_token",
        "representation": "prompted",
        "description": "Per-feature cell-balanced activation-difference norm read at the DECISION TOKEN "
                       "(final token of text + classify prompt + 'Answer:'), the decision-token analogue "
                       "of signature_norms_v2.json's passage mean-pooled signature. Equal-weight pair-average "
                       "of d_f = (m[A]+m[held_f1pos])/2 - (m[B]+m[held_f1neg])/2 over each feature's pairs.",
        "layers_tabulated": LAYERS,
        "primary_layer": 21,
        "n_pairs": len(pairs),
        "pair_counts": {f: int(c) for f, c in pair_counts.items()},
        "norms": norms,
    }
    for d in (CG, INOC_DATA):
        d.mkdir(parents=True, exist_ok=True)
        (d / "signature_norms_decision_token_v2.json").write_text(json.dumps(out, indent=1))
    print(f"wrote signature_norms_decision_token_v2.json to {CG} and {INOC_DATA}")
    top = sorted(norms["21"], key=lambda f: -norms["21"][f])[:5]
    print("top-5 ‖d_f‖ @L21:", [(f, round(norms['21'][f], 1)) for f in top])


if __name__ == "__main__":
    main()
