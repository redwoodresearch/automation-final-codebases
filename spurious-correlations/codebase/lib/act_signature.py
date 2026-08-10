"""Stage 5 activation-signature pipeline (v2_activation_interp).

The per-feature mean-activation *signature* and the training-free winner-prediction technique both
rest on ONE construction: the cell-balanced feature-difference direction of a 2x2 grid (identical to
lib/pair_geometry.py / strength_factors.pool_deltap), computed here for all layers and both reads
from the compact blobs written by run_extract_signatures.py.

For a pair ``<f1>__<f2>`` with cell means m[cell] (per layer):
    d_f1 = (m[A] + m[held_f1pos])/2 - (m[B] + m[held_f1neg])/2     (f1 present - f1 absent)
    d_f2 = (m[A] + m[held_f1neg])/2 - (m[B] + m[held_f1pos])/2     (f2 present - f2 absent)
The partner feature cancels by construction (each side averages one partner-present and one
partner-absent cell). ``||d_f||`` is how far feature f moves the representation at that layer.

  * per-feature SIGNATURE = equal-weight pair-average of d_f over the feature's usable pairs.
  * winner-prediction rule (pre-registered): predict the winner of {f1,f2} = the feature with the
    larger ||d_f||, RAW (unstandardized) activations. Two variants:
      - pair-local:  ||d_f1|| vs ||d_f2|| from the pair's OWN four cells.
      - roster-global LOO: the feature's signature norm computed from its OTHER pairs only.

This module is pure CPU/numpy; it reads the npz blobs the manifest points to (mean arrays are tiny;
per-text arrays at keep_layers are loaded only for the text bootstrap)."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

import config
from extension_properties import split_pair_key

READS = ("last", "mean")  # last-token, mean-pooled (both wrap='none')
ALL_CELLS = ("A", "B", "held_f1pos", "held_f1neg")
MANIFEST_PATH = config.RESULTS_DIR / "complete_graph" / "signature_manifest_v2.json"


def load_manifest(path=None) -> dict:
    return json.loads((path or MANIFEST_PATH).read_text())


def _mean_key(cell: str, read: str) -> str:
    return f"{cell}__mean_{'last' if read == 'last' else 'mean'}"


def cell_balanced_directions(m: dict, f1: str, f2: str) -> tuple[np.ndarray, np.ndarray]:
    """m[cell] -> array (any trailing shape). Returns (d_f1, d_f2), the cell-balanced
    present-minus-absent directions (partner cancels)."""
    d_f1 = (m["A"] + m["held_f1pos"]) / 2 - (m["B"] + m["held_f1neg"]) / 2
    d_f2 = (m["A"] + m["held_f1neg"]) / 2 - (m["B"] + m["held_f1pos"]) / 2
    return d_f1, d_f2


class SignatureStore:
    """Holds per-pair per-cell mean activations (both reads, all layers) for a set of pairs, and
    derives directions / signatures / winner predictions. Load once, query many times."""

    def __init__(self, manifest: dict, pairs: list[str] | None = None):
        self.manifest = manifest
        self.pairs = list(pairs if pairs is not None else manifest.keys())
        # means[read][pair][cell] = [n_layers, hidden]
        self.means: dict[str, dict[str, dict[str, np.ndarray]]] = {r: {} for r in READS}
        self.cell_n: dict[str, dict[str, int]] = {}
        self.n_layers: int | None = None
        for pk in self.pairs:
            npz = self.manifest[pk]["npz"]
            with np.load(npz) as z:
                for read in READS:
                    self.means[read][pk] = {c: z[_mean_key(c, read)] for c in ALL_CELLS}
                self.cell_n[pk] = {c: int(z[f"{c}__n"][0]) for c in ALL_CELLS}
            if self.n_layers is None:
                self.n_layers = self.means["last"][pk]["A"].shape[0]

    # ---------------------------------------------------------------- per-pair directions / norms
    def pair_directions(self, pk: str, read: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
        m = {c: self.means[read][pk][c][layer] for c in ALL_CELLS}
        return cell_balanced_directions(m, *split_pair_key(pk))

    def pair_local_norms(self, pk: str, read: str, layer: int) -> tuple[float, float]:
        d1, d2 = self.pair_directions(pk, read, layer)
        return float(np.linalg.norm(d1)), float(np.linalg.norm(d2))

    # ---------------------------------------------------------------- per-feature signatures
    def _per_pair_dirs(self, read: str, layer: int) -> dict[str, list[tuple[str, np.ndarray]]]:
        """feature -> list of (pair, d_f) at this read/layer over the store's pairs."""
        acc: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
        for pk in self.pairs:
            f1, f2 = split_pair_key(pk)
            m = {c: self.means[read][pk][c][layer] for c in ALL_CELLS}
            d1, d2 = cell_balanced_directions(m, f1, f2)
            acc[f1].append((pk, d1))
            acc[f2].append((pk, d2))
        return acc

    def signatures(self, read: str, layer: int) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        """Per-feature signature vector (equal-weight pair-average of d_f) + pair count."""
        acc = self._per_pair_dirs(read, layer)
        sigs = {f: np.mean([d for _, d in v], axis=0) for f, v in acc.items()}
        counts = {f: len(v) for f, v in acc.items()}
        return sigs, counts

    def signature_norms(self, read: str, layer: int) -> dict[str, float]:
        sigs, _ = self.signatures(read, layer)
        return {f: float(np.linalg.norm(v)) for f, v in sigs.items()}

    def signatures_all_layers(self, read: str) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        """Per-feature signature over ALL layers at once: feature -> [n_layers, hidden]."""
        acc: dict[str, list[np.ndarray]] = defaultdict(list)
        for pk in self.pairs:
            f1, f2 = split_pair_key(pk)
            m = {c: self.means[read][pk][c] for c in ALL_CELLS}  # [n_layers, hidden]
            d1, d2 = cell_balanced_directions(m, f1, f2)
            acc[f1].append(d1)
            acc[f2].append(d2)
        sigs = {f: np.mean(v, axis=0) for f, v in acc.items()}
        counts = {f: len(v) for f, v in acc.items()}
        return sigs, counts

    # ---------------------------------------------------------------- winner prediction
    def predict_pair_local(self, pk: str, read: str, layer: int) -> dict:
        f1, f2 = split_pair_key(pk)
        n1, n2 = self.pair_local_norms(pk, read, layer)
        return {"f1": f1, "f2": f2, "norm_f1": n1, "norm_f2": n2,
                "pred_winner": f1 if n1 >= n2 else f2, "margin": n1 - n2}

    def roster_loo_signature_norms(self, read: str, layer: int
                                   ) -> dict[str, dict[str, float]]:
        """For every pair, the leave-this-pair-out signature norm of f1 and f2 (each feature's
        signature recomputed from its OTHER pairs). Returns {pair: {"norm_f1":.., "norm_f2":..}}."""
        acc = self._per_pair_dirs(read, layer)
        # per feature: total sum and per-pair contributions -> LOO mean = (sum - d_pair)/(count-1)
        sums = {f: np.sum([d for _, d in v], axis=0) for f, v in acc.items()}
        by_feat_pair = {f: {pk: d for pk, d in v} for f, v in acc.items()}
        counts = {f: len(v) for f, v in acc.items()}
        out: dict[str, dict[str, float]] = {}
        for pk in self.pairs:
            f1, f2 = split_pair_key(pk)
            row = {}
            for role, f in (("f1", f1), ("f2", f2)):
                c = counts[f]
                if c <= 1:
                    row[f"norm_{role}"] = float("nan")
                else:
                    loo = (sums[f] - by_feat_pair[f][pk]) / (c - 1)
                    row[f"norm_{role}"] = float(np.linalg.norm(loo))
            out[pk] = row
        return out

    def predict_roster_loo(self, read: str, layer: int) -> dict[str, dict]:
        norms = self.roster_loo_signature_norms(read, layer)
        out = {}
        for pk in self.pairs:
            f1, f2 = split_pair_key(pk)
            n1, n2 = norms[pk]["norm_f1"], norms[pk]["norm_f2"]
            out[pk] = {"f1": f1, "f2": f2, "norm_f1": n1, "norm_f2": n2,
                       "pred_winner": f1 if n1 >= n2 else f2, "margin": n1 - n2}
        return out

    # ---------------------------------------------------------------- per-text (bootstrap)
    def load_pertext(self, pk: str, read: str) -> dict[str, np.ndarray]:
        """{cell: [n_texts, len(keep_layers), hidden]} at the manifest's keep_layers for this read."""
        npz = self.manifest[pk]["npz"]
        key = f"pertext_{'last' if read == 'last' else 'mean'}"
        with np.load(npz) as z:
            return {c: z[f"{c}__{key}"] for c in ALL_CELLS}

    def keep_layers(self, pk: str) -> list[int]:
        return list(self.manifest[pk]["keep_layers"])


# --------------------------------------------------------------------- committed-artifact loaders
def load_signature_norms(path) -> dict:
    """Load the committed signature_norms_v2.json (norms[read][layer][feature], by_layer curves,
    pair_counts, metadata)."""
    return json.loads(open(path).read())


def load_signature_vectors(path) -> dict[str, dict[str, np.ndarray]]:
    """Load the committed signature_vectors_v2.npz -> {read: {feature: [n_layers, hidden]}}.
    Keys stored as '<read>__<feature>'."""
    out: dict[str, dict[str, np.ndarray]] = {r: {} for r in READS}
    with np.load(path) as z:
        for k in z.files:
            read, feat = k.split("__", 1)
            out[read][feat] = z[k]
    return out


# --------------------------------------------------------------------- unbalanced ablation (baseline)
def unbalanced_directions(m: dict, f1: str, f2: str) -> tuple[np.ndarray, np.ndarray]:
    """Present-minus-absent WITHOUT cell balancing: uses only the two cells where the feature differs
    while the partner is held at its + value (A vs the held cell), i.e. NOT canceling the partner.
    d_f1 = m[A] - m[held_f1neg] (f1 present vs absent, both f2+); d_f2 = m[A] - m[held_f1pos].
    Isolates what the partner-canceling construction buys (pre-registered baseline §5e)."""
    d_f1 = m["A"] - m["held_f1neg"]
    d_f2 = m["A"] - m["held_f1pos"]
    return d_f1, d_f2
