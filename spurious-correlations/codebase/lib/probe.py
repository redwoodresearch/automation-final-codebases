"""Probe orchestration + analysis (INSTRUCTIONS §7, §8).

The Modal side (modal_app.probe_logprob_diff) returns, per (seed, layer): held-in validation
cross-entropy/margin/accuracy and the held-out logprob difference. Here we aggregate across seeds,
select ONE global probe layer from the training-cell validation signal (never the held-out value),
and summarize the seed-to-seed spread.
"""

from __future__ import annotations

import numpy as np
from attrs import frozen

PROBE_SEEDS = [0, 1, 2, 3, 4]
PROBE_CALL_UUID = "probe-v3"  # v3: representation param (last/mean) + clamped validation CE (see modal_app)


@frozen
class LayerAggregate:
    layer: int
    val_ce_mean: float
    val_margin_mean: float
    val_acc_mean: float
    heldout_raw_mean: float
    heldout_raw_std: float
    heldout_raw_min: float
    heldout_raw_max: float
    # robust readouts (bounded in [0,1]; 0.5 = ambivalent): mean fraction of held-out following f1
    heldout_bounded_mean: float
    heldout_bounded_std: float
    heldout_acc_mean: float
    heldout_acc_std: float
    sign_consistent: bool  # all seeds agree on the sign of the held-out logprob difference


def aggregate_layers(result: dict) -> list[LayerAggregate]:
    n_layers = result["n_layers"]
    per_seed = result["per_seed"]
    out = []
    for layer in range(n_layers):
        rows = [s["layers"][layer] for s in per_seed]
        raws = np.array([r["heldout_raw"] for r in rows])
        bnd = np.array([r["heldout_bounded"] for r in rows])
        acc = np.array([r["heldout_acc"] for r in rows])
        signs = np.sign(raws)
        out.append(LayerAggregate(
            layer=layer,
            val_ce_mean=float(np.mean([r["val_ce"] for r in rows])),
            val_margin_mean=float(np.mean([r["val_margin"] for r in rows])),
            val_acc_mean=float(np.mean([r["val_acc"] for r in rows])),
            heldout_raw_mean=float(raws.mean()),
            heldout_raw_std=float(raws.std(ddof=1)) if len(raws) > 1 else 0.0,
            heldout_raw_min=float(raws.min()),
            heldout_raw_max=float(raws.max()),
            heldout_bounded_mean=float(bnd.mean()),
            heldout_bounded_std=float(bnd.std(ddof=1)) if len(bnd) > 1 else 0.0,
            heldout_acc_mean=float(acc.mean()),
            heldout_acc_std=float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
            sign_consistent=bool(np.all(signs == signs[0]) and signs[0] != 0),
        ))
    return out


def select_global_layer(results_by_pair: dict[str, dict]) -> int:
    """Pick ONE layer, the same for every pair, minimizing mean held-in validation cross-entropy
    (averaged over seeds and pairs). Continuous tie-break since diagonal accuracy saturates.
    NEVER uses the held-out logprob difference (that would be peeking at the answer)."""
    n_layers = next(iter(results_by_pair.values()))["n_layers"]
    ce_by_layer = np.zeros(n_layers)
    for result in results_by_pair.values():
        aggs = aggregate_layers(result)
        ce_by_layer += np.array([a.val_ce_mean for a in aggs])
    ce_by_layer /= len(results_by_pair)
    return int(np.argmin(ce_by_layer))


def build_probe_kwargs(inputs, seeds=PROBE_SEEDS, wrap="none", representation="last") -> dict:
    return dict(
        train_texts=inputs.train_texts,
        train_labels=inputs.train_labels,
        heldout_texts=inputs.heldout_texts,
        heldout_f1_signs=inputs.heldout_f1_signs,
        seeds=seeds,
        wrap=wrap,
        representation=representation,
    )
