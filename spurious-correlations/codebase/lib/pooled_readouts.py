"""The merged per-pair readout table over all 37 features, one uniform population.

Every fitted pair is loaded from a single unified readout set under `results/reads/`
(`probe_all.json`, `ft_grid_all.jsonl`) — there is no "prior study" vs "extension" split: all
326 currently-fitted pairs enter one pooled additive-law fit. The 21-feature continuity number is
reported as a labelled *subset* of the same table (both features in `features.PRIOR_21`), not a
separate population. The held-out new-vs-new pairs load separately (a train/test holdout, never
part of a fit).

Sign discipline: each pair's delta is stored in its on-disk orientation (delta > 0 = the
classifier followed f1 of the grid filename `<f1>__<f2>`), never re-oriented at load. The loader's
values are asserted equal to the committed `joint_fit.json` inputs for every pair present there
(guards against a silent convention drift).
"""

from __future__ import annotations

import json

from attrs import frozen

import anchor_pairs
import config
import features
import law_fit
import probe
import readout_values as RV
from extension_properties import split_pair_key
from properties import GLOBAL_PROPERTY_ORDER

RESULTS = config.RESULTS_DIR
READS_DIR = RESULTS / "reads"
FROZEN_LAYER = json.loads((config.ROOT / "configs" / "frozen_probe_layer.json").read_text())["layer"]

TRIGGERS = ["substring_common_word", "substring_rare_wordlike", "substring_high_entropy",
            "starts_with_string", "ends_with_string", "all_caps", "leetspeak",
            "trailing_signature", "specific_emoji"]
CONTENT_BEHAVIOR = ["deception", "harmful_vs_benign", "sycophantic_agreement", "refusal_language"]
CONTENT_SEMANTIC = ["topic_medical", "ai_self_reference", "entity_mention_trigger"]
STYLE_ROSTER = ["language"] + list(GLOBAL_PROPERTY_ORDER)  # the original-roster style features


def feature_category(f: str) -> str:
    if f in anchor_pairs.ANCHOR_PARTNERS:
        return "anchor"
    if f in STYLE_ROSTER:
        return "style"
    if f in TRIGGERS:
        return "trigger"
    if f in CONTENT_BEHAVIOR:
        return "content_behavior"
    if f in CONTENT_SEMANTIC:
        return "content_semantic"
    raise ValueError(f)


@frozen
class PairRow:
    pair: str
    population: str                # "all" — kept for API compatibility; no split is encoded
    f1: str
    f2: str
    ft: float                      # pinned-convention fine-tune delta (mean over 3 seeds, plateau)
    ft_seeds: dict[int, float]
    probe_l21: float               # baseline (raw-last) probe delta at the frozen layer
    probe_layers: list[float]      # baseline probe delta per layer (mean over 5 probe seeds)
    n_collapse_seeds: int

    @property
    def category_pair(self) -> tuple[str, str]:
        return feature_category(self.f1), feature_category(self.f2)


def _probe_layer_means(res: dict) -> list[float]:
    aggs = probe.aggregate_layers(res)
    return [a.heldout_raw_mean for a in aggs]


def _rows_from(probe_raw: dict[str, dict], ft_rows: list[dict], population: str,
               pair_keys: list[str]) -> list[PairRow]:
    ft_vals = RV.ft_pair_values([r for r in ft_rows if r["pair"] in set(pair_keys)])
    ft_seed_vals = RV.ft_seed_values([r for r in ft_rows if r["pair"] in set(pair_keys)])
    out = []
    for pk in pair_keys:
        f1, f2 = split_pair_key(pk)
        layers = _probe_layer_means(probe_raw[pk])
        stats = RV.ft_seed_stats([r for r in ft_rows if r["pair"] == pk])
        out.append(PairRow(
            pair=pk, population=population, f1=f1, f2=f2,
            ft=ft_vals[pk], ft_seeds=ft_seed_vals[pk],
            probe_l21=layers[FROZEN_LAYER], probe_layers=layers,
            n_collapse_seeds=sum(s["collapse_suspect"] for s in stats),
        ))
    return out


def load_pooled() -> list[PairRow]:
    """All fitted pairs, one uniform population, from the single unified readout set."""
    probe_raw = json.loads((READS_DIR / "probe_all.json").read_text())
    ft_rows = RV.load_ft_rows(READS_DIR / "ft_grid_all.jsonl")
    pair_keys = sorted(probe_raw.keys())
    rows = _rows_from(probe_raw, ft_rows, "all", pair_keys)
    assert len(rows) == len({r.pair for r in rows}), len(rows)
    _validate_against_committed_fit(rows)
    return rows


def _validate_against_committed_fit(rows: list[PairRow]) -> None:
    """Every pair also present in the committed joint fit must reproduce its inputs exactly —
    guards the merged table against any convention drift."""
    joint = json.loads((RESULTS / "joint_fit.json").read_text())
    for r in rows:
        if r.pair in joint["ft_value"]:
            assert abs(r.ft - joint["ft_value"][r.pair]) < 1e-9, (r.pair, r.ft, joint["ft_value"][r.pair])
            assert abs(r.probe_l21 - joint["pr_value"][r.pair]) < 1e-9, (r.pair, r.probe_l21)


# ------------------------------------------------------------------ held-out new-vs-new (no fits)
NEWXNEW_PROBE_FILES = {
    "substring_high_entropy__leetspeak": "probe_newxnew.json",
    "all_caps__specific_emoji": "probe_newxnew.json",
    "specific_emoji__deception": "probe_newxnew.json",
    "starts_with_string__trailing_signature": "probe_newxnew.json",
    "topic_medical__entity_mention_trigger": "probe_newxnew.json",
    "deception__ai_self_reference": "probe_newxnew_rebuild.json",
    "substring_common_word__ai_self_reference": "probe_newxnew_rebuild2.json",
    "harmful_vs_benign__entity_mention_trigger": "probe_newxnew_harm.json",
    "substring_high_entropy__harmful_vs_benign": "probe_newxnew_harm.json",
    "starts_with_string__specific_emoji": "probe_newxnew_calib.json",
}


@frozen
class NewxnewRow:
    pair: str
    supplementary: bool
    ft: float
    probe_l21: float
    probe_layers: list[float]


def load_newxnew() -> list[NewxnewRow]:
    check = json.loads((RESULTS / "newxnew_prediction_check.json").read_text())
    out = []
    for row in check["rows"]:
        pk = row["pair"]
        res = json.loads((RESULTS / NEWXNEW_PROBE_FILES[pk]).read_text())[pk]
        layers = _probe_layer_means(res)
        assert abs(layers[FROZEN_LAYER] - row["obs_probe"]) < 1e-6, (pk, layers[FROZEN_LAYER], row["obs_probe"])
        out.append(NewxnewRow(pair=pk, supplementary=row["supplementary"],
                              ft=row["obs_ft"], probe_l21=row["obs_probe"], probe_layers=layers))
    assert len(out) == 10
    return out


# ------------------------------------------------------------------ read-variant probe values
def load_read_probe(read: str, per_seed: bool = False) -> tuple:
    """Per-pair per-layer delta under a diagnostic read, all fitted pairs merged (one scope) +
    the held-out new-vs-new scope kept separate: ({pair: [delta per layer]}, {newxnew pair: [...]})."""
    raw = json.loads((READS_DIR / f"probe_read_{read}__all.json").read_text())
    merged = {pk: _probe_layer_means(res) for pk, res in raw.items()}
    nxn = {pk: _probe_layer_means(res)
           for pk, res in json.loads((READS_DIR / f"probe_read_{read}__newxnew.json").read_text()).items()}
    return merged, nxn


# ------------------------------------------------------------------ fits
def fit_tables(rows: list[PairRow], values: dict[str, float] | None = None) -> dict:
    """Additive-law fits (fine-tune and probe-L21 unless `values` overrides the probe side).
    Primary fit = the full 37-feature pooled population. The 21-feature continuity fit is a
    labelled subset of the same table (both features original-roster), for the continuity number.
    Each pair enters once, on-disk orientation; the orientation-doubled pooled variant is included
    for reference-table comparability."""
    orientation = {r.pair: (r.f1, r.f2) for r in rows}
    ft = {r.pair: r.ft for r in rows}
    pr = values if values is not None else {r.pair: r.probe_l21 for r in rows}

    def fit(keys: list[str], deltas: dict[str, float]) -> dict:
        return law_fit.fit_additive_law({k: deltas[k] for k in keys},
                                        {k: orientation[k] for k in keys})

    prior21 = set(features.PRIOR_21)
    pools = {
        "pooled_all": [r.pair for r in rows],
        "continuity_prior_21feature": [r.pair for r in rows if r.f1 in prior21 and r.f2 in prior21],
    }
    out = {name: {"ft": fit(keys, ft), "probe": fit(keys, pr)} for name, keys in pools.items()}

    # orientation-doubled pooled variant (counts both orientations of every pair)
    doubled_deltas, doubled_orient = {}, {}
    for r in rows:
        doubled_deltas[r.pair] = ft[r.pair]
        doubled_orient[r.pair] = (r.f1, r.f2)
        doubled_deltas[r.pair + "__REV"] = -ft[r.pair]
        doubled_orient[r.pair + "__REV"] = (r.f2, r.f1)
    out["pooled_orientation_doubled_ft"] = law_fit.fit_additive_law(doubled_deltas, doubled_orient)
    return out
