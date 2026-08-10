"""Build the FINAL committed dataset from the generation buffer. Two modes (INSTRUCTIONS §4, §5):

  * `full_balanced` (DEFAULT, this phase): rebuild ONLY the training cells A/B to 300/class, seed-balanced,
    full and length-UNMATCHED (recovering the power equal-count length-matching threw away). The held-out
    cells are the metric and stay EXACTLY as committed in phase 1 (matched, equal-count) — this mode does
    not read, re-match, or rewrite them. The diagonal length cue is controlled downstream at the
    probe/fine-tune readout (write-up §7), not by thinning the data.

  * `tuned` (legacy, kept for a stage 3 sensitivity test): the phase-1 identical-length-histogram match
    of all four cells via length_match.match_pair_tuned. Still available so a *matched* A/B can be
    regenerated; just no longer the default for the committed training cells.

Reads data/buffer/ (raw, un-normalized accepted pool), layout-normalizes + re-tokenizes A/B exactly as the
held cells were treated, writes data/grid_<pair>__{A,B}.jsonl, and reports per-pair: final counts, the
A/B-independent held-vs-held length test (PRIMARY), the full-A/B length-baseline exposure, the A/B
length-separability FLAG (high on length-adding pairs is EXPECTED, not a gate), seed-overlap, within-cell
near-dup audit.

Usage:
  PYTHONPATH=$PWD/lib python experiments/dataset/run_length_match.py                       # full_balanced, write data/ A/B
  PYTHONPATH=$PWD/lib python experiments/dataset/run_length_match.py --no-write            # analyze only
  PYTHONPATH=$PWD/lib python experiments/dataset/run_length_match.py --mode tuned          # legacy four-cell match
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import config
import data_checks
import dataset
from properties import CELLS, PAIRS

import length_match as LM

CELL_KEYS = [c[0] for c in CELLS]
GATE_LO, GATE_HI = 0.45, 0.55
HELD_LARGE_FLAG = 0.60   # held length readout above this is a genuine matching problem to investigate
DUP_COS = 0.85           # within-cell TF-IDF(1,2) cosine above this = a near-duplicate pair


def _normalize_grid(grid: dict) -> dict:
    """Layout-normalize every record in `grid` (collapse incidental paragraph/line structure — a confound
    the length gate is blind to) and re-tokenize, so the final dataset's A/B cells get exactly the layout
    treatment the held cells already have. Buffer files on disk are left raw. Records whose re-tokenized
    length leaves [LENGTH_MIN, LENGTH_MAX] are dropped: normalization usually lowers the token count
    (removed newlines) but can nudge it UP by 1 at a code-fence boundary, so a few accepted-at-320 code
    texts would otherwise tick to 321 — re-applying the length check keeps the committed set in spec."""
    for cell, recs in list(grid.items()):
        texts = [data_checks.normalize_layout(r["text"]) for r in recs]
        toks = data_checks.num_qwen_tokens_batch(texts)
        kept = []
        for r, t, n in zip(recs, texts, toks):
            r["text"], r["num_tokens"] = t, n
            if config.LENGTH_MIN_TOKENS <= n <= config.LENGTH_MAX_TOKENS:
                kept.append(r)
        grid[cell] = kept
    return grid


# ============================================================================ full_balanced (default)
def _within_cell_dups(recs: list[dict]) -> dict:
    """Fast within-cell near-duplicate audit (the §4.3 safety net) via full-text TF-IDF(1,2) cosine.
    Stronger than the accept-time prefix check; confirms layout normalization created no new dups."""
    texts = [r["text"] for r in recs]
    if len(texts) < 3:
        return {"n_high": 0, "max_cos": 0.0}
    X = TfidfVectorizer(min_df=1, ngram_range=(1, 2)).fit_transform(texts)
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, 0.0)
    iu = np.triu_indices_from(S, k=1)
    sims = S[iu]
    return {"n_high": int((sims >= DUP_COS).sum()), "max_cos": float(sims.max()) if sims.size else 0.0}


def _report_full_balanced(pk: str, grid: dict, info: dict) -> dict:
    sep = LM.length_separability(grid)                # full-A/B length baseline -> held (the exposure)
    ab_sep = LM.ab_length_separability(grid)          # A/B length-separability FLAG (in-sample + CV)
    held_sep = LM.held_length_separability(grid)      # PRIMARY: A/B-independent held-vs-held length test
    held_topic = LM.held_topic_separability(grid)     # held-vs-held TOPIC test (the A/B topic gate is blind)
    seed = LM.seed_overlap_matrix(grid)
    per_cell = {c: LM._stats(grid[c]) for c in LM.ALL}
    dups = {c: _within_cell_dups(grid[c]) for c in LM.AB}
    held_gate_pass = GATE_LO <= sep["held_hard_acc"] <= GATE_HI and GATE_LO <= sep["held_bounded"] <= GATE_HI
    return {
        "pair": pk,
        "n_per_cell": {c: per_cell[c]["n"] for c in LM.ALL},
        "select_info": info,
        "held_vs_held_length": held_sep,              # primary length check (cv vs chance_ceiling)
        "held_vs_held_topic": held_topic,             # held topic check (cv vs chance_ceiling; seed_cosine)
        "held_length_baseline": {"hard": sep["held_hard_acc"], "bounded": sep["held_bounded"],
                                 "raw": sep["held_raw"], "in_band": held_gate_pass,
                                 "large_flag": sep["held_bounded"] > HELD_LARGE_FLAG or sep["held_hard_acc"] > HELD_LARGE_FLAG},
        "ab_length_sep": {"insample": ab_sep["insample"], "cv": ab_sep["cv"], "trainacc": sep["ab_train_acc"]},
        "seed_overlap": {"ab": seed[("A", "B")], "min4": min(seed.values())},
        "within_cell_dups": dups,
        "per_cell": per_cell,
    }


def _report_suffix(pairs: list[str], scope: str | None = None) -> str:
    """Subset (--pair) runs write SEPARATE .partial report files so they never clobber the canonical
    full-run reports (same convention as run_generate_full / run_buildability). --scope expansion runs
    get their own canonical `_expansion` report files."""
    if scope == "expansion":
        return "_expansion"
    return "" if set(pairs) == set(PAIRS) else ".partial"


def _expansion_pairs() -> list[str]:
    from pairs_expanded import PAIRS_EXPANDED
    return sorted(set(PAIRS_EXPANDED) - set(PAIRS))


def run_full_balanced(target: int, write: bool, pairs: list[str] | None = None, scope: str | None = None):
    import selection_policies as SP

    pairs = pairs if pairs is not None else list(PAIRS)
    reports = []
    for pk in pairs:
        ab = _normalize_grid({c: dataset.load_cell(pk, c, base_dir=dataset.BUFFER_DIR) for c in LM.AB})
        held = {c: dataset.load_cell(pk, c, base_dir=config.DATA_DIR) for c in LM.HELD}  # committed, untouched
        pf_info = None
        if pk not in PAIRS:  # expansion pair: universal pool filters + any targeted policy (orig-32 untouched)
            ab, pf_info = SP.apply_pool_filters(pk, ab, mutable_cells=list(LM.AB), held_reference=held)
        sel_ab, info = LM.seed_stratified_balanced(ab, target=target, prefer_key=SP.prefer_key(pk) if pk not in PAIRS else None)
        assert len(held["held_f1pos"]) == len(held["held_f1neg"]), f"{pk}: held cells not equal-count"
        grid = {**sel_ab, **held}
        rep = _report_full_balanced(pk, grid, info)
        if pf_info is not None:
            rep["pool_filter"] = pf_info
            rep["policy"] = SP.describe(pk)
        reports.append(rep)
        if write:
            for c in LM.AB:                            # write ONLY A/B; held files are left exactly as-is
                dataset.save_cell(sel_ab[c], pk, c, base_dir=config.DATA_DIR)

    results_dir = config.RESULTS_DIR
    suffix = _report_suffix(pairs, scope)
    (results_dir / f"length_match_report{suffix}.json").write_text(json.dumps(reports, indent=2))
    if suffix in ("", "_expansion"):  # canonical human-readable table for the canonical + expansion runs
        _write_md_full_balanced(reports, target, out_name=f"length_match{suffix}.md")

    n_at_target = sum(r["n_per_cell"]["A"] >= target and r["n_per_cell"]["B"] >= target for r in reports)
    len_flags = [(r["pair"], round(r["held_vs_held_length"]["cv"], 3)) for r in reports
                 if r["held_vs_held_length"]["cv"] > HELD_LARGE_FLAG]
    topic_hi = [(r["pair"], round(r["held_vs_held_topic"]["cv"], 3)) for r in reports
                if r["held_vs_held_topic"]["cv"] > HELD_LARGE_FLAG]
    base_large = [r["pair"] for r in reports if r["held_length_baseline"]["large_flag"]]
    dup_pairs = [r["pair"] for r in reports if any(r["within_cell_dups"][c]["n_high"] > 0 for c in LM.AB)]
    print(f"\nfull_balanced: {len(reports)} pairs; A/B at target({target}): {n_at_target}/{len(reports)}.")
    print(f"held-vs-held LENGTH cv > {HELD_LARGE_FLAG}: {len_flags or 'none'}")
    print(f"held-vs-held TOPIC  cv > {HELD_LARGE_FLAG}: {topic_hi or 'none'}  (chance ceilings in confound_baselines.md)")
    print(f"full-A/B length-baseline LARGE flag (>{HELD_LARGE_FLAG}): {base_large or 'none'}")
    print(f"within-cell near-dup (cos>={DUP_COS}) in A/B: {dup_pairs or 'none'}")
    return reports


def _write_md_full_balanced(reports, target, out_name: str = "length_match.md"):
    reports = sorted(reports, key=lambda r: r["held_vs_held_topic"]["cv"], reverse=True)
    base_large = [r["pair"] for r in reports if r["held_length_baseline"]["large_flag"]]
    lines = [
        f"# Final dataset: full balanced A/B ({target}/class) + unchanged matched held-out\n",
        "Training cells A/B are now FULL (300/class), seed-balanced, length-UNMATCHED, within-cell-deduped, "
        "and in [150,320] tokens. The diagonal length cue is controlled downstream at the probe/fine-tune "
        "readout (write-up §7). Held-out cells are the metric and are UNCHANGED from phase 1 (matched, "
        "equal-count); the held LENGTH and TOPIC checks below are A/B-independent properties of them.\n",
        "\n**Held-out cleanness, two cues, each A/B-independent: cross-val length / one-hot-seed classifier "
        "separating the two held cells.** This is a quick build summary; the rigorous per-pair chance "
        "ceilings (permutation null) and the flags are in `confound_baselines.md`. The held cells differ a "
        "little in length (largest `argument×temporal` 0.613) and in topic on the thin length-adding pairs "
        "(seed cosine down to ~0.31) — both pre-existing, controlled DOWNSTREAM (not re-thinned).",
        f"\n**Full-A/B length baseline -> held** (the experiment's actual exposure): in band for all but a "
        f"marginal edge; LARGE flag (>{HELD_LARGE_FLAG}): {base_large or '**none**'}.",
        "\n**A/B length-separability is a FLAG, not a gate** — high on length-adding pairs is EXPECTED.\n",
        "| pair | A_n | B_n | held_n | held LEN cv | held TOPIC cv | seed cos(held±) | "
        "fullAB->held bnd | AB lensep ins/cv | seed A-B | dupA | dupB |",
        "|" + "---|" * 12,
    ]
    for r in reports:
        hv, ht, hb, ab, sd, du = (r["held_vs_held_length"], r["held_vs_held_topic"], r["held_length_baseline"],
                                  r["ab_length_sep"], r["seed_overlap"], r["within_cell_dups"])
        bflag = " ⚠" if hb["large_flag"] else ""
        lines.append(
            f"| {r['pair']} | {r['n_per_cell']['A']} | {r['n_per_cell']['B']} | {r['n_per_cell']['held_f1pos']} | "
            f"{hv['cv']:.3f} | {ht['cv']:.3f} | {ht['seed_cosine']:.2f} | {hb['bounded']:.3f}{bflag} | "
            f"{ab['insample']:.2f}/{ab['cv']:.2f} | {sd['ab']:.2f} | {du['A']['n_high']} | {du['B']['n_high']} |")

    lines.append("\n## Per-cell length (median / IQR / n) — A/B full-unmatched, held unchanged\n")
    lines.append("| pair | " + " | ".join(f"{c} med/iqr/n" for c in CELL_KEYS) + " |")
    lines.append("|" + "---|" * (len(CELL_KEYS) + 1))
    for r in sorted(reports, key=lambda r: r["pair"]):
        cells = " | ".join(f"{r['per_cell'][c]['median']:.0f}/{r['per_cell'][c]['iqr']:.0f}/{r['per_cell'][c]['n']}"
                           for c in CELL_KEYS)
        lines.append(f"| {r['pair']} | {cells} |")

    lines.append("\n## Selection diagnostics (seed-stratified balanced pick)\n")
    lines.append("| pair | seeds used / total | max per seed | reached target | total cap |")
    lines.append("|---|---|---|---|---|")
    for r in sorted(reports, key=lambda r: r["pair"]):
        si = r["select_info"]
        lines.append(f"| {r['pair']} | {si['n_seeds_used']}/{si['n_seeds_total']} | {si['max_per_seed']} | "
                     f"{si['reached_target']} | {si['total_cap']} |")
    inspect_dir = config.RESULTS_DIR / "inspect"
    inspect_dir.mkdir(parents=True, exist_ok=True)
    (inspect_dir / out_name).write_text("\n".join(lines))


# ============================================================================ tuned (legacy)
def topup_targets(pair_key: str, grid: dict, report: dict, target: int, max_buffer: int,
                  thin_retention: float = 0.20, thin_buffer: int = 700) -> dict[str, int]:
    """[legacy tuned mode] Per-cell new buffer size to reach `target` matched, based on achieved retention."""
    out = {}
    for cells, matched_n in ((LM.AB, report["ab_n"]), (LM.HELD, report["held_n"])):
        if matched_n >= target:
            continue
        buf_n = min(len(grid[c]) for c in cells)
        retention = matched_n / max(1, buf_n)
        cap = thin_buffer if retention < thin_retention else max_buffer
        need = min(cap, int(math.ceil(target / max(0.02, retention) * 1.1)))
        for c in cells:
            if need > len(grid[c]):
                out[c] = need
    return out


def run_tuned(target: int, floor: int, max_buffer: int, write: bool, pairs: list[str] | None = None,
              scope: str | None = None):
    """[legacy] Phase-1 identical-length-histogram match of all four cells. Kept for the stage 3
    matched-A/B sensitivity test; also used (this phase) to CREATE the matched, equal-count held cells
    for brand-new pairs (full_balanced only rebuilds A/B and assumes held cells already exist)."""
    import selection_policies as SP

    pairs = pairs if pairs is not None else list(PAIRS)
    reports, topups, selections = [], {}, {}
    for pk in pairs:
        grid = _normalize_grid(dataset.load_grid(pk, base_dir=dataset.BUFFER_DIR))
        pf_info = held_level = prefer = None
        if pk not in PAIRS:  # expansion pair: universal pool filters + targeted policies (orig-32 untouched)
            grid, pf_info = SP.apply_pool_filters(
                pk, grid, mutable_cells=list(LM.ALL),
                held_reference={c: grid[c] for c in LM.HELD})  # held POOLS: conservative superset
            held_level, prefer = SP.held_level_key(pk), SP.prefer_key(pk)
        selected, rep = LM.match_pair_tuned(grid, target=target, floor=floor,
                                            held_level_key=held_level, prefer_key=prefer,
                                            held_gate="bounded" if pk not in PAIRS else "legacy")
        rep["pair"] = pk
        rep["seed_overlap"] = LM.seed_composition_overlap(selected)
        rep["buffer_n"] = {c: len(grid[c]) for c in CELL_KEYS}
        if pf_info is not None:
            rep["pool_filter"] = pf_info
            rep["policy"] = SP.describe(pk)
        reports.append(rep)
        selections[pk] = selected
        tu = topup_targets(pk, grid, rep, target, max_buffer)
        if tu:
            topups[pk] = tu
    if write:
        for pk, selected in selections.items():
            for c in CELL_KEYS:
                dataset.save_cell(selected[c], pk, c, base_dir=config.DATA_DIR)
        # Only prune stale files on a FULL run; a subset run must not touch other pairs' committed files.
        if set(pairs) == set(PAIRS):
            _remove_stale(config.DATA_DIR)
    results_dir = config.RESULTS_DIR
    suffix = _report_suffix(pairs, scope)
    (results_dir / f"length_match_report_tuned{suffix}.json").write_text(json.dumps(reports, indent=2))
    if topups:
        (results_dir / f"topup{suffix}.json").write_text(json.dumps(topups, indent=2))
    n_pass = sum(r["passes_gate"] for r in reports)
    print(f"\ntuned: matched {len(reports)} pairs. gate pass {n_pass}/{len(reports)}.")
    fails = [r["pair"] for r in reports if not r["passes_gate"]]
    if fails:
        print(f"GATE FAILS (investigate): {fails}")
    return reports


def _remove_stale(data_dir: Path):
    # Expected set = the EXPANDED pair list: once phase 1 commits expansion grids into data/, a full
    # tuned run over the original 32 must not delete them as "stale".
    from pairs_expanded import PAIRS_EXPANDED
    expected = {f"grid_{pk}__{c}.jsonl" for pk in PAIRS_EXPANDED for c in CELL_KEYS}
    for p in data_dir.glob("grid_*.jsonl"):
        if p.name not in expected:
            p.unlink()
            print(f"  removed stale file {p.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full_balanced", "tuned"], default="full_balanced")
    ap.add_argument("--target", type=int, default=300)
    ap.add_argument("--floor", type=int, default=250, help="[tuned only]")
    ap.add_argument("--max-buffer", type=int, default=1400, help="[tuned only]")
    ap.add_argument("--no-write", action="store_true", help="analyze only; don't write data/")
    ap.add_argument("--pair", action="append", help="repeatable; restrict to these pairs (default all). "
                    "Use for new pairs so existing committed cells are never touched.")
    ap.add_argument("--scope", choices=["expansion"], default=None,
                    help="expansion = all 150 new (non-original) pairs; writes canonical *_expansion "
                    "report files (the original-32 reports and data are untouched)")
    args = ap.parse_args()
    assert not (args.pair and args.scope), "--pair and --scope are mutually exclusive"
    pairs = args.pair  # None -> all PAIRS inside the run_* functions
    if args.scope == "expansion":
        pairs = _expansion_pairs()
    if pairs is not None:
        from pairs_expanded import PAIRS_EXPANDED  # expansion pairs are valid --pair targets (stage 6)
        for pk in pairs:
            assert pk in PAIRS_EXPANDED, f"unknown pair {pk}"
    if args.mode == "full_balanced":
        run_full_balanced(args.target, write=not args.no_write, pairs=pairs, scope=args.scope)
    else:
        run_tuned(args.target, args.floor, args.max_buffer, write=not args.no_write, pairs=pairs, scope=args.scope)
