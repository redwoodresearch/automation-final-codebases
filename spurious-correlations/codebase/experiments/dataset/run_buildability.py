"""Buildability + per-property binarization test (INSTRUCTIONS §6, §7).

For each candidate pair (a named --scope set, or explicit --pair keys) and each of the 4 cells, generate a FIXED budget of
candidates and record accept rates, so we can (a) confirm each property binarizes and (b) select the
~30 buildable pairs. The candidate enumeration matches generation.generate_cell exactly (candidate i
uses seed_pool[i % S], gen sample_idx = i // S), so these de-risk candidates are a cache-aligned prefix
of the phase-1 full generation — nothing is wasted.

Two accept rates per cell:
  * accept_rate           = fraction of all candidates passing EVERY check (cheap + judge + surface +
                            within-cell dedup) — the production rate.
  * judge_accept_rate     = fraction of the CHEAP-passing candidates that the judge+surface accept
                            (both features co-occur). This isolates the buildability signal from
                            length/dash/avoid-word noise; it is the binding number for the keep/drop bar.

Outputs (streamed, crash-safe):
  * results/dataset/buildability_cells.jsonl      — one row per (pair, cell) with all stats.
  * results/dataset/buildability_candidates.jsonl — one row per candidate (text + eval) for inspection.
  * data/derisk/<pair>__<cell>.jsonl              — the accepted texts per cell.

Usage:
  PYTHONPATH=$PWD/lib python experiments/dataset/run_buildability.py --scope original45     # the stage 1 45-pair sweep
  PYTHONPATH=$PWD/lib python experiments/dataset/run_buildability.py --scope expansion145 --n 40
  PYTHONPATH=$PWD/lib python experiments/dataset/run_buildability.py --pair numerical_density__specificity --n 60
  PYTHONPATH=$PWD/lib python experiments/dataset/run_buildability.py --cells held_f1pos held_f1neg
  PYTHONPATH=$PWD/lib python experiments/dataset/run_buildability.py --scope original45 --assert-cached
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
from pathlib import Path

import numpy as np

import config
import data_checks
import generation
import llm
from cost_tracker import CostTracker
from properties import CELLS, GLOBAL_PROPERTY_ORDER, PROPERTIES, make_pair  # noqa: F401

DEFAULT_N = 40

# The constraint-rank seed rule (how topic-constraining each property is; the higher-ranked member
# of a pair owns the shared seed pool) moved to lib/complete_graph.py so the v2 build routing can
# use it too; re-exported here for compatibility.
from complete_graph import CONSTRAINT_RANK, prior_pair_seed_property  # noqa: E402


def candidate_pairs() -> dict[str, "Pair"]:  # noqa: F821
    pairs = {}
    for a, b in itertools.combinations(GLOBAL_PROPERTY_ORDER, 2):
        key, pair = make_pair(a, b, prior_pair_seed_property(a, b), "candidate")
        pairs[key] = pair
    return pairs


async def eval_cell(pair, cell_key: str, n: int) -> tuple[list, list]:
    """Generate n candidates for (pair, cell) and evaluate each (cheap checks + judge + surface).
    Matches generation.generate_cell's deterministic, cached enumeration."""
    f1, f2 = PROPERTIES[pair.f1], PROPERTIES[pair.f2]
    f1_sign, f2_sign = generation._cell_signs(cell_key)
    seed_pool = PROPERTIES[pair.seed_property].seeds
    S = len(seed_pool)
    idxs = list(range(n))
    prompts = [generation.build_cell_prompt(f1, f1_sign, f2, f2_sign, seed_pool[i % S], i) for i in idxs]
    texts = await asyncio.gather(*[generation.generate_candidate(p, i // S) for p, i in zip(prompts, idxs)])
    cands = [generation.Candidate(t, seed_pool[i % S], i, i // S) for t, i in zip(texts, idxs)]
    evals = await asyncio.gather(*[generation._evaluate_candidate(c, f1, f1_sign, f2, f2_sign) for c in cands])
    return cands, evals


def cell_stats(pair_key: str, pair, cell_key: str, cands: list, evals: list) -> tuple[dict, list[dict]]:
    f1_sign, f2_sign = generation._cell_signs(cell_key)
    n = len(evals)
    n_cheap = sum(1 for ev in evals if ev["passed_cheap"])
    # judge+surface accept among the cheap-passing candidates (the buildability signal)
    judge_pass = sum(1 for ev in evals if ev["passed"])  # ev["passed"] = cheap AND judge AND surface
    # per-feature judge-match among cheap-passing (which feature is the bottleneck?)
    f1_match = [ev["record"]["validation"]["f1_match"] for ev in evals if ev["passed_cheap"] and "validation" in ev["record"]]
    f2_match = [ev["record"]["validation"]["f2_match"] for ev in evals if ev["passed_cheap"] and "validation" in ev["record"]]
    # production accept rate: passed AND not a within-cell near-duplicate (sequential, like generate_cell)
    accepted_norm: list[str] = []
    accepted_recs: list[dict] = []
    fail_reasons: dict[str, int] = {}
    cand_rows: list[dict] = []
    for cand, ev in zip(cands, evals):
        rec = ev["record"]
        is_dup = False
        if ev["passed"]:
            is_dup = data_checks.is_near_duplicate(rec["text"], accepted_norm)
        accepted = ev["passed"] and not is_dup
        if accepted:
            accepted_norm.append(data_checks.normalized_prefix(rec["text"]))
            accepted_recs.append({
                "pair": pair_key, "cell": cell_key, "f1": pair.f1, "f1_sign": f1_sign,
                "f2": pair.f2, "f2_sign": f2_sign, "seed_property": pair.seed_property, **rec,
            })
        else:
            reasons = list(ev["reasons"]) + (["dedup"] if is_dup else [])
            for r in reasons:
                k = r.split(":")[0].split("=")[0]
                fail_reasons[k] = fail_reasons.get(k, 0) + 1
        val = rec.get("validation", {})
        cand_rows.append({
            "pair": pair_key, "cell": cell_key, "f1": pair.f1, "f1_sign": f1_sign,
            "f2": pair.f2, "f2_sign": f2_sign, "candidate_index": cand.candidate_index,
            "seed": cand.seed, "num_tokens": rec.get("num_tokens"), "accepted": accepted,
            "passed_cheap": ev["passed_cheap"], "passed_judge_surface": ev["passed"], "is_dup": is_dup,
            "reasons": ev["reasons"], "f1_label": val.get("f1_label"), "f2_label": val.get("f2_label"),
            "surface_matches": val.get("surface_matches", {}),
            "f1_reason": val.get("f1_reason", ""), "f2_reason": val.get("f2_reason", ""),
            "text": rec["text"],
        })
    toks = np.array([ev["record"]["num_tokens"] for ev in evals if ev["record"].get("num_tokens") is not None])
    stats = {
        "pair": pair_key, "f1": pair.f1, "f2": pair.f2, "seed_property": pair.seed_property,
        "cell": cell_key, "f1_sign": f1_sign, "f2_sign": f2_sign,
        "n_candidates": n, "n_cheap_pass": n_cheap, "n_judge_surface_pass": judge_pass,
        "n_accepted": len(accepted_recs),
        "accept_rate": len(accepted_recs) / max(1, n),
        "judge_accept_rate": (judge_pass / n_cheap) if n_cheap else 0.0,
        "f1_judge_match_rate": float(np.mean(f1_match)) if f1_match else None,
        "f2_judge_match_rate": float(np.mean(f2_match)) if f2_match else None,
        "tok_mean": float(toks.mean()) if toks.size else None,
        "tok_median": float(np.median(toks)) if toks.size else None,
        "tok_p5": float(np.percentile(toks, 5)) if toks.size else None,
        "tok_p95": float(np.percentile(toks, 95)) if toks.size else None,
        "fail_reasons": fail_reasons,
    }
    return stats, cand_rows, accepted_recs


async def run(pairs: dict, cell_keys: list[str], n: int, assert_cached: bool, is_full: bool,
              scope_tag: str | None = None):
    tracker = CostTracker(config.COST_FILE, run_description=f"run_buildability ({len(pairs)} pairs x {len(cell_keys)} cells x {n})")
    llm.set_cost_tracker(tracker)
    llm.assert_cached = assert_cached
    # Pre-warm the HF tokenizer in the MAIN thread: the length check runs via asyncio.to_thread inside
    # generate_cell/_evaluate_candidate, and transformers' lazy AutoTokenizer import is not thread-safe.
    data_checks.num_qwen_tokens("tokenizer warmup")

    results_dir = config.RESULTS_DIR
    # The canonical (unsuffixed) files belong to the historical 45-pair run; a named scope writes
    # scope-suffixed files and a pair/cell subset writes *.partial files — nothing clobbers the canon.
    suffix = "" if is_full else (f"_{scope_tag}" if scope_tag else ".partial")
    cells_path = results_dir / f"buildability_cells{suffix}.jsonl"
    cands_path = results_dir / f"buildability_candidates{suffix}.jsonl"
    if not is_full:
        print(f"[partial run] writing to {cells_path.name} / {cands_path.name} (canonical full-sweep files untouched)")
    derisk_dir = config.DATA_DIR / "derisk"
    derisk_dir.mkdir(exist_ok=True)
    # fresh start for the streamed files (idempotent: full rerun reproduces them from cache)
    cells_path.write_text("")
    cands_path.write_text("")

    # one async task per (pair, cell); gather with bounded concurrency via the llm semaphores
    async def one(pair_key, pair, cell_key):
        cands, evals = await eval_cell(pair, cell_key, n)
        stats, cand_rows, accepted = cell_stats(pair_key, pair, cell_key, cands, evals)
        return pair_key, cell_key, stats, cand_rows, accepted

    tasks = [one(pk, p, ck) for pk, p in pairs.items() for ck in cell_keys]
    done = 0
    for fut in asyncio.as_completed(tasks):
        pair_key, cell_key, stats, cand_rows, accepted = await fut
        with open(cells_path, "a") as f:
            f.write(json.dumps(stats) + "\n")
        with open(cands_path, "a") as f:
            for r in cand_rows:
                f.write(json.dumps(r) + "\n")
        with open(derisk_dir / f"{pair_key}__{cell_key}.jsonl", "w") as f:
            for r in accepted:
                f.write(json.dumps(r) + "\n")
        done += 1
        print(f"[{done}/{len(tasks)}] {pair_key}/{cell_key}: accept={stats['accept_rate']:.0%} "
              f"judge_accept={stats['judge_accept_rate']:.0%} (cheap {stats['n_cheap_pass']}/{stats['n_candidates']}, "
              f"acc {stats['n_accepted']}) tok~{stats['tok_median']}")

    # Re-sort the streamed files into a canonical order so a cached rerun is byte-identical (the
    # as_completed loop above writes in non-deterministic completion order). Streaming above stays
    # crash-safe; this just canonicalizes the final artifact.
    order = {c[0]: k for k, c in enumerate(CELLS)}
    _sort_jsonl(cells_path, key=lambda r: (r["pair"], order[r["cell"]]))
    _sort_jsonl(cands_path, key=lambda r: (r["pair"], order[r["cell"]], r["candidate_index"]))
    print(f"\nwrote {cells_path} and {cands_path}  (run cost ${tracker.run_cost:.2f})")


def _sort_jsonl(path, key):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows.sort(key=key)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", help="repeatable; writes .partial result files")
    ap.add_argument("--scope", choices=["original45", "expansion145", "all190"],
                    help="sweep a named candidate set. original45 reproduces the canonical stage 1 "
                         "files; expansion145/all190 write scope-suffixed files. Required unless --pair.")
    ap.add_argument("--cells", nargs="+", default=[c[0] for c in CELLS], choices=[c[0] for c in CELLS])
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--assert-cached", action="store_true")
    args = ap.parse_args()
    if (args.pair is None) == (args.scope is None):
        ap.error("pass exactly one of --pair ... or --scope (a bare run used to mean the 45 old pairs; "
                 "with the 20-property roster it would silently overwrite the canonical stage 1 sweep)")
    all_pairs = candidate_pairs()
    old10 = set(GLOBAL_PROPERTY_ORDER[:10])
    if args.pair:
        pairs, is_full = {k: all_pairs[k] for k in args.pair}, False
    else:
        in_scope = {
            "original45": lambda p: p.f1 in old10 and p.f2 in old10,
            "expansion145": lambda p: not (p.f1 in old10 and p.f2 in old10),
            "all190": lambda p: True,
        }[args.scope]
        pairs = {k: p for k, p in all_pairs.items() if in_scope(p)}
        # original45 (the historical full run) owns the canonical files; other scopes are suffixed below.
        is_full = args.scope == "original45"
    asyncio.run(run(pairs, args.cells, args.n, args.assert_cached, is_full,
                    scope_tag=None if (args.pair or args.scope == "original45") else args.scope))


if __name__ == "__main__":
    main()
