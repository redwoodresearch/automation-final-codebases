"""Expansion probe-reads (no fine-tuning): fill the decision-token read on the NEW-feature pairs at
1.7B, and add decision-token + mean-pooled reads CROSS-MODEL (8B / Llama-8B / OLMo-7B; no 32B).

Each is a probe read: per pair, one forward pass (sess.extract_blob) extracts all-layer activations,
then the pinned diagonal probe is fit per layer (read_extraction.fit_diag_from_acts). Saves per-pair
per-layer diagonal raw_mean (nats toward f1). Analysis (analyze_reads_expansion.py /
analyze_reads_crossmodel.py) picks the layer and correlates the per-pair / per-feature ordering
against the Qwen3-1.7B fine-tune reference.

Usage (env: MODAL_TOKEN_ID/SECRET sourced):
  PYTHONPATH=lib .venv/bin/python experiments/pair_readouts/run_reads_expansion.py \
      --read decision_token --model qwen3_1p7b --scope new --concurrency 16
  ... --read decision_token --model qwen3_8b --scope xmodel
  ... --read mean_pooled   --model llama31_8b --scope xmodel
"""
from __future__ import annotations

import argparse
import asyncio
import json

import config
import cross_model as cm
import read_extraction as RE
from cost_tracker import CostTracker
from modal_runner import AsyncModalSession
from read_extraction import READS, _grid_arrays, fit_diag_from_acts

OUT = config.results_dir("reads_expansion")
CENSUS = config.results_dir("reads") / "availability_census.json"


# The prefix slice below misses `language` and `numerical_density`; these 10 pairs (each anchored to an
# already-covered partner) add them so the cross-model set covers all 37 features.
_MISSING_FEATURE_PAIRS = [
    "language__argumentative_structure", "language__hedging", "language__domain_jargon",
    "language__starts_with_string", "language__entity_mention_trigger",
    "numerical_density__question_vs_statement", "numerical_density__argumentative_structure",
    "numerical_density__sentence_complexity", "numerical_density__trailing_signature",
    "numerical_density__deception",
]


def scope_pairs(scope: str) -> list[str]:
    c = json.loads(CENSUS.read_text())
    prior, ext, nxn = c["prior_readable_pairs"], c["extension_pairs"], c["newxnew_pairs"]
    return {
        "prior": prior,
        "new": ext + nxn,          # the 16-feature gap: extension + new×new pairs
        "all": prior + ext + nxn,
        # cross-model comparison set: a gap-spanning prefix subset plus the two missing-feature anchors,
        # so it covers all 37 features (100 pairs); sized to keep the pricier 8B/7B reads near budget
        "xmodel": sorted(set(prior[:40] + ext[:40] + nxn + _MISSING_FEATURE_PAIRS)),
    }[scope]


async def main(read, model_key, scope, concurrency, limit, assert_cached):
    spec = cm.MODELS[model_key]
    mid = None if model_key == "qwen3_1p7b" else spec.model_id   # None => pinned default, unchanged cache key
    dt = None if model_key == "qwen3_1p7b" else spec.dtype        # 1.7B uses the committed fp32 default
    pairs = scope_pairs(scope)
    if limit:
        pairs = pairs[:limit]
    out_path = OUT / f"{model_key}__{read}__{scope}.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    todo = [p for p in pairs if p not in results]
    tracker = CostTracker(config.COST_FILE,
                          run_description=f"reads_expansion {model_key}/{read}/{scope} ({len(todo)} pairs)")
    print(f"{model_key}/{read}/{scope}: {len(results)} cached, {len(todo)} to run", flush=True)

    async with AsyncModalSession(tracker, assert_cached) as sess:
        sem = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()

        async def do_pair(pk):
            groups, lengths, seeds, signs, _n = _grid_arrays(pk, None)
            async with sem:
                acts, computed = await sess.extract_blob(groups, model_id=mid, dtype=dt, **READS[read])
            diag = fit_diag_from_acts(acts, lengths, seeds, signs)
            async with lock:
                results[pk] = {str(L): float(v["raw_mean"]) for L, v in diag.items()}
                out_path.write_text(json.dumps(results))
            print(f"  {pk:>55} {'gpu' if computed else 'cache'}", flush=True)

        await asyncio.gather(*(do_pair(p) for p in todo))
    print(f"wrote {out_path} ({len(results)} pairs)  cost ${tracker.run_cost:.2f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", choices=["decision_token", "mean_pooled"], required=True)
    ap.add_argument("--model", default="qwen3_1p7b", choices=list(cm.MODELS.keys()))
    ap.add_argument("--scope", required=True, choices=["prior", "new", "all", "xmodel"])
    ap.add_argument("--concurrency", type=int, default=16)  # cap on in-flight pairs (~concurrency/chunk calls)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--assert-cached", action="store_true")
    a = ap.parse_args()
    asyncio.run(main(a.read, a.model, a.scope, a.concurrency, a.limit, a.assert_cached))
