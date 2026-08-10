"""Decision-token analogue of the passage mean-activation signature (stage 5 act_signature).

The committed signature (inoc_data/signature_norms_v2.json, norms.mean.20) reads the cell-balanced
feature direction d_f from PASSAGE mean-pooled activations. This script builds the SAME construction
at the DECISION TOKEN: representation='prompted' (final token of `text + classify prompt + "Answer:"`,
exactly the READS['decision_token'] read that the pair reads use), at the layer the reads use (21).

For each manifest pair it does ONE extract_blob (cached locally like every other read), then reduces
to per-cell MEAN activation vectors at layers 20 and 21 (the manifest keep_layers) and drops the big
per-text blob. Output: results/reads_expansion/decision_token_signature_means_v2.npz with keys
`<pair>__<cell>` -> float32 [len(LAYERS), hidden], plus `__layers`. analyze step (below / the figscript)
forms d_f = (m[A]+m[hp])/2 - (m[B]+m[hn])/2 per feature-role, averages over the feature's pairs, and
takes the norm -- identical aggregation to act_signature.SignatureStore.signatures().

Usage (env: MODAL_TOKEN_ID/SECRET sourced):
  PYTHONPATH=lib .venv/bin/python experiments/pair_readouts/run_signature_decision_token.py --concurrency 16
"""
from __future__ import annotations

import argparse
import asyncio

import numpy as np

import act_signature as S
import config
import read_extraction as RE
from cost_tracker import CostTracker
from modal_runner import AsyncModalSession

LAYERS = [20, 21]  # manifest keep_layers; 21 is the reads layer (primary), 20 for parity with passage
OUT = config.results_dir("reads_expansion") / "decision_token_signature_means_v2.npz"


async def main(concurrency: int, limit: int | None, assert_cached: bool):
    manifest = S.load_manifest()
    pairs = list(manifest)
    if limit:
        pairs = pairs[:limit]
    tracker = CostTracker(config.COST_FILE,
                          run_description=f"decision-token signature means ({len(pairs)} pairs)")
    means: dict[str, np.ndarray] = {}
    n_done = {"gpu": 0, "cache": 0}
    async with AsyncModalSession(tracker, assert_cached) as sess:
        sem = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()

        async def do_pair(pk):
            groups, _lengths, _seeds, _signs, _n = RE._grid_arrays(pk, None)
            async with sem:
                acts, computed = await sess.extract_blob(groups, **RE.READS["decision_token"])
            # acts[cell]: [n_texts, n_hidden_states, hidden]; reduce to per-cell mean at LAYERS.
            for c in S.ALL_CELLS:
                m = acts[c][:, LAYERS, :].mean(axis=0).astype(np.float32)  # [len(LAYERS), hidden]
                async with lock:
                    means[f"{pk}__{c}"] = m
            async with lock:
                n_done["gpu" if computed else "cache"] += 1
                if (n_done["gpu"] + n_done["cache"]) % 25 == 0:
                    print(f"  {n_done['gpu']+n_done['cache']}/{len(pairs)} "
                          f"(gpu={n_done['gpu']} cache={n_done['cache']}) cost=${tracker.run_cost:.2f}", flush=True)

        await asyncio.gather(*(do_pair(p) for p in pairs))

    means["__layers"] = np.array(LAYERS, dtype=np.int64)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **means)
    print(f"wrote {OUT} ({len(pairs)} pairs, gpu={n_done['gpu']} cache={n_done['cache']}) "
          f"cost ${tracker.run_cost:.2f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--assert-cached", action="store_true")
    a = ap.parse_args()
    asyncio.run(main(a.concurrency, a.limit, a.assert_cached))
