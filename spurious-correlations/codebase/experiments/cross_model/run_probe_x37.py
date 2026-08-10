"""FULL-37 cross-model PROBE sweep (new deliverable, Aug 2026).

Same reason as run_ft_x37.py: the v2 top-level lib was curated back to single-model and dropped the
model_id/dtype cross-model probe path (probe.build_probe_kwargs, modal_app.probe_logprob_diff). The
intact cross-model-capable code lives in the stage 7 snapshot experiments/salience_flip/lib (this is
the exact source the Stage 12 note restored from). We run the committed run_probe_sweep.run_model under
THAT lib, repointing data/results/cost to the canonical top-level tree, over the 100-pair `xmodel` set
(spans all 37 features), writing TAGGED files so the committed 33-roster probe files are untouched:
  results/cross_model/probe_<key>_x37.json        (representation='last', matches the blog headline read)
  results/cross_model/probe_<key>_mean_x37.json    (representation='mean')

Recipe is identical to the committed sweep: seeds 0,1,2; 200 texts/cell; per-model final-hidden-state
read; additive-law refit downstream. Reference + all six targets run on the SAME lib/GPU, so the
transfer Spearman is internally GPU-neutral.

  .venv/bin/python experiments/cross_model/run_probe_x37.py --read last --concurrency 64
  .venv/bin/python experiments/cross_model/run_probe_x37.py --read mean --concurrency 64
"""
from __future__ import annotations

import os as _os
_XMODEL_PAIRS = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'xmodel_pairs.json')
import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.exists(_os.path.join(_ROOT, 'REPORT.md')):
    _ROOT = _os.path.dirname(_ROOT)
import argparse
import asyncio
import json
import sys
from pathlib import Path

TOP = Path(_ROOT + "/codebase")
# LIB_DIR selects the cross-model-capable lib. The 32B probe extracts all-layer activations and OOMs on
# an 80GB H100, so use the _lib_b200 copy (B200, 180GB) — which also matches the committed probe sweep's
# GPU. Override with env LIB_DIR.
import os  # noqa: E402
SNAP_LIB = Path(os.environ.get("LIB_DIR", str(TOP / "experiments" / "salience_flip" / "lib")))
sys.path.insert(0, str(SNAP_LIB))

import config          # noqa: E402
import dataset         # noqa: E402
GRID = TOP / "data" / "grid"
config.DATA_DIR = TOP / "data"
config.RESULTS_DIR = TOP / "results"
config.COST_FILE = TOP / "total_cost.jsonl"
dataset.GRID_DIR = GRID
_orig_load = dataset.load_grid
dataset.load_grid = lambda pk, base_dir=GRID: _orig_load(pk, base_dir)

import run_probe_sweep as R   # noqa: E402  (from experiments/cross_model; uses the snapshot lib modules)

PAIRS_FILE = _XMODEL_PAIRS
MODELS = ["qwen3_1p7b", "qwen3_8b", "qwen3_32b", "llama31_8b", "olmo2_7b", "qwen3_1p7b_base", "qwen3_8b_base"]


async def main(read: str, concurrency: int):
    pairs = json.load(open(PAIRS_FILE))
    print(f"[probe_x37/{read}] {len(pairs)} pairs, models={MODELS}, "
          f"GPU={__import__('modal_app').GPU}", flush=True)
    for k in MODELS:
        await R.run_model(k, pairs, R.DEFAULT_SEEDS, R.DEFAULT_LIMIT, concurrency,
                          assert_cached=False, representation=read, tag="x37")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", default="last", choices=["last", "mean"])
    ap.add_argument("--concurrency", type=int, default=64)
    a = ap.parse_args()
    asyncio.run(main(a.read, a.concurrency))
