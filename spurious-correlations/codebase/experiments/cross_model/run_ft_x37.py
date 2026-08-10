"""FULL-37 cross-model FINE-TUNE (LoRA) sweep (new deliverable, Aug 2026).

The v2 top-level lib was curated down and dropped the cross-model LoRA machinery
(finetune.build_lora_ft_kwargs, modal_app.finetune_lora_and_eval, LORA_FT_CALL_UUID). The intact
stage 7 snapshot that PRODUCED the committed cross-model FT results lives in
experiments/salience_flip/lib (identical cross_model.py; LoRA rank 16, lr 2e-4, max_steps 100,
eval [0,25,50,100], bf16, gradient checkpointing, label tokens " X"/" Y", plateau-gated read-out).
We run the committed run_ft_sweep.run_model under THAT lib, but repoint its data/results/cost paths to
the canonical top-level tree and write to a NEW dir so nothing committed is touched:
  results/cross_model_ft_37/ft_<key>_lora.jsonl

Run over the 100-pair `xmodel` set (spans all 37 features). Seed counts match the original: 5 seeds
(0-4) for the <=8B models, 3 seeds (0-2) for the 32B. Each Modal call runs in its own H100 container
(no @modal.concurrent), so --concurrency = parallel GPUs, result-neutral; --conc32 allows the 32B to run
on several containers (the original concurrency-1 pin was a shared-container OOM note, N/A here).

  .venv/bin/python experiments/cross_model/run_ft_x37.py --models qwen3_8b llama31_8b olmo2_7b --concurrency 60
  .venv/bin/python experiments/cross_model/run_ft_x37.py --models qwen3_32b --conc32 8
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
# LIB_DIR selects which cross-model-capable lib to run under: the stage 7 snapshot (H100; fine for the
# <=8B models) or the _lib_b200 copy (B200; needed for the 32B fine-tune, which OOMs on an 80GB H100 and
# matches the original 32B recipe). Override with env LIB_DIR.
import os  # noqa: E402
SNAP_LIB = Path(os.environ.get("LIB_DIR", str(TOP / "experiments" / "salience_flip" / "lib")))
sys.path.insert(0, str(SNAP_LIB))  # chosen lib wins over the top-level lib for finetune/modal_app/config

import config          # noqa: E402  (snapshot config)
import dataset         # noqa: E402
GRID = TOP / "data" / "grid"
config.DATA_DIR = TOP / "data"
config.RESULTS_DIR = TOP / "results"
config.COST_FILE = TOP / "total_cost.jsonl"
dataset.GRID_DIR = GRID
_orig_load = dataset.load_grid
dataset.load_grid = lambda pk, base_dir=GRID: _orig_load(pk, base_dir)  # default arg was bound to snapshot path

import cross_model as cm     # noqa: E402
import run_ft_sweep as R     # noqa: E402  (from experiments/cross_model, uses the snapshot lib modules)

PAIRS_FILE = _XMODEL_PAIRS
SEEDS_SMALL = [0, 1, 2, 3, 4]
SEEDS_32B = [0, 1, 2]


async def main(models, concurrency, conc32, assert_cached):
    R.OUT_DIR = TOP / "results" / "cross_model_ft_37"
    R.OUT_DIR.mkdir(parents=True, exist_ok=True)
    cm.FT_CONCURRENCY1_KEYS = set()          # allow explicit per-model concurrency (grad-ckpt config untouched)
    pairs = json.load(open(PAIRS_FILE))
    print(f"[ft_x37] {len(pairs)} pairs, models={models}, out={R.OUT_DIR}, GPU={__import__('modal_app').GPU}",
          flush=True)
    for k in models:
        seeds = SEEDS_32B if k == "qwen3_32b" else SEEDS_SMALL
        conc = conc32 if k == "qwen3_32b" else concurrency
        await R.run_model(k, "lora", pairs, seeds, conc, assert_cached)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, choices=list(cm.MODELS.keys()))
    ap.add_argument("--concurrency", type=int, default=60)
    ap.add_argument("--conc32", type=int, default=8)
    ap.add_argument("--assert-cached", action="store_true")
    a = ap.parse_args()
    asyncio.run(main(a.models, a.concurrency, a.conc32, a.assert_cached))
