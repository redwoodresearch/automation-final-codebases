"""Compute the per-pair CPU baseline inputs for winner-prediction blog analysis 2: pair-local
TF-IDF lexical CV-AUC and token-length separability for each feature of each usable pair. Cached to
a results JSON (resumable; skips pairs already present). No API/GPU cost.

Usage: PYTHONPATH=lib .venv/bin/python experiments/complete_graph/run_winner_pred_baselines.py
       [--workers 8]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import config
import dataset
import winner_predict as WP
from extension_properties import split_pair_key
from merged_readouts import load_table

OUT = config.RESULTS_DIR / "complete_graph" / "winner_pred_baseline_inputs_v2.json"


def _one(pk: str) -> tuple[str, dict]:
    f1, f2 = split_pair_key(pk)
    grid = dataset.load_grid(pk)
    return pk, {"f1": f1, "f2": f2,
                "tfidf_auc": WP.tfidf_auc(grid, f1, f2),
                "length_sep": WP.length_separability(grid, f1, f2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    pairs = [r["pair"] for r in load_table()]
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [pk for pk in pairs if pk not in out]
    print(f"{len(todo)} pairs to compute ({len(out)} cached)")
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, (pk, res) in enumerate(ex.map(_one, todo), 1):
                out[pk] = res
                if i % 50 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)
                    OUT.write_text(json.dumps(out, indent=1))
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT} ({len(out)} pairs)")


if __name__ == "__main__":
    main()
