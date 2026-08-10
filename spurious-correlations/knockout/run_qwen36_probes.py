"""Qwen3.6-27B BASE-model probes for the 28 selected pairs: passage-averaged (bare text,
mean-pooled) and decision-token (rendered classify prompt, last position) linear probes at every
hidden state (frozen instrument: probe_fit.fit_grid, uncontrolled, C=0.1, 5 seeds, layer by
held-in val-CE), plus the un-fine-tuned baseline held logprob diff. One Modal call per pair
(activations never leave the container). Resume-safe: results/qwen36_27b/probes/<pair>.json.

The first pair acts as a cost gate: its measured cost projects the probe phase; aborts if the
overall qwen36 total would exceed the cap.

Usage:
  cd knockout && set -a && . path/to/your/.env  # MODAL_TOKEN_ID/SECRET && set +a
  PYTHONPATH=../codebase/lib:. ../codebase/.venv/bin/python run_qwen36_probes.py
"""

from __future__ import annotations

import asyncio
import sys
import traceback

import modal

import qwen36_common as QC
import qwen36_modal as QM
from cost_tracker import CostTracker  # codebase lib

CONCURRENCY = 8


async def main() -> None:
    val = QC.load_json(QC.RESULTS_DIR / "validation.json")
    if not val.get("gate", {}).get("passed"):
        print("validation gate not passed — refusing to run")
        sys.exit(1)

    tracker = CostTracker(QC.COST_FILE, run_description="qwen36 27B base-model probes (28 pairs)")
    QC.PROBE_DIR.mkdir(parents=True, exist_ok=True)
    pairs = [r["pair"] for r in QC.selected_pair_records()
             if not (QC.PROBE_DIR / f"{r['pair']}.json").exists()]
    print(f"[probes] {len(QC.SELECTED)} pairs total, {len(QC.SELECTED) - len(pairs)} cached, "
          f"{len(pairs)} to run (concurrency {CONCURRENCY})")

    sem = asyncio.Semaphore(CONCURRENCY)
    failures: dict[str, str] = {}
    abort = asyncio.Event()
    done = 0

    async def run_pair(pair: str):
        nonlocal done
        async with sem:
            if abort.is_set():
                failures[pair] = "skipped: aborted on projection/budget"
                return
            try:
                res = await QM.probe_extract_fit.remote.aio(
                    model_name=QC.MODEL_27B, **QC.probe_payload(pair),
                    probe_c=QC.PROBE_C, n_seeds=QC.PROBE_N_SEEDS)
            except Exception as e:  # noqa: BLE001
                failures[pair] = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
                print(f"[FAIL] {pair}: {type(e).__name__}: {e}")
                return
            cost = tracker.add_modal_gpu_cost(wall_seconds=res["elapsed"], gpu=QM.GPU, is_sandbox=False)
            res["cost"] = cost
            QC.save_json(QC.PROBE_DIR / f"{pair}.json", res)
            done += 1
            mp, dc = res["representations"]["meanpooled"], res["representations"]["decision"]
            print(f"[{done}/{len(pairs)}] {pair}: meanpooled L{mp['selected_layer']} "
                  f"{mp['raw_selected']:+.2f} | decision L{dc['selected_layer']} "
                  f"{dc['raw_selected']:+.2f} | base {res['baseline_heldout']['raw']:+.2f} "
                  f"({res['elapsed']:.0f}s, ${cost:.2f}, cum ${tracker.run_cost:.2f})")
            if done == 1:
                projected = QC.qwen36_spend() + tracker.run_cost + (len(pairs) - 1) * cost
                print(f"[projection] per-pair ${cost:.2f} => qwen36 projected total "
                      f"${projected:.0f} (cap ${QC.BUDGET_CAP:.0f})")
                if projected > QC.BUDGET_CAP:
                    print("[projection] ABOVE CAP — aborting remaining probe pairs")
                    abort.set()
            if QC.qwen36_spend() + tracker.run_cost > QC.BUDGET_CAP:
                print("[budget] cumulative qwen36 spend over cap — aborting")
                abort.set()

    with modal.enable_output():
        async with QM.app.run.aio():
            # first pair alone (cost gate + warms the shared containers), then the rest
            if pairs:
                await run_pair(pairs[0])
                await asyncio.gather(*[run_pair(p) for p in pairs[1:]])

    QC.save_json(QC.RESULTS_DIR / "probe_failures.json", failures)
    n_done = sum((QC.PROBE_DIR / f"{r['pair']}.json").exists() for r in QC.selected_pair_records())
    QC.record_spend("run_qwen36_probes.py", tracker.run_cost,
                    f"{n_done}/{len(QC.SELECTED)} done, {len(failures)} failures")
    print(f"\n[probes] {n_done}/{len(QC.SELECTED)} pair summaries on disk, {len(failures)} failures; "
          f"run cost ${tracker.run_cost:.2f}, qwen36 total ${QC.qwen36_spend():.2f}")
    sys.exit(3 if abort.is_set() else 0)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
