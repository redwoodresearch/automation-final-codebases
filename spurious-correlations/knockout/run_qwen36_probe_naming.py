"""Qwen3.6-27B BASE-model MEAN-POOLED probe UNDER EXPLICIT PROMPTING (feature naming), on the 28
selected safety-behavior pairs. The probe analogue of the 27B fine-tune-under-naming result
(inoc_safety_27b / inoc_lying_27b) and the direct mirror of the 1.7B explicit-prompting probe
(codebase/experiments/complete_graph/run_inoculation_probe.py): NO training — the naming prompt
for the pair's safety behavior is baked into every passage, the base model's mean-pooled hidden
states are read, the SAME frozen linear-probe instrument (probe_fit.fit_grid, uncontrolled, C=0.1,
5 seeds, every hidden state) is fit, and the held-out margin is read.

Two named layouts per pair (both instruction style, primary wording variant 0):
  * named_appended : naming.render(behavior, text) — instruction AFTER the passage. This is the
                     EXACT layout the 1.7B explicit-prompting probe uses (naming.naming_template),
                     so it is the like-for-like 1.7B mirror and the PRIMARY reported condition.
  * named_first    : the instruction-FIRST layout (inoc_first_common.first_naming_template) —
                     instruction BEFORE the passage. Matches the 27B fine-tune-under-naming PRIMARY
                     condition (named_first) so the probe table aligns with the FT table.

The NEUTRAL baseline (plain passage, no naming) is the existing 27B mean-pooled probe in
results/qwen36_27b/probes/<pair>.json (representations.meanpooled) — REUSED, not recomputed.

Resume-safe: results/qwen36_27b/probe_naming/<pair>__<layout>.json. First (pair, layout) acts as a
cost gate against qwen36_common.BUDGET_CAP.

Usage:
  cd knockout && set -a && . path/to/your/.env  # MODAL_TOKEN_ID/SECRET && set +a
  PYTHONPATH=../codebase/lib:. ../codebase/.venv/bin/python run_qwen36_probe_naming.py
"""

from __future__ import annotations

import asyncio
import sys
import traceback

import modal

import inoc_first_common as IFC
import naming
import qwen36_common as QC
import qwen36_modal as QM
from cost_tracker import CostTracker  # codebase lib

CONCURRENCY = 8
LAYOUTS = ["named_appended", "named_first"]
OUT_DIR = QC.RESULTS_DIR / "probe_naming"


def out_path(pair: str, layout: str):
    return OUT_DIR / f"{pair}__{layout}.json"


def render_cell(texts: list[str], feat: str, layout: str) -> list[str]:
    """Bake the naming prompt for `feat` into every text, in the requested layout."""
    if layout == "named_appended":
        return [naming.render(feat, t) for t in texts]  # instruction after the passage (1.7B mirror)
    if layout == "named_first":
        tmpl = IFC.first_naming_template(feat)           # instruction before the passage
        return [tmpl.format(text=t, a="X", b="Y") for t in texts]
    raise ValueError(layout)


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recs = {r["pair"]: r for r in QC.selected_pair_records()}
    tracker = CostTracker(QC.COST_FILE,
                          run_description="qwen36 27B named (explicit-prompting) mean-pooled probe (28 pairs x 2 layouts)")

    jobs = [(r["pair"], layout) for r in QC.selected_pair_records() for layout in LAYOUTS
            if not out_path(r["pair"], layout).exists()]
    n_total = len(QC.SELECTED) * len(LAYOUTS)
    print(f"[named-probe] {n_total} (pair x layout) total, {n_total - len(jobs)} cached, "
          f"{len(jobs)} to run (concurrency {CONCURRENCY})", flush=True)

    sem = asyncio.Semaphore(CONCURRENCY)
    failures: dict[str, str] = {}
    abort = asyncio.Event()
    done = 0

    async def run_job(pair: str, layout: str):
        nonlocal done
        async with sem:
            if abort.is_set():
                failures[f"{pair}__{layout}"] = "skipped: aborted on projection/budget"
                return
            feat = recs[pair]["behavior"]
            payload = QC.probe_payload(pair)  # bare groups + lengths + seeds + toward-f1 signs
            rendered = {c: render_cell(payload["groups"][c], feat, layout) for c in payload["groups"]}
            try:
                res = await QM.probe_named_meanpooled.remote.aio(
                    model_name=QC.MODEL_27B, pair=pair, rendered_groups=rendered,
                    lengths=payload["lengths"], seeds_by_cell=payload["seeds_by_cell"],
                    signs=payload["signs"], probe_c=QC.PROBE_C, n_seeds=QC.PROBE_N_SEEDS)
            except Exception as e:  # noqa: BLE001
                failures[f"{pair}__{layout}"] = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
                print(f"[FAIL] {pair}/{layout}: {type(e).__name__}: {e}", flush=True)
                return
            cost = tracker.add_modal_gpu_cost(wall_seconds=res["elapsed"], gpu=QM.GPU, is_sandbox=False)
            res["cost"] = cost
            res["named_feature"] = feat
            res["layout"] = layout
            QC.save_json(out_path(pair, layout), res)
            done += 1
            mp = res["meanpooled"]
            print(f"[{done}/{len(jobs)}] {pair} [{layout}] name={feat}: L{mp['selected_layer']} "
                  f"raw(->f1) {mp['raw_selected']:+.2f} ({res['elapsed']:.0f}s, ${cost:.2f}, "
                  f"cum ${tracker.run_cost:.2f})", flush=True)
            if done == 1:
                projected = QC.qwen36_spend() + tracker.run_cost + (len(jobs) - 1) * cost
                print(f"[projection] per-job ${cost:.2f} => qwen36 projected total "
                      f"${projected:.0f} (cap ${QC.BUDGET_CAP:.0f})", flush=True)
                if projected > QC.BUDGET_CAP:
                    print("[projection] ABOVE CAP — aborting remaining jobs", flush=True)
                    abort.set()
            if QC.qwen36_spend() + tracker.run_cost > QC.BUDGET_CAP:
                print("[budget] cumulative qwen36 spend over cap — aborting", flush=True)
                abort.set()

    with modal.enable_output():
        async with QM.app.run.aio():
            if jobs:
                await run_job(*jobs[0])                                  # cost gate + warms container
                await asyncio.gather(*[run_job(*j) for j in jobs[1:]])

    QC.save_json(QC.RESULTS_DIR / "probe_naming_failures.json", failures)
    n_done = sum(out_path(r["pair"], layout).exists()
                 for r in QC.selected_pair_records() for layout in LAYOUTS)
    QC.record_spend("run_qwen36_probe_naming.py", tracker.run_cost,
                    f"{n_done}/{n_total} done, {len(failures)} failures")
    print(f"\n[named-probe] {n_done}/{n_total} on disk, {len(failures)} failures; "
          f"run cost ${tracker.run_cost:.2f}, qwen36 total ${QC.qwen36_spend():.2f}", flush=True)
    sys.exit(3 if abort.is_set() else 0)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
