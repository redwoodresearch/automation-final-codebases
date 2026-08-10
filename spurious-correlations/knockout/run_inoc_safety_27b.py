"""Qwen3.6-27B inoculation sweep for the four non-deception safety behaviors (see
inoc_safety_27b_common.py). Generalizes run_inoc_lying_27b.py over the pair's safety behavior.

For each safety pair, fine-tune the 27B model under neutral-first, named-first and named-appended
prompts (2 seeds each), reusing the safety run's bare-neutral fine-tune. Resume-safe (one JSON per
pair/condition/seed under results/qwen36_27b/inoc_safety/). Projects the full cost after a derisk
and aborts if it exceeds the cap.

22 pairs x 3 conditions x 2 seeds = 132 new fine-tunes.

Usage:
  cd knockout && set -a && . path/to/your/.env  # MODAL_TOKEN_ID/SECRET && set +a
  PYTHONPATH=../codebase/lib:. ../codebase/.venv/bin/python run_inoc_safety_27b.py [feature|pair ...]
    (no args = all four features; a feature name expands to its pairs; a pair runs just that pair)
"""

from __future__ import annotations

import asyncio
import sys
import traceback

import modal

import inoc_safety_27b_common as IS
import qwen36_common as QC
import qwen36_modal as QM
from cost_tracker import CostTracker  # codebase lib

CONCURRENCY = 100            # per standing guidance for Modal fine-tune sweeps
PROJECTION_AFTER = 6         # completed runs before the go/no-go projection
BUDGET_CAP = 150.0           # task cap (USD) for THIS experiment's new compute


def _resolve_pairs(args: list[str]) -> list[str]:
    if not args:
        return list(IS.SAFETY_PAIRS)
    out: list[str] = []
    for a in args:
        if a in IS.FEATURES:
            out.extend(IS.PAIRS_BY_FEATURE[a])
        elif a in IS.SAFETY_PAIRS:
            out.append(a)
        else:
            raise SystemExit(f"{a!r} is neither a safety feature {IS.FEATURES} nor a safety pair")
    # de-dup, keep order
    seen = set()
    return [p for p in out if not (p in seen or seen.add(p))]


def _ckpt_summary(pair: str, cond: str, seed: int, res: dict, cost: float) -> dict:
    row = next(c for c in res["curve"] if c["step"] == res["saved_step"])
    _, is_named = IS.condition_template(cond, IS.behavior_of(pair))
    return {
        "pair": pair, "condition": cond, "seed": seed, "cost": cost,
        "behavior": IS.behavior_of(pair),
        "named_feature": IS.behavior_of(pair) if is_named else None,
        "behavior_sign": IS.behavior_sign(pair),
        "saved_step": res["saved_step"], "plateau_step": res["plateau_step"],
        "heldout_raw": row["heldout_raw"], "heldout_acc": row["heldout_acc"],
        "toward_behavior": IS.toward_behavior(pair, row["heldout_raw"]),
        "train_acc": row["train_acc"], "train_correct_logp": row["train_correct_logp"],
        "baseline_step0_raw": next(c for c in res["curve"] if c["step"] == 0)["heldout_raw"],
        "elapsed": res["elapsed"], "load_elapsed": res["load_elapsed"],
        "max_mem_gib": res["max_mem_gib"], "dtype": res["dtype"], "optimizer": res["optimizer"],
        "curve": res["curve"], "train_losses": res["train_losses"],
    }


async def main() -> None:
    pairs = _resolve_pairs(sys.argv[1:])

    tracker = CostTracker(QC.COST_FILE, run_description="qwen36 27B inoculation-for-safety (4 behaviors)")
    IS.FT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(p, cond, s) for p in pairs for cond in IS.CONDITIONS for s in QC.SEEDS
            if not IS.ft_path(p, cond, s).exists()]
    n_total = len(pairs) * len(IS.CONDITIONS) * len(QC.SEEDS)
    print(f"[sweep] pairs={pairs}")
    print(f"[sweep] {n_total} runs across {len(pairs)} pairs x {len(IS.CONDITIONS)} conds x "
          f"{len(QC.SEEDS)} seeds, {n_total - len(jobs)} cached, {len(jobs)} to run "
          f"(concurrency {CONCURRENCY}, cap ${BUDGET_CAP:.0f})")

    sem = asyncio.Semaphore(CONCURRENCY)
    failures: dict[str, str] = {}
    done = 0
    abort = asyncio.Event()

    async def run_one(pair: str, cond: str, seed: int):
        nonlocal done
        async with sem:
            if abort.is_set():
                failures[f"{pair}__{cond}__seed{seed}"] = "skipped: aborted on projection/budget"
                return
            template, _is_named = IS.condition_template(cond, IS.behavior_of(pair))
            inputs = QC.load_pair_inputs(pair)
            try:
                res = await QM.finetune_and_eval.remote.aio(
                    model_name=QC.MODEL_27B,
                    train_texts=inputs.train_texts,
                    train_labels=inputs.train_labels,
                    heldout_texts=inputs.heldout_texts,
                    heldout_f1_signs=inputs.heldout_f1_signs,
                    template=template, tok_a=QC.TOK_A, tok_b=QC.TOK_B,
                    lr=QC.PINNED_LR, max_steps=QC.FT_MAX_STEPS, seed=seed,
                    dtype="bfloat16", optimizer="adamw8bit", grad_ckpt=True,
                    eval_steps=QC.FT_EVAL_STEPS,
                )
            except Exception as e:  # noqa: BLE001
                failures[f"{pair}__{cond}__seed{seed}"] = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
                print(f"[FAIL] {pair} {cond} seed{seed}: {type(e).__name__}: {e}")
                return
            cost = tracker.add_modal_gpu_cost(wall_seconds=res["elapsed"], gpu=QM.GPU, is_sandbox=False)
            summ = _ckpt_summary(pair, cond, seed, res, cost)
            QC.save_json(IS.ft_path(pair, cond, seed), summ)
            done += 1
            warn = "" if summ["train_acc"] >= 0.98 else "  <-- train_acc<0.98 (did not memorize)"
            print(f"[{done}/{len(jobs)}] {pair} {cond} seed{seed}: "
                  f"toward_behavior={summ['toward_behavior']:+.2f} (raw={summ['heldout_raw']:+.2f}) "
                  f"step={summ['saved_step']} ({res['elapsed']:.0f}s, ${cost:.2f}, "
                  f"cum ${tracker.run_cost:.2f}){warn}")

            if done == PROJECTION_AFTER:
                costs = [QC.load_json(f).get("cost", 0.0) for f in IS.FT_DIR.glob("*.json")]
                per_ft = sum(costs) / max(1, len(costs))
                remaining = len(jobs) - done
                projected = tracker.run_cost + remaining * per_ft
                print(f"[projection] per-ft ${per_ft:.2f}, {remaining} runs left => this experiment "
                      f"projects ${projected:.0f} (cap ${BUDGET_CAP:.0f})")
                if projected > BUDGET_CAP:
                    print("[projection] ABOVE CAP — aborting remaining runs for reassessment")
                    abort.set()
            if tracker.run_cost > BUDGET_CAP:
                print("[budget] over cap — aborting remaining runs")
                abort.set()

    with modal.enable_output():
        async with QM.app.run.aio():
            await asyncio.gather(*[run_one(p, c, s) for p, c, s in jobs])

    QC.save_json(IS.FT_DIR / "failures.json", failures)
    QC.record_spend("run_inoc_safety_27b.py", tracker.run_cost,
                    f"{done}/{len(jobs)} new runs, {len(failures)} failures, pairs={pairs}")
    print(f"\n[sweep] {done}/{len(jobs)} new runs, {len(failures)} failures; "
          f"run cost ${tracker.run_cost:.2f}, qwen36 total ${QC.qwen36_spend():.2f}")
    sys.exit(3 if abort.is_set() else 0)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
