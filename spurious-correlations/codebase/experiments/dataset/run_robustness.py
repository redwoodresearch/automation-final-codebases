"""§2b robustness: does the fine-tune's held-out logprob difference keep its sign across prompt
wordings and neutral label-token choices? Re-fine-tunes for every (template x token-pair) at the
pinned LR/steps and reports the swing. Also re-evaluates the un-fine-tuned baseline for each combo.

The held-out logprob difference per run is taken at the checkpoint selected by the training
(diagonal) signal — never the held-out cells.

Usage: PYTHONPATH=$PWD/lib python experiments/dataset/run_robustness.py --lr 2e-5 --max-steps 150 [--seeds 0 1]
"""

from __future__ import annotations

import argparse
import asyncio
import json

import config
import dataset
import finetune
import modal_app as M
from cost_tracker import CostTracker
from modal_runner import AsyncModalSession
from properties import PAIRS

# The §2b grid (>=2 wordings x >=2 token pairs; we use 3 x 3).
TEMPLATES = ["classify", "group", "label"]
TOKEN_PAIRS = [(" A", " B"), (" X", " Y"), (" M", " N")]


async def main(lr: float, max_steps: int, seeds: list[int], assert_cached: bool):
    tracker = CostTracker(config.COST_FILE, run_description="run_robustness")
    inputs_by_pair = {pk: dataset.build_inputs(dataset.load_grid(pk)) for pk in PAIRS}
    eval_steps = sorted(set([0, 25, 50, 75, 100, max_steps]))

    ft_tasks, base_tasks = {}, {}
    async with AsyncModalSession(tracker, assert_cached) as sess:
        for pk in PAIRS:
            for tname in TEMPLATES:
                template = config.PROMPT_TEMPLATES[tname]
                for tok_a, tok_b in TOKEN_PAIRS:
                    base_tasks[(pk, tname, tok_a)] = sess.call(
                        "baseline_logprob_diff", M.baseline_logprob_diff,
                        finetune.build_baseline_kwargs(inputs_by_pair[pk], template, tok_a, tok_b),
                        finetune.BASELINE_CALL_UUID)
                    for seed in seeds:
                        kwargs = finetune.build_ft_kwargs(
                            inputs_by_pair[pk], template, tok_a, tok_b, lr, seed,
                            max_steps=max_steps, eval_steps=eval_steps)
                        ft_tasks[(pk, tname, tok_a, seed)] = sess.call(
                            "finetune_and_eval", M.finetune_and_eval, kwargs, finetune.FT_CALL_UUID)
        base_done = await asyncio.gather(*base_tasks.values())
        ft_done = await asyncio.gather(*ft_tasks.values())

    rows = []
    for (pk, tname, tok_a, seed), (res, _c) in zip(ft_tasks, ft_done):
        # Read at the SAME pinned checkpoint as the headline §2a result (training-loss plateau).
        plateau = finetune.loss_plateau_step(res["train_losses"], eval_steps)
        held = finetune.checkpoint_value(res["curve"], plateau) if plateau is not None else res["curve"][-1]["heldout_raw"]
        rows.append({"pair": pk, "template": tname, "tokens": tok_a.strip(), "seed": seed,
                     "plateau_step": plateau, "heldout_raw_at_plateau": held,
                     "heldout_raw_final": res["curve"][-1]["heldout_raw"]})
    base_rows = []
    for (pk, tname, tok_a), (res, _c) in zip(base_tasks, base_done):
        base_rows.append({"pair": pk, "template": tname, "tokens": tok_a.strip(),
                          "heldout_raw": res["heldout"]["raw"], "base_mean_p_a": res["base_mean_p_a"]})

    out = {"lr": lr, "max_steps": max_steps, "seeds": seeds, "finetune": rows, "baseline": base_rows}
    (config.RESULTS_DIR / "robustness.json").write_text(json.dumps(out, indent=2))

    print(f"\n=== Fine-tune held-out logprob diff across prompt x token (lr={lr:.0e}, @plateau checkpoint) ===")
    for pk in PAIRS:
        vals = [r["heldout_raw_at_plateau"] for r in rows if r["pair"] == pk]
        signs = set(v > 0 for v in vals)
        print(f"  {pk}: n={len(vals)} range=[{min(vals):+.3f}, {max(vals):+.3f}] "
              f"mean={sum(vals)/len(vals):+.3f}  sign_stable={'YES' if len(signs)==1 else 'NO (FLIPS)'}")
    print(f"\nwrote results/dataset/robustness.json  (run cost ${tracker.run_cost:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--max-steps", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--assert-cached", action="store_true")
    args = ap.parse_args()
    asyncio.run(main(args.lr, args.max_steps, args.seeds, args.assert_cached))
