"""Shared cadence-eval builder for the SFT driver (`run_unfiltered_sft.py`).

`build_neutral_eval_fn` returns an async `eval_fn(step, sampler)` that measures the headline metric —
clean neutral-prompt subset-sum accuracy on the reserved eval pool — plus the pre-registered collapse
guards: pass@1 AND pass@k (diversity-sensitive; collapse can raise pass@1 while pass@k falls) and
CoT-token stats (the test-time-compute confound). Same eval path (prompt / forcing / verifier) as the
GRPO C-condition, so the number is identical to the banked headline. Kept here (not in the entry
scripts) so both drivers share exactly one eval path."""

from __future__ import annotations

import json
import time

from lib.endpoint_eval import (
    neutral_eval_records, passk_from_records, pass_rates_from_records, scaffold_pass_rates,
    token_stats_from_records,
)


def build_neutral_eval_fn(*, clean_instances, eval_path, n_eval_samples, concurrency, cache, tracker,
                          scaffold_eval=False, scaffold_instances=None):
    async def eval_fn(step, sampler):
        t0 = time.time()
        recs = await neutral_eval_records(
            sampler, clean_instances, n_samples=n_eval_samples, cache=cache, tracker=tracker,
            max_concurrency=concurrency)
        rates = pass_rates_from_records(recs)
        row = {"step": step, "neutral_subset_accuracy": sum(rates) / len(rates) if rates else 0.0,
               "neutral_passk": passk_from_records(recs), **token_stats_from_records(recs)}
        if scaffold_eval:
            scaf = await scaffold_pass_rates(
                sampler, scaffold_instances, n_samples=n_eval_samples, cache=cache, tracker=tracker,
                max_concurrency=concurrency)
            row["scaffold_subset_accuracy"] = sum(scaf) / len(scaf) if scaf else 0.0
        row["elapsed_s"] = time.time() - t0
        with open(eval_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        scaf_s = f" scaffold={row['scaffold_subset_accuracy']:.3f}" if scaffold_eval else ""
        print(f"  [eval round {step}] neutral acc={row['neutral_subset_accuracy']:.3f} "
              f"pass@k={row['neutral_passk']:.3f}{scaf_s}  ({row['elapsed_s']:.0f}s)", flush=True)

    return eval_fn
