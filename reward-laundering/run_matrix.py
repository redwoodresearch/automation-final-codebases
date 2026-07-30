"""Orchestrate the multi-seed A/B/C/D matrix: launch run_rl_pilot.py per (seed, condition) with a
gated concurrency limit and the per-seed train-problem offset (500 + 240*seed).

Each seed's 4 conditions share the SAME train problems (paired B-vs-A/D); seeds use disjoint train
sets (honest across-seed variance — see the phase progress log). The locked config: LR 1e-4 linear,
group 16, batch 8, 30 steps, eval every 6, panels 60 neutral / 40 coupled x 4, keep steps 17,23,29.

Usage (run in background; it waits for every job):
  .venv/bin/python run_matrix.py --seeds 1,2,3,4 --max_parallel 8
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONDS = ("a", "b", "c", "d")
LOGDIR = ROOT / "results" / "rl_matrix_logs"


def job_cmd(seed: int, cond: str, args) -> list[str]:
    offset = args.base_offset + args.offset_step * seed
    return [
        ".venv/bin/python", "run_rl_pilot.py", "--condition", cond, "--seed", str(seed),
        "--n_steps", str(args.n_steps), "--group_size", "16", "--batch_size", "8",
        "--lr", "1e-4", "--lr_schedule", "linear", "--eval_every", "6",
        "--problem_offset", str(offset),
        "--n_eval_clean", "60", "--n_eval_coupling", "40", "--n_eval_samples", "4",
        "--concurrency", str(args.concurrency), "--tag", f"seed{seed}", "--keep_steps", "17,23",
    ]


async def run_job(seed: int, cond: str, args, sem: asyncio.Semaphore) -> tuple:
    async with sem:
        log = LOGDIR / f"seed{seed}_{cond}.log"
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] START seed{seed} {cond} (offset "
              f"{args.base_offset + args.offset_step * seed})", flush=True)
        with open(log, "w") as f:
            proc = await asyncio.create_subprocess_exec(
                *job_cmd(seed, cond, args), cwd=str(ROOT), stdout=f,
                stderr=asyncio.subprocess.STDOUT)
            rc = await proc.wait()
        print(f"[{time.strftime('%H:%M:%S')}] DONE  seed{seed} {cond} rc={rc} "
              f"({(time.time() - t0) / 60:.0f} min)", flush=True)
        return (seed, cond, rc)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True, help="comma-separated seed indices, e.g. 1,2,3,4")
    ap.add_argument("--conditions", default="a,b,c,d")
    ap.add_argument("--n_steps", type=int, default=30)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max_parallel", type=int, default=8, help="max concurrent training subprocesses")
    ap.add_argument("--base_offset", type=int, default=500)
    ap.add_argument("--offset_step", type=int, default=240)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    conds = [c for c in args.conditions.split(",") if c.strip()]
    LOGDIR.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(args.max_parallel)
    jobs = [(s, c) for s in seeds for c in conds]
    print(f"Launching {len(jobs)} jobs (seeds {seeds} x conds {conds}), max_parallel "
          f"{args.max_parallel}, concurrency {args.concurrency}/job", flush=True)
    results = await asyncio.gather(*[run_job(s, c, args, sem) for s, c in jobs])
    fails = [(s, c, rc) for s, c, rc in results if rc != 0]
    print(f"\nAll done. {len(results) - len(fails)}/{len(results)} succeeded.", flush=True)
    if fails:
        print(f"FAILURES: {fails}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
