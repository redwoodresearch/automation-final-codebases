"""Salience-crank blog examples, MID-strength partner edition.

Follow-up to run_salience_crank_examples.py (leetspeak/all_caps). Same demonstration, but with
MID-strength style partners g instead of the two strongest ones, so the margin is smaller and more
believable: a passage where the weak feature f=deception (lying) is cranked to the max but a
mid-strength partner g is ABSENT, and the fine-tuned pair classifier still prefers g.

Pairs (f1 = g = mid-strength style feature, ABSENT in the cranked passages; f2 = deception, cranked
max). All are decisive g-wins (verdict follows_f1) per the committed ft_delta in
inoc_data/merged_readouts_v2.jsonl:
  figurative_language__deception   ft_delta +9.35
  emotional_tone__deception        ft_delta +10.25
  negation_density__deception      ft_delta +10.76
  second_person_address__deception ft_delta +10.38
  first_person_voice__deception    ft_delta +11.39
(vs leetspeak +13.36 / all_caps +14.94 in the prior run.)

Class convention (dataset.build_inputs / modal_app): class A = 0 = " X" = (f1+, f2+);
class B = 1 = " Y" = (f1-, f2-). The cranked passages are (g-absent, deception-present) =
held_f1neg cell type (f1 sign -1); if the trained classifier follows g it labels them " Y"
(g's ABSENT pole) despite the blatant lying, i.e. d = logit(" X") - logit(" Y") < 0.

Readout orientation:
  d = logit(" X") - logit(" Y")   at the label position, saved (plateau) checkpoint.
  toward g pole (" Y", absence of the style) = -d nats ; toward deception pole (" X") = +d nats.
  "classifier picked g"  <=>  d < 0.

Usage:
  cd knockout && set -a && . path/to/your/.env  # MODAL_TOKEN_ID/SECRET && set +a
  PYTHONPATH=../codebase/lib:. ../codebase/.venv/bin/python run_salience_crank_examples_mid.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import modal
import numpy as np

import common
import knockout_modal as KM
from cost_tracker import CostTracker  # codebase lib

HERE = Path(__file__).resolve().parent
CRANK_FILE = HERE / "results" / "salience_crank" / "cranked_deception.jsonl"
OUTDIR = common.RESULTS_DIR / "salience_crank_examples_mid"
SEEDS = [0, 1]

# f1 = g (MID-strength style feature, ABSENT in the cranked passages); f2 = deception (cranked max).
PAIRS = [
    {"pair": "figurative_language__deception", "g": "figurative_language", "committed_ft_delta": 9.34857638888889},
    {"pair": "emotional_tone__deception", "g": "emotional_tone", "committed_ft_delta": 10.254083333333334},
    {"pair": "negation_density__deception", "g": "negation_density", "committed_ft_delta": 10.759444444444444},
    {"pair": "second_person_address__deception", "g": "second_person_address", "committed_ft_delta": 10.374600694444444},
    {"pair": "first_person_voice__deception", "g": "first_person_voice", "committed_ft_delta": 11.392430555555556},
]


def load_cranked() -> list[dict]:
    return [json.loads(l) for l in open(CRANK_FILE) if l.strip()]


async def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tracker = CostTracker(
        common.COST_FILE,
        run_description="salience_crank_examples_mid: ft 5 mid-strength g__deception pairs, read out on cranked lying passages",
    )
    cranked = load_cranked()
    cr_texts = [r["text"] for r in cranked]
    n_cr = len(cr_texts)
    print(f"loaded {n_cr} cranked (max-deception, style-absent) passages")

    raw_path = OUTDIR / "ft_raw.json"
    raw = json.loads(raw_path.read_text()) if raw_path.exists() else {}

    # ---- fine-tune each pair x seed; eval on [real held-out cells] ++ [cranked] in one pass ----
    jobs = []
    for pc in PAIRS:
        inputs = common.load_pair_inputs(pc["pair"])
        heldout_texts = list(inputs.heldout_texts) + cr_texts
        # cranked are (g-absent, deception-present) => f1 sign -1, same as held_f1neg
        heldout_signs = list(inputs.heldout_f1_signs) + [-1] * n_cr
        for seed in SEEDS:
            key = f"{pc['pair']}__seed{seed}"
            if key in raw:
                continue
            jobs.append((pc, inputs, heldout_texts, heldout_signs, seed, key))

    if jobs:
        with modal.enable_output():
            async with KM.app.run.aio():
                async def run_one(pc, inputs, heldout_texts, heldout_signs, seed, key):
                    res = await KM.finetune_save_and_eval.remote.aio(
                        train_texts=inputs.train_texts,
                        train_labels=inputs.train_labels,
                        heldout_texts=heldout_texts,
                        heldout_f1_signs=heldout_signs,
                        template=common.TEMPLATE,
                        tok_a=common.TOK_A,
                        tok_b=common.TOK_B,
                        lr=common.PINNED_LR,
                        max_steps=common.FT_MAX_STEPS,
                        seed=seed,
                        ckpt_name=f"saliencecrankmid__{common.ckpt_name(pc['pair'], seed)}",
                        eval_steps=common.FT_EVAL_STEPS,
                    )
                    tracker.add_modal_gpu_cost(wall_seconds=res["elapsed"], gpu=KM.GPU, is_sandbox=False)
                    return key, res

                for key, res in await asyncio.gather(
                    *[run_one(*j) for j in jobs]
                ):
                    raw[key] = res
                    raw_path.write_text(json.dumps(raw, indent=1))
                    print(f"[done] {key}: plateau_step={res['plateau_step']} saved_step={res['saved_step']} elapsed={res['elapsed']:.0f}s")

    # ---- analyze ----
    per_pair = {}
    cr_d_by_pair: dict[str, np.ndarray] = {}
    for pc in PAIRS:
        pair = pc["pair"]
        inputs = common.load_pair_inputs(pair)
        n_pos = inputs.n_per_cell["held_f1pos"]
        n_neg = inputs.n_per_cell["held_f1neg"]
        n_real = n_pos + n_neg
        seed_rows = []
        cr_d_seeds = []
        for seed in SEEDS:
            res = raw[f"{pair}__seed{seed}"]
            row = next(c for c in res["curve"] if c["step"] == res["saved_step"])
            d = np.asarray(row["d_values"], dtype=np.float64)
            assert len(d) == n_real + n_cr, (len(d), n_real, n_cr)
            d_pos = d[:n_pos]                 # held_f1pos = (g+, dec-), sign +1
            d_neg = d[n_pos:n_real]           # held_f1neg = (g-, dec+), sign -1
            d_cr = d[n_real:]                 # cranked    = (g-, dec++)
            cr_d_seeds.append(d_cr)
            oriented = np.concatenate([d_pos * 1.0, d_neg * -1.0])
            ft_delta_repro = float(oriented.mean())
            frac_to_g = float((d_cr < 0).mean())
            mean_logodds_g = float((-d_cr).mean())   # nats toward g pole
            seed_rows.append({
                "seed": seed,
                "saved_step": res["saved_step"],
                "plateau_step": res["plateau_step"],
                "ft_delta_repro_realheld": ft_delta_repro,
                "cranked_frac_to_g": frac_to_g,
                "cranked_mean_logodds_toward_g_nats": mean_logodds_g,
                "cranked_mean_d_logitX_minus_logitY": float(d_cr.mean()),
            })
            print(f"{pair} seed{seed}: step={res['saved_step']:3d} "
                  f"ft_delta_repro={ft_delta_repro:+.2f} (committed {pc['committed_ft_delta']:+.2f}) | "
                  f"cranked frac->g={frac_to_g:.3f} mean_logodds->g={mean_logodds_g:+.2f} nats")
        cr_d = np.mean(cr_d_seeds, axis=0)      # seed-averaged per-passage d
        cr_d_by_pair[pair] = cr_d
        per_pair[pair] = {
            "g_feature": pc["g"],
            "f_feature": "deception",
            "committed_ft_delta": pc["committed_ft_delta"],
            "seeds": SEEDS,
            "per_seed": seed_rows,
            "mean_ft_delta_repro_realheld": float(np.mean([r["ft_delta_repro_realheld"] for r in seed_rows])),
            "cranked_frac_to_g_seedmean": float(np.mean([r["cranked_frac_to_g"] for r in seed_rows])),
            "cranked_mean_logodds_toward_g_nats_seedmean": float(np.mean([r["cranked_mean_logodds_toward_g_nats"] for r in seed_rows])),
            "cranked_per_passage_seedavg_nats_toward_g": [float(-x) for x in cr_d],
            "n_cranked": n_cr,
        }

    # ---- per-pair: a representative vivid passage (won by g, margin nearest the pair mean) ----
    examples = []
    for pc in PAIRS:
        pair = pc["pair"]
        cr_d = cr_d_by_pair[pair]
        nats_g = -cr_d
        won = nats_g > 0
        target = float(nats_g[won].mean()) if won.any() else float(nats_g.mean())
        # nearest-to-mean among won passages (representative, not the extreme)
        cand = np.where(won)[0] if won.any() else np.arange(len(nats_g))
        idx = int(cand[np.argmin(np.abs(nats_g[cand] - target))])
        d_seeds = []
        for s in SEEDS:
            r = raw[f"{pair}__seed{s}"]
            ci = next(i for i, c in enumerate(r["curve"]) if c["step"] == r["saved_step"])
            d_seeds.append(float(r["curve"][ci]["d_values"][-n_cr + idx]))
        examples.append({
            "pair": pair,
            "g_feature": pc["g"],
            "passage_index": idx,
            "topic": cranked[idx].get("topic"),
            "passage_text": cr_texts[idx],
            "seed_avg_d_logitX_minus_logitY": float(cr_d[idx]),
            "seed_avg_nats_toward_g": float(-cr_d[idx]),
            "per_seed_d": d_seeds,
            "picked_pole": "Y (g-absent / style-absent pole)" if cr_d[idx] < 0 else "X (deception pole)",
            "nats_toward_g": float(-cr_d[idx]),
            "nats_toward_deception": float(cr_d[idx]),
        })
        print(f"\n[example] {pair} topic={cranked[idx].get('topic')!r} idx={idx} "
              f"seed-avg d={cr_d[idx]:+.2f} -> {(-cr_d[idx]):+.2f} nats toward g")

    out = {
        "description": (
            "Salience-crank blog examples, mid-strength partner edition. Pair classifier fine-tuned "
            "with the study recipe (app spurious-knockout, classify template, ' X'/' Y' label-token "
            "loss, lr 1e-5, plateau checkpoint), then read out on the max-cranked deception passages "
            "(deception cranked to the max; the mid-strength style feature g is ABSENT). Companion to "
            "salience_crank_examples (leetspeak/all_caps); goal is a smaller, more believable margin."
        ),
        "label_pole_mapping": {
            "X": "class A = (f1+, f2+) = (g-present, deception-present)",
            "Y": "class B = (f1-, f2-) = (g-absent, deception-absent)",
            "d": "logit(' X') - logit(' Y') at the label position",
            "cranked_cell": "(g-absent, deception-present) = held_f1neg cell type",
            "picks_g_means": "outputs ' Y' (labels by ABSENCE of the style feature), i.e. d < 0",
            "nats_toward_g": "-d",
            "nats_toward_deception": "+d",
        },
        "recipe": {
            "app": "spurious-knockout",
            "template": common.TEMPLATE,
            "label_tokens": [common.TOK_A, common.TOK_B],
            "lr": common.PINNED_LR,
            "max_steps": common.FT_MAX_STEPS,
            "eval_steps": common.FT_EVAL_STEPS,
            "seeds": SEEDS,
            "model": KM.MODEL_NAME,
        },
        "per_pair": per_pair,
        "examples": examples,
        "run_cost_usd": tracker.run_cost,
    }
    (OUTDIR / "results.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUTDIR / 'results.json'}")
    print(f"run cost ${tracker.run_cost:.2f}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
