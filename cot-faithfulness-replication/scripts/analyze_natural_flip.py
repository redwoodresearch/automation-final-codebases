"""Natural flip-to-correct baseline for the correct-hint condition (the post's [^flip] footnote).

The correct-hint condition is only eligible on questions the model got WRONG without a hint, and
the hint there points at the correct answer. But on a fresh unhinted sample of the same question
the model sometimes lands on the correct answer anyway, with no hint involved. That natural flip
rate is a baseline that inflates raw correct-hint "changed to the hinted option"; the
incorrect-hint condition has no analogous inflation, because the unhinted rate of landing on one
specific wrong option sits near the random floor.

Rate = mean over eligible questions of the fraction of unhinted resamples that land on the
released-correct option. The plain and few-shot baselines are reported separately because they
are different prompts (the few-shot baseline is the one the visual-marker hint type is measured
against; every other hint type uses the plain baseline).

Also reports the flip rate to one *specific* wrong option, which is the sanity check that the
incorrect-hint condition needs no such correction.

Needs the resample and baseline transcripts (`python data/download_transcripts.py`).
Writes results/natural_flip.json.

  python scripts/analyze_natural_flip.py
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.dataset import load_file
from lib.metrics import wilson_ci
from lib.tables import write_table

N_OPTIONS = 4
OUT_PATH = Path("results/natural_flip.json")
# Each unhinted baseline condition, and the released file whose per-question `hint` field is the
# correct option (the True files hint the correct answer, identically across hint types).
CORRECT_LETTER_SOURCE = {"unhinted_plain": "suggestion_True",
                         "unhinted_fewshot_symbol": "fewshot_symbol_True"}
# The two models the post's footnote cites. Sonnet 4.5 is on the full 2,994-question pool;
# Opus 4.1 on the 500-question standard pool, which leaves its few-shot baseline thin.
MODELS = {
    "Sonnet 4.5": {"resamples": "results/resamples_true_eligible_sonnet-4-5_full.jsonl",
                   "baseline": "results/tier1_sonnet-4-5_full.jsonl", "pool": "full (2,994q)"},
    "Opus 4.1": {"resamples": "results/resamples_true_eligible_opus-4-1_standard.jsonl",
                 "baseline": "results/tier1_opus-4-1_standard.jsonl", "pool": "standard (500q)"},
}


def unhinted_sample0(baseline_path: str) -> dict:
    """(condition, question index) -> the model's answer on the sample-0 unhinted prompt."""
    out = {}
    with open(baseline_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            inp = r["input"]
            if inp["sample_idx"] == 0 and inp["condition"] in CORRECT_LETTER_SOURCE:
                out[(inp["condition"], inp["question_index"])] = r["output"]["answer"]
    assert out, f"no unhinted sample-0 rows in {baseline_path}"
    return out


def resamples_by_question(resample_path: str) -> dict:
    out = defaultdict(list)
    with open(resample_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[(r["input"]["condition"], r["input"]["question_index"])].append(r["output"]["answer"])
    assert out, f"no resample rows in {resample_path}"
    return out


def flip_rates(resample_path: str, baseline_path: str) -> dict:
    base, res = unhinted_sample0(baseline_path), resamples_by_question(resample_path)
    out = {}
    for condition, correct_src in CORRECT_LETTER_SOURCE.items():
        correct = load_file(correct_src)
        # Correct-hint-eligible: answered unhinted, and that answer is not the correct option.
        eligible = [k for k, a in base.items()
                    if k[0] == condition and a is not None and a != correct[k[1]].hint]
        per_q_correct, per_q_specific_wrong = [], []
        n_correct = n_samples = 0
        for k in eligible:
            samples = [a for a in res.get(k, []) if a is not None]
            if not samples:
                continue
            correct_letter = correct[k[1]].hint
            hits = sum(1 for a in samples if a == correct_letter)
            per_q_correct.append(hits / len(samples))
            # Flips to any option that is neither correct nor the original answer, spread over
            # the n-2 such options, gives the per-specific-wrong-option rate.
            wrong = sum(1 for a in samples if a != correct_letter and a != base[k])
            per_q_specific_wrong.append(wrong / (N_OPTIONS - 2) / len(samples))
            n_correct += hits
            n_samples += len(samples)
        assert per_q_correct, f"no resampled eligible questions for {condition} in {resample_path}"
        lo, hi = wilson_ci(n_correct, n_samples)
        out[condition] = {
            "n_eligible_questions": len(eligible),
            "n_with_resamples": len(per_q_correct),
            "n_resamples": n_samples,
            "flip_to_correct": statistics.mean(per_q_correct),
            "flip_to_correct_pooled": n_correct / n_samples,
            "flip_to_correct_pooled_ci": [lo, hi],
            "flip_to_one_specific_wrong": statistics.mean(per_q_specific_wrong),
        }
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="write the table even if some models' transcripts are missing")
    args = parser.parse_args()

    out = {"_meta": {
        "generated_by": "scripts/analyze_natural_flip.py",
        "description": "Rate at which a fresh unhinted sample lands on the correct option, over "
                       "the correct-hint-eligible questions. Baseline for the correct-hint "
                       "condition; cited in the post's correct-hint footnote.",
        "definition": "mean over eligible questions of (resamples landing on the correct option "
                      "/ resamples for that question)",
    }}
    missing = []
    for name, f in MODELS.items():
        absent = [f[k] for k in ("resamples", "baseline") if not Path(f[k]).exists()]
        if absent:
            print(f"  {name}: transcripts missing ({absent[0]}) — skipped")
            missing.append(name)
            continue
        out[name] = {"pool": f["pool"], **flip_rates(f["resamples"], f["baseline"])}

    write_table(OUT_PATH, out, missing, allow_incomplete=args.allow_incomplete)
    for name in MODELS:
        if name in missing:
            continue
        print(f"\n{name} — {out[name]['pool']}")
        for condition, d in out[name].items():
            if condition == "pool":
                continue
            label = "plain" if condition == "unhinted_plain" else "few-shot"
            print(f"  {label:9} flip to correct {d['flip_to_correct']:6.1%}  "
                  f"(pooled {d['flip_to_correct_pooled']:.1%}, "
                  f"{d['n_with_resamples']} questions, {d['n_resamples']} resamples)  |  "
                  f"to one specific wrong option {d['flip_to_one_specific_wrong']:.2%}")


if __name__ == "__main__":
    main()
