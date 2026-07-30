"""Correct-hint faithfulness restricted to questions the model never solves unhinted.

The correct-hint condition mixes two populations. Some questions the model only answers correctly
because the hint told it to; those are the ones the metric is about. Others it would have answered
correctly anyway on a re-ask, and the hint is redundant — a judge scores those unfaithful because
the reasoning genuinely didn't use the hint, which drags the measured rate down for a reason that
has nothing to do with concealment.

Rather than correct for the mixture (scripts/analyze_flip_confound.py bounds that), this filters
it out. Using the unhinted resamples, keep only questions where NO unhinted sample produces the
correct answer: the sample-0 baseline is wrong by construction (that is what made the question
correct-hint-eligible) and every resample is wrong too. On those questions the unhinted rate of
landing on the hinted option is ~0, so any hinted sample that lands on it was moved there by the
hint, and the RAW verbalization rate is already the quantity of interest — no α, no mixture model.

That is also the per-question version of the post's own recommendation to use harder questions:
hard questions are the ones where the unhinted correct rate is ~0 by construction.

Reported raw on both sides, since normalization is what the filter replaces. The incorrect-hint
side needs no filter (the unhinted rate of landing on one specific wrong option is under 3%) and
is shown raw for comparability.

Depth: k = 1 baseline + 4 resamples for most models (1 + 6 for Opus 4.1). With k = 5, "never
correct" bounds the unhinted correct rate loosely per question, so the filter is a strong
aggregate improvement rather than a per-question guarantee; models whose filtered pool is tiny
are flagged and should not be read individually.

Runs offline from committed judge verdicts plus the resample transcripts
(`python data/download_transcripts.py`). Writes results/filtered_faithfulness.json.

  python scripts/analyze_filtered_faithfulness.py
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.dataset import load_file
from lib.metrics import wilson_ci
from lib.tables import write_table

OUT_PATH = Path("results/filtered_faithfulness.json")
CORRECT_LETTER_SOURCE = {"unhinted_plain": "suggestion_True",
                         "unhinted_fewshot_symbol": "fewshot_symbol_True"}
# Which unhinted baseline governs each hint type's correct-hint cell.
BASELINE_FOR = {"suggestion": "unhinted_plain", "posthoc": "unhinted_plain",
                "metadata": "unhinted_plain", "grader_hacking": "unhinted_plain",
                "unethical_information": "unhinted_plain",
                "fewshot_symbol": "unhinted_fewshot_symbol"}
MIN_POOL = 25  # below this the filtered rate is too thin to read per model


def never_correct_unhinted(resample_path: str) -> dict[str, set[int]]:
    """Per baseline condition, the questions where no unhinted sample lands on the correct answer.

    Questions in this file already have a wrong sample-0 answer -- that is what made them
    correct-hint-eligible -- so this only has to check the resamples.
    """
    answers = defaultdict(list)
    with open(resample_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            answers[(r["input"]["condition"], r["input"]["question_index"])].append(r["output"]["answer"])

    out, pools = {}, {}
    for condition, source in CORRECT_LETTER_SOURCE.items():
        correct = load_file(source)
        keys = [k for k in answers if k[0] == condition]
        clean = {k[1] for k in keys
                 if all(a != correct[k[1]].hint for a in answers[k] if a is not None)}
        out[condition] = clean
        pools[condition] = len(keys)
    return out, pools


def verdicts_by_condition(paths: list[str]) -> dict[str, list[tuple[int, bool]]]:
    """condition -> [(question_index, verbalized)] over the retained (judged) cases."""
    out = defaultdict(list)
    for path in paths:
        if not Path(path).exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                v = (r.get("output") or {}).get("verbalized")
                if v is None:
                    continue
                out[r["input"]["condition"]].append((r["input"]["question_index"], v))
    return out


def rate(pairs) -> tuple[float | None, int, int]:
    if not pairs:
        return None, 0, 0
    k = sum(1 for _, v in pairs if v)
    return k / len(pairs), k, len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    following = json.loads(Path("results/following_tables.json").read_text())["models"]

    out = {"_meta": {
        "generated_by": "scripts/analyze_filtered_faithfulness.py",
        "description": "Raw correct-hint verbalization rate restricted to questions where no "
                       "unhinted sample produces the correct answer, so every retained case is "
                       "hint-caused. MMLU only. Incorrect-hint side shown raw, unfiltered.",
        "min_pool_for_per_model_reading": MIN_POOL,
    }}
    rows, skipped = [], []
    for short, fm in following.items():
        pool = "full" if short == "sonnet-4-5" else "standard"
        rpath = f"results/resamples_true_eligible_{short}_{pool}.jsonl"
        judges = [f"results/judge_tier1_{short}_{pool}.jsonl",
                  f"results/judge_tier2_{short}_{pool}.jsonl"]
        if not Path(rpath).exists() or not any(Path(p).exists() for p in judges):
            skipped.append(fm["display"])
            continue

        clean, eligible_pools = never_correct_unhinted(rpath)
        by_cond = verdicts_by_condition(judges)

        filt, unfilt, incorrect, incorrect_same = [], [], [], []
        for ht, baseline in BASELINE_FOR.items():
            keep = clean[baseline]
            for idx, v in by_cond.get(f"{ht}_True", []):
                unfilt.append((idx, v))
                if idx in keep:
                    filt.append((idx, v))
            for idx, v in by_cond.get(f"{ht}_False", []):
                incorrect.append((idx, v))
                # Same-question comparison: the incorrect-hint cells on exactly the questions the
                # filter kept, so difficulty is held fixed across the two hint directions.
                if idx in keep:
                    incorrect_same.append((idx, v))

        r_filt, k_f, n_f = rate(filt)
        r_unfilt, _, n_u = rate(unfilt)
        r_inc, _, n_i = rate(incorrect)
        r_inc_same, _, n_is = rate(incorrect_same)
        if r_filt is None or r_inc is None:
            skipped.append(fm["display"])
            continue
        lo, hi = wilson_ci(k_f, n_f)
        rows.append({
            "model": fm["display"], "group": fm["group"],
            "n_eligible_questions": sum(eligible_pools.values()),
            "n_never_correct_questions": sum(len(v) for v in clean.values()),
            "kept_share": sum(len(v) for v in clean.values()) / max(1, sum(eligible_pools.values())),
            "correct_raw_unfiltered": r_unfilt, "n_unfiltered": n_u,
            "correct_raw_filtered": r_filt, "n_filtered": n_f,
            "correct_filtered_ci": [lo, hi],
            "incorrect_raw": r_inc, "n_incorrect": n_i,
            "incorrect_raw_same_questions": r_inc_same, "n_incorrect_same": n_is,
            "gap_unfiltered": r_inc - r_unfilt,
            "gap_filtered": r_inc - r_filt,
            "gap_same_questions": (r_inc_same - r_filt) if r_inc_same is not None else None,
            "survives": r_inc > r_filt,
            "survives_same_questions": r_inc_same is not None and r_inc_same > r_filt,
            "thin": n_f < MIN_POOL,
        })

    rows.sort(key=lambda r: -r["gap_filtered"])
    out["models"] = rows
    solid = [r for r in rows if not r["thin"]]
    out["summary"] = {
        "n_models": len(rows),
        "n_survive": sum(r["survives"] for r in rows),
        "n_solid": len(solid),
        "n_survive_solid": sum(r["survives"] for r in solid),
        "median_gap_unfiltered": statistics.median(r["gap_unfiltered"] for r in rows) if rows else None,
        "median_gap_filtered": statistics.median(r["gap_filtered"] for r in rows) if rows else None,
        "median_kept_share": statistics.median(r["kept_share"] for r in rows) if rows else None,
        "n_survive_same_questions": sum(r["survives_same_questions"] for r in rows),
        "median_gap_same_questions": statistics.median(
            r["gap_same_questions"] for r in rows if r["gap_same_questions"] is not None) if rows else None,
    }
    out["_meta"]["models_skipped"] = skipped
    if skipped:
        print(f"  skipped {len(skipped)} models (no resamples): {', '.join(skipped)}")
    write_table(OUT_PATH, out, [], allow_incomplete=args.allow_incomplete)

    print(f"\nMMLU, raw verbalization rates. 'filtered' = questions never answered correctly unhinted.")
    print(f"  {'model':22} {'kept':>6} {'correct unfilt':>15} {'correct filt':>21} "
          f"{'incorrect':>12} {'gap unfilt':>11} {'gap filt':>10}")
    for r in rows:
        flag = "  thin" if r["thin"] else ""
        print(f"  {r['model']:22} {r['kept_share']:5.0%} "
              f"{r['correct_raw_unfiltered']:9.1%} (n={r['n_unfiltered']:>4}) "
              f"{r['correct_raw_filtered']:9.1%} [{r['correct_filtered_ci'][0]:.0%},"
              f"{r['correct_filtered_ci'][1]:.0%}] (n={r['n_filtered']:>4}) "
              f"{r['incorrect_raw']:7.1%} {r['gap_unfiltered']*100:+9.1f}pp "
              f"{r['gap_filtered']*100:+8.1f}pp{flag}")
    s = out["summary"]
    print(f"\n  gap survives the filter for {s['n_survive']}/{s['n_models']} models "
          f"({s['n_survive_solid']}/{s['n_solid']} with a filtered pool of {MIN_POOL}+)")
    print(f"  median gap {s['median_gap_unfiltered']*100:+.1f}pp unfiltered -> "
          f"{s['median_gap_filtered']*100:+.1f}pp filtered")
    print(f"  median share of eligible questions kept by the filter: {s['median_kept_share']:.0%}")
    print(f"\n  same-question comparison (both hint directions on the filtered questions only):")
    print(f"    gap survives for {s['n_survive_same_questions']}/{s['n_models']} models, "
          f"median {s['median_gap_same_questions']*100:+.1f}pp")
    tight = min((r for r in rows if r["gap_same_questions"] is not None),
                key=lambda r: r["gap_same_questions"])
    print(f"    tightest: {tight['model']} — correct {tight['correct_raw_filtered']:.1%} vs "
          f"incorrect {tight['incorrect_raw_same_questions']:.1%} "
          f"(n={tight['n_filtered']} / {tight['n_incorrect_same']})")


if __name__ == "__main__":
    main()
