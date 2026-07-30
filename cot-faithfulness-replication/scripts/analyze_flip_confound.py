"""How much of the correct-vs-incorrect faithfulness gap survives the spontaneous-flip confound?

The correct-hint condition is eligible only on questions the model got wrong unhinted, and the
hint there points at the correct answer. Some of the cases we count as "followed the hint" are
really the model re-solving the question on its own: on a fresh unhinted sample it lands on the
correct answer anyway (scripts/analyze_natural_flip.py measures how often). Those cases have no
hint to verbalize, so a judge scores them unfaithful — which deflates measured correct-hint
faithfulness on top of inflating the follow rate.

Chen et al.'s normalization does NOT remove them. Its α correction is estimated from q, the rate
of switching to some OTHER wrong option, i.e. it models chance switching as uniform over the
non-default options. A spontaneous flip to the correct answer is not uniform — it is systematically
toward correct — so q understates it badly whenever the hint IS the correct answer. The
mention-only robustness check does not remove them either: a model that solved the question
itself has no reason to mention the hint.

The adjustment. Per model and hint type, with p = P(changed to hint | eligible) in the
correct-hint condition and p_flip = P(a fresh unhinted resample lands on the correct answer):

    f = p_flip / p                      the share of retained cases that are spontaneous solvers
    adjusted faithfulness = observed / (1 - f)

This is deliberately an UPPER BOUND on the correction: it assumes every spontaneous solver is
judged unfaithful. Some of them do mention and depend on the hint as well, so the real correction
is smaller and the real adjusted gap is larger than what this prints.

Scope: this bound needs the flip RATE, which is only estimable where resampling was run to a
fixed depth without early stopping — the original MMLU collections. So it stays MMLU-only and is
not comparable to the post's equal-weight MMLU+GPQA figures.

scripts/analyze_filtered_faithfulness.py is the primary analysis and supersedes this one: it
filters the contaminated questions out rather than bounding their effect, needs only a binary
"ever correct unhinted" verdict rather than a rate, and covers both datasets. Keep this as the
independent cross-check — the two approaches agree.

Writes results/flip_confound.json.

  python scripts/analyze_flip_confound.py
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.tables import write_table
from scripts.analyze_natural_flip import flip_rates

OUT_PATH = Path("results/flip_confound.json")
# Which unhinted baseline each hint type's correct-hint cell is measured against.
BASELINE_FOR = {"suggestion": "unhinted_plain", "posthoc": "unhinted_plain",
                "metadata": "unhinted_plain", "grader_hacking": "unhinted_plain",
                "unethical_information": "unhinted_plain",
                "fewshot_symbol": "unhinted_fewshot_symbol"}


def resample_path(short: str, pool: str) -> str:
    return f"results/resamples_true_eligible_{short}_{pool}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    following = json.loads(Path("results/following_tables.json").read_text())["models"]
    faith = json.loads(Path("results/faithfulness_tables.json").read_text())
    faith = faith.get("models", faith)
    mentions = json.loads(Path("results/mentions_split.json").read_text())
    mentions = mentions.get("models", mentions)

    out = {"_meta": {
        "generated_by": "scripts/analyze_flip_confound.py",
        "description": "Correct-hint faithfulness adjusted for spontaneous flips to the correct "
                       "answer. MMLU only (the resamples are MMLU); upper bound on the correction.",
        "adjustment": "f = p_flip / p_follow; adjusted = observed / (1 - f)",
    }}
    rows, missing = [], []
    for short, fm in following.items():
        if short not in faith or not fm.get("mmlu"):
            continue
        pool = "full" if short == "sonnet-4-5" else "standard"
        rpath, bpath = resample_path(short, pool), f"results/tier1_{short}_{pool}.jsonl"
        if not (Path(rpath).exists() and Path(bpath).exists()):
            missing.append(fm["display"])
            continue

        flips = flip_rates(rpath, bpath)
        per_type = faith[short]["per_type"]["mmlu"]
        correct_cell = fm["mmlu"]["correct"]
        p_follow = correct_cell["change_to_hint"] / correct_cell["n_eligible"]

        # Per hint type: inflate that type's correct-hint faithfulness by 1/(1-f), then average
        # the six types the same way the published mean does.
        adjusted, observed, fs = [], [], []
        for ht, baseline in BASELINE_FOR.items():
            cell = per_type.get(ht)
            if not cell or not cell["correct"]["retained"]:
                continue
            p_flip = flips[baseline]["flip_to_correct"]
            f = min(p_flip / p_follow, 0.95) if p_follow else 0.95
            obs = cell["correct"]["normalized"]
            observed.append(obs)
            adjusted.append(min(obs / (1 - f), 1.0))
            fs.append(f)
        if not observed:
            missing.append(fm["display"])
            continue

        inc = faith[short]["mean_normalized"]["mmlu"]["incorrect"]
        obs_mean, adj_mean = statistics.mean(observed), statistics.mean(adjusted)

        # The mention-only robustness check inherits the same confound: a model that solved the
        # question itself has no reason to mention the hint. Same adjustment, model level.
        men = mentions.get(short)
        men_obs = men_adj = men_inc = None
        if men:
            f_model = min(statistics.mean(fs), 0.95)
            men_obs = men["correct"]["mmlu"]["mentions_pct"] / 100
            men_adj = min(men_obs / (1 - f_model), 1.0)
            men_inc = men["incorrect"]["mmlu"]["mentions_pct"] / 100

        rows.append({
            "mentions_correct_observed": men_obs,
            "mentions_correct_adjusted": men_adj,
            "mentions_incorrect": men_inc,
            "mentions_gap_adjusted": (men_inc - men_adj) if men_adj is not None else None,
            "model": fm["display"], "group": fm["group"],
            "p_follow_correct": p_follow,
            "p_flip_plain": flips["unhinted_plain"]["flip_to_correct"],
            "p_flip_fewshot": flips["unhinted_fewshot_symbol"]["flip_to_correct"],
            "spontaneous_share_mean": statistics.mean(fs),
            "faith_correct_observed": obs_mean,
            "faith_correct_adjusted": adj_mean,
            "faith_incorrect": inc,
            "gap_observed": inc - obs_mean,
            "gap_adjusted": inc - adj_mean,
            "survives": inc > adj_mean,
        })

    rows.sort(key=lambda r: -r["gap_adjusted"])
    out["models"] = rows
    out["summary"] = {
        "n_models": len(rows),
        "n_survive": sum(r["survives"] for r in rows),
        "median_gap_observed": statistics.median(r["gap_observed"] for r in rows) if rows else None,
        "median_gap_adjusted": statistics.median(r["gap_adjusted"] for r in rows) if rows else None,
        "median_spontaneous_share": statistics.median(r["spontaneous_share_mean"] for r in rows) if rows else None,
        "n_survive_mentions_only": sum(1 for r in rows if r.get("mentions_gap_adjusted") is not None
                                       and r["mentions_gap_adjusted"] > 0),
        "n_with_mentions": sum(1 for r in rows if r.get("mentions_gap_adjusted") is not None),
    }
    if missing:
        print(f"  no resample transcripts for {len(missing)} models — excluded: {', '.join(missing)}")
    # `missing` here means "not covered", not "would corrupt the table" — the table is explicitly
    # a partial-coverage analysis, so it is written regardless and records who was left out.
    out["_meta"]["models_without_resamples"] = missing
    write_table(OUT_PATH, out, [], allow_incomplete=args.allow_incomplete)

    s = out["summary"]
    print(f"\nMMLU only, {s['n_models']} models with resample data")
    print(f"  {'model':22} {'p_follow':>8} {'p_flip':>7} {'spont.':>7} "
          f"{'correct obs':>11} {'correct adj':>11} {'incorrect':>9} {'gap obs':>8} {'gap adj':>8}")
    for r in rows:
        print(f"  {r['model']:22} {r['p_follow_correct']:7.1%} {r['p_flip_plain']:6.1%} "
              f"{r['spontaneous_share_mean']:6.1%} {r['faith_correct_observed']:10.1%} "
              f"{r['faith_correct_adjusted']:10.1%} {r['faith_incorrect']:8.1%} "
              f"{r['gap_observed']*100:+6.1f}pp {r['gap_adjusted']*100:+6.1f}pp"
              f"{'' if r['survives'] else '   <- REVERSES'}")
    print(f"\n  gap survives the adjustment for {s['n_survive']}/{s['n_models']} models")
    print(f"  median gap {s['median_gap_observed']*100:+.1f}pp observed -> "
          f"{s['median_gap_adjusted']*100:+.1f}pp adjusted")
    print(f"  median share of correct-hint follows that are spontaneous: {s['median_spontaneous_share']:.1%}")
    mg = [r for r in rows if r.get("mentions_gap_adjusted") is not None]
    if mg:
        print(f"\n  mention-only metric, same adjustment: gap survives for "
              f"{s['n_survive_mentions_only']}/{s['n_with_mentions']} models "
              f"(median {statistics.median(r['mentions_gap_adjusted'] for r in mg)*100:+.1f}pp)")
        worst = min(mg, key=lambda r: r["mentions_gap_adjusted"])
        print(f"    tightest: {worst['model']} — correct {worst['mentions_correct_observed']:.1%} "
              f"-> {worst['mentions_correct_adjusted']:.1%} adjusted vs incorrect "
              f"{worst['mentions_incorrect']:.1%}")


if __name__ == "__main__":
    main()
