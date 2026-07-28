"""DeepSeek R1 judge-dependence numbers (temperature-0 MMLU + GPQA runs).

Two questions about the same transcripts:
1. Does R1 follow the incorrect hint? (judge-free change-to-hint decomposition, per dataset
   and the equal-weight MMLU+GPQA average)
2. When it follows, is the CoT judged faithful? Per-hint-type normalized faithfulness under
   two judges — Claude Opus 4.8 (this project's standard judge) and the era-matched Claude 3
   Opus (same prompt, no extended thinking) — per dataset and averaged.

Writes results/judge_dependence.json (committed). The R1 t=0 transcripts and all eight judge
verdict files (2 judges x 2 datasets x 2 hint tiers) are committed, so this analysis runs
offline from a fresh clone.

  python scripts/analyze_judge_dependence.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.analysis import build_pairs, load_results, results_file_exists
from lib.faith_band import load_verdicts, mean, type_normalized
from lib.faithfulness import TIER1_PAPER_TYPES, TIER2_TYPES
from lib.metrics import make_cell
from lib.tables import write_table

ALL_TYPES = TIER1_PAPER_TYPES + TIER2_TYPES
OUT_PATH = Path("results/judge_dependence.json")
R = Path("results")

TRANSCRIPTS = {
    "mmlu": (R / "tier1_deepseek-r1_t0_standard.jsonl", R / "tier2_deepseek-r1_t0_standard.jsonl"),
    "gpqa": (R / "gpqa_tier1_deepseek-r1_t0.jsonl", R / "gpqa_tier2_deepseek-r1_t0.jsonl"),
}
JUDGES = {
    "opus48": {
        "judge_model": "claude-opus-4-8",
        "mmlu": (R / "judge_tier1_deepseek-r1_t0_standard.jsonl",
                 R / "judge_tier2_deepseek-r1_t0_standard.jsonl"),
        "gpqa": (R / "judge_gpqa_tier1_deepseek-r1_t0.jsonl", R / "judge_gpqa_tier2_deepseek-r1_t0.jsonl"),
    },
    "claude3opus": {
        "judge_model": "claude-3-opus-20240229",
        "mmlu": (R / "judge_model3opus_std_tier1_deepseek-r1_t0_standard.jsonl",
                 R / "judge_model3opus_std_tier2_deepseek-r1_t0_standard.jsonl"),
        "gpqa": (R / "judge_model3opus_std_gpqa_tier1_deepseek-r1_t0.jsonl",
                 R / "judge_model3opus_std_gpqa_tier2_deepseek-r1_t0.jsonl"),
    },
}


def merged_rows(t1, t2):
    rows = dict(load_results(Path(t1)))
    for key, row in load_results(Path(t2)).items():
        rows.setdefault(key, row)
    return rows


def decomposition(rows, arm):
    trips = []
    for ht in ALL_TYPES:
        for p in build_pairs(rows, f"{ht}_{arm}"):
            if p.is_valid:
                trips.append((p.a_u, p.a_h, p.hint))
    c = make_cell(trips)
    n = c.n_eligible
    return {
        "n_eligible": n,
        "change_to_hint": c.n_switch_to_hint,
        "change_to_non_hint": c.n_switch_to_other,
        "no_change": n - c.n_switch_to_hint - c.n_switch_to_other,
    }


def main() -> None:
    needed = [p for paths in TRANSCRIPTS.values() for p in paths]
    needed += [p for cfg in JUDGES.values() for ds in TRANSCRIPTS for p in cfg[ds]]
    missing = [str(p) for p in needed if not results_file_exists(p)]
    assert not missing, f"missing committed input(s): {missing}"

    rows = {ds: merged_rows(*paths) for ds, paths in TRANSCRIPTS.items()}

    following = {ds: decomposition(rows[ds], "False") for ds in rows}
    pct = {ds: {k: 100 * v / following[ds]["n_eligible"]
                for k, v in following[ds].items() if k != "n_eligible"} for ds in following}
    following["avg_pct"] = {k: (pct["mmlu"][k] + pct["gpqa"][k]) / 2 for k in pct["mmlu"]}

    faith = {}
    for judge, cfg in JUDGES.items():
        per_ds = {}
        for ds in TRANSCRIPTS:
            verdicts = {**load_verdicts(cfg[ds][0]), **load_verdicts(cfg[ds][1])}
            per_ds[ds] = {}
            for ht in ALL_TYPES:
                d = type_normalized(rows[ds], verdicts, ht)
                per_ds[ds][ht] = {"normalized": d["normalized"], "raw": d["raw"],
                                  "n_judged": d["n_judged"], "n_retained": d["n_retained"]}
        avg = {ht: (None if per_ds["mmlu"][ht]["normalized"] is None or per_ds["gpqa"][ht]["normalized"] is None
                    else (per_ds["mmlu"][ht]["normalized"] + per_ds["gpqa"][ht]["normalized"]) / 2)
               for ht in ALL_TYPES}
        faith[judge] = {
            "judge_model": cfg["judge_model"],
            **per_ds,
            "avg_normalized": avg,
            "six_type_mean": mean(list(avg.values())),
        }
        fmt = lambda v: "-" if v is None else f"{100*v:.0f}"
        print(f"{judge:12s} avg per type: {[fmt(avg[ht]) for ht in ALL_TYPES]}  "
              f"mean={fmt(faith[judge]['six_type_mean'])}")

    out = {
        "_meta": {
            "generated_by": "scripts/analyze_judge_dependence.py",
            "description": "DeepSeek R1 (temp 0): incorrect-hint following decomposition per dataset "
                           "(counts + the equal-weight MMLU+GPQA percentage average), and per-hint-type "
                           "normalized faithfulness under the Opus 4.8 and era-matched Claude 3 Opus "
                           "judges. Hint-type order: " + ", ".join(ALL_TYPES),
        },
        "incorrect_hint_following": following,
        "faithfulness_by_judge": faith,
    }
    write_table(OUT_PATH, out, [])


if __name__ == "__main__":
    main()
