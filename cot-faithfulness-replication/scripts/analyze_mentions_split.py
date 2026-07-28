"""Mentions-only robustness check for the faithfulness-by-hint-direction result.

The full metric (`verbalized`) requires the judge to affirm that the CoT DEPENDED on the hint,
a counterfactual that is intrinsically harder to affirm when the hint points at the correct
answer (the model might simply have re-derived it), and the judge prompt resolves uncertainty
toward false. That asymmetry could manufacture a correct-vs-incorrect faithfulness gap on its
own. The `mentions_hint` verdict has no such asymmetry: it only asks whether the CoT refers to
the hint at all. This script recomputes the gap on mentions alone, from the committed judge
verdict files (no transcripts or API needed), writing results/mentions_split.json.

  python scripts/analyze_mentions_split.py
"""

import glob
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

OUT_PATH = Path("results/mentions_split.json")

MMLU_RE = re.compile(r"results/judge_tier[12]_(.+)_(full|standard|std250)\.jsonl$")
GPQA_RE = re.compile(r"results/judge_gpqa_tier[12]_(.+)\.jsonl$")


def ingest(path: str, dataset: str, tag: str, acc: dict) -> None:
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out = r.get("output") or {}
        if out.get("mentions_hint") is None:  # judge parse failures are excluded, as in the main metric
            continue
        cond = r["task_id"].split("|")[1]
        direction = "correct" if cond.endswith("_True") else "incorrect"
        cell = acc.setdefault((tag, dataset, direction), {"mentions": 0, "verbalized": 0, "n": 0})
        cell["n"] += 1
        cell["mentions"] += bool(out.get("mentions_hint"))
        cell["verbalized"] += bool(out.get("verbalized"))


def main() -> None:
    acc: dict = {}
    for p in sorted(glob.glob("results/judge_tier*_*.jsonl")):
        m = MMLU_RE.match(p)
        if m:
            ingest(p, "mmlu", m.group(1), acc)
    for p in sorted(glob.glob("results/judge_gpqa_tier*_*.jsonl")):
        m = GPQA_RE.match(p)
        if not m:
            continue
        tag = m.group(1)
        if tag.endswith("_t0"):  # the R1 temp-0 judge-dependence anchor; the lineup basis is t1
            continue
        ingest(p, "gpqa", tag.removesuffix("_t1"), acc)

    models = sorted({k[0] for k in acc})
    table, gaps_m, gaps_v = {}, [], []
    for tag in models:
        entry = {}
        for direction in ("correct", "incorrect"):
            rates_m, rates_v, per_ds = [], [], {}
            for ds in ("mmlu", "gpqa"):
                cell = acc.get((tag, ds, direction))
                if not cell or cell["n"] == 0:
                    continue
                per_ds[ds] = {
                    "n_judged": cell["n"],
                    "mentions_pct": 100 * cell["mentions"] / cell["n"],
                    "verbalized_pct": 100 * cell["verbalized"] / cell["n"],
                }
                rates_m.append(per_ds[ds]["mentions_pct"])
                rates_v.append(per_ds[ds]["verbalized_pct"])
            if not rates_m:
                continue
            entry[direction] = {
                **per_ds,
                "avg_mentions_pct": sum(rates_m) / len(rates_m),
                "avg_verbalized_pct": sum(rates_v) / len(rates_v),
            }
        if len(entry) < 2:
            continue
        entry["gap_mentions_pp"] = entry["incorrect"]["avg_mentions_pct"] - entry["correct"]["avg_mentions_pct"]
        entry["gap_verbalized_pp"] = entry["incorrect"]["avg_verbalized_pct"] - entry["correct"]["avg_verbalized_pct"]
        gaps_m.append(entry["gap_mentions_pp"])
        gaps_v.append(entry["gap_verbalized_pp"])
        table[tag] = entry

    out = {
        "_meta": {
            "generated_by": "scripts/analyze_mentions_split.py",
            "description": "Mention rate vs full verbalized rate among followed-hint pairs, by hint "
                           "direction, per dataset plus the equal-weight average. Robustness check: "
                           "the incorrect-vs-correct gap on mentions alone needs no counterfactual "
                           "dependence judgment.",
            "n_models": len(table),
            "n_models_gap_positive_mentions": sum(1 for g in gaps_m if g > 0),
            "median_gap_mentions_pp": statistics.median(gaps_m),
            "median_gap_verbalized_pp": statistics.median(gaps_v),
        },
        "models": table,
    }
    OUT_PATH.write_text(json.dumps(out, indent=1) + "\n")
    meta = out["_meta"]
    print(f"wrote {OUT_PATH}")
    print(f"{meta['n_models_gap_positive_mentions']}/{meta['n_models']} models have incorrect > correct on mentions alone")
    print(f"median gap: mentions {meta['median_gap_mentions_pp']:+.1f}pp, verbalized {meta['median_gap_verbalized_pp']:+.1f}pp")


if __name__ == "__main__":
    main()
