"""§6 layout re-judge: confirm the per-feature judge labels survive layout normalization.

Phase 1 judged the PARAGRAPHED text, then `data_checks.normalize_layout` collapsed prose to one paragraph
and never re-judged the collapsed version. The change is whitespace-only, so labels should hold — but it
is the load-bearing validation and was unconfirmed. This re-judges a covering sample of the FINAL
(normalized) texts with the SAME blind Sonnet-4.6 judge and the SAME per-property settings used at
generation (so the only difference is the collapsed layout), on BOTH features, and reports per-property /
per-cell agreement with the committed label. It prioritizes the held-out cells (they carry the signal)
and the layout-leaning properties (argumentative_structure, narrative_vs_expository, question_vs_statement,
numerical_density, specificity). Cached via llm.py; `--assert-cached` makes a rerun free.

Usage:
  PYTHONPATH=$PWD/lib python experiments/dataset/run_layout_rejudge.py                 # judge (cached) + write report
  PYTHONPATH=$PWD/lib python experiments/dataset/run_layout_rejudge.py --assert-cached # verify fully cached (0 new calls)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random

import config
import dataset
import llm
import validation
from cost_tracker import CostTracker
from properties import PROPERTIES

LAYOUT_SENSITIVE = ["argumentative_structure", "narrative_vs_expository", "question_vs_statement",
                    "numerical_density", "specificity"]

# 8 held-cell pairs cover all 10 properties on BOTH signs (each held cell carries one feature at + and the
# other at -); specificity (the fuzziest) and the other layout-leaning props land on several pairs.
HELD_PAIRS = [
    "argumentative_structure__specificity",   # arg ± , spec ±
    "narrative_vs_expository__hedging",        # narr ± , hedging ±
    "question_vs_statement__specificity",      # question ± , spec ±
    "numerical_density__hedging",              # numerical ± , hedging ±
    "specificity__temporal_scope",             # spec ± , temporal ±
    "code_presence__hedging",                  # code ± , hedging ±
    "imperative_mood__specificity",            # imperative ± , spec ±
    "domain_jargon__hedging",                  # jargon ± , hedging ±
]
# Plus a few training (A) cells for the most layout-leaning + values, where the generator most likely used
# paragraph-per-point / scene breaks / a question-per-line / a figure burst that normalization collapses.
EXTRA_A_CELLS = [(pk, "A") for pk in [
    "argumentative_structure__specificity", "narrative_vs_expository__hedging",
    "question_vs_statement__specificity", "numerical_density__hedging"]]

# --- stage 6 expansion (INSTRUCTIONS §4.6): the 10 ADOPTED properties were never layout-re-judged
# (the side run judged pre-normalization). 10 held-cell pairs cover every new property on BOTH signs,
# each against an original-roster partner; A-cells added for the new values most likely to have used
# list/line layout the normalization collapses (short simple sentences, parens, negation scaffolds,
# acronym-dense citations).
HELD_PAIRS_EXPANSION = [
    "hedging__second_person_address",           # second_person ±
    "temporal_scope__first_person_voice",       # first_person ±
    "code_presence__figurative_language",       # figurative ±
    "narrative_vs_expository__emotional_tone",  # emotional ±
    "specificity__formal_register",             # formal ±
    "domain_jargon__sentence_complexity",       # complexity ±
    "question_vs_statement__passive_voice",     # passive ±
    "imperative_mood__negation_density",        # negation ±
    "question_vs_statement__acronym_density",   # acronym ±
    "argumentative_structure__parenthetical_asides",  # parenthetical ±
]
EXTRA_A_CELLS_EXPANSION = [(pk, "A") for pk in [
    "domain_jargon__sentence_complexity", "argumentative_structure__parenthetical_asides",
    "imperative_mood__negation_density", "question_vs_statement__acronym_density"]]
NEW_PROPERTIES = ["second_person_address", "first_person_voice", "figurative_language", "emotional_tone",
                  "formal_register", "sentence_complexity", "passive_voice", "negation_density",
                  "acronym_density", "parenthetical_asides"]


def build_targets(scope: str = "original") -> list[tuple[str, str]]:
    held, extra = ((HELD_PAIRS_EXPANSION, EXTRA_A_CELLS_EXPANSION) if scope == "expansion"
                   else (HELD_PAIRS, EXTRA_A_CELLS))
    if scope == "expansion":  # every new property must be exercised on BOTH signs by the held cells
        covered = {p for pk in held for p in pk.split("__")}
        missing = [p for p in NEW_PROPERTIES if p not in covered]
        assert not missing, f"expansion re-judge targets miss new properties: {missing}"
    return [(pk, c) for pk in held for c in ("held_f1pos", "held_f1neg")] + extra


async def rejudge(r: dict) -> tuple[dict, dict]:
    f1p, f2p = PROPERTIES[r["f1"]], PROPERTIES[r["f2"]]
    return await validation.judge_feature(r["text"], f1p), await validation.judge_feature(r["text"], f2p)


async def main(per_cell: int, assert_cached: bool, scope: str = "original"):
    tracker = CostTracker(config.COST_FILE, run_description=f"run_layout_rejudge (per_cell={per_cell}, {scope})")
    llm.set_cost_tracker(tracker)
    llm.assert_cached = assert_cached

    sample = []  # (pair, cell, record)
    for pk, cell in build_targets(scope):
        recs = dataset.load_cell(pk, cell)
        rng = random.Random(f"layout-rejudge/{pk}/{cell}")  # per-cell seed: stable if other cells change
        sample += [(pk, cell, r) for r in rng.sample(recs, min(per_cell, len(recs)))]

    print(f"re-judging {len(sample)} normalized texts x 2 features = {2*len(sample)} judge calls "
          f"(Sonnet-4.6){' [assert-cached]' if assert_cached else ''}")
    judged = await asyncio.gather(*[rejudge(r) for _, _, r in sample])

    rows = []
    for (pk, cell, r), (nf1, nf2) in zip(sample, judged):
        for which, newj in (("f1", nf1), ("f2", nf2)):
            committed = r["validation"][f"{which}_label"]
            rows.append({
                "pair": pk, "cell": cell, "candidate_index": r["candidate_index"], "seed": r["seed"],
                "property": r[which], "intended_sign": r[f"{which}_sign"], "committed_label": committed,
                "rejudge_label": newj["label"], "agree": newj["label"] == committed,
                "rejudge_reason": newj["reason"],
            })
    suffix = "" if scope == "original" else f"_{scope}"
    (config.RESULTS_DIR / f"layout_rejudge{suffix}.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    _write_report(rows, per_cell, suffix)
    print(f"run cost ${tracker.run_cost:.2f}")


def _write_report(rows: list[dict], per_cell: int, suffix: str = ""):
    from collections import defaultdict

    n = len(rows)
    n_agree = sum(r["agree"] for r in rows)
    disagree = [r for r in rows if not r["agree"]]

    by_prop = defaultdict(lambda: [0, 0])  # property -> [agree, total]
    by_cellkind = defaultdict(lambda: [0, 0])  # held vs A
    for r in rows:
        by_prop[r["property"]][0] += r["agree"]; by_prop[r["property"]][1] += 1
        kind = "held" if r["cell"].startswith("held") else "train(A)"
        by_cellkind[kind][0] += r["agree"]; by_cellkind[kind][1] += 1

    lines = [
        "# Layout re-judge: do the per-feature labels survive layout normalization? (INSTRUCTIONS §6)\n",
        "Same blind Sonnet-4.6 judge and per-property settings as generation; only the text differs "
        "(final one-paragraph normalized text vs the paragraphed text judged at accept time). `agree` = the "
        "re-judge label equals the committed label.\n",
        f"\n**Overall agreement: {n_agree}/{n} = {n_agree/n:.1%}** across {n//2} texts x 2 features, "
        f"{per_cell}/cell over {len({(r['pair'],r['cell']) for r in rows})} cells (held-out prioritized).\n",
        "\n## Agreement by property (layout-leaning original properties marked *)\n",
        "| property | agree / total | rate |", "|---|---|---|"]
    for p in sorted(by_prop, key=lambda p: by_prop[p][0] / by_prop[p][1]):
        a, t = by_prop[p]
        star = " *" if p in LAYOUT_SENSITIVE else ""
        lines.append(f"| {p}{star} | {a}/{t} | {a/t:.1%} |")
    lines.append("\n## Agreement by cell kind\n")
    lines.append("| cell kind | agree / total | rate |")
    lines.append("|---|---|---|")
    for k, (a, t) in by_cellkind.items():
        lines.append(f"| {k} | {a}/{t} | {a/t:.1%} |")

    lines.append(f"\n## Disagreements ({len(disagree)})\n")
    if disagree:
        lines.append("| pair | cell | idx | property | committed | re-judge | re-judge reason |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in disagree:
            lines.append(f"| {r['pair']} | {r['cell']} | {r['candidate_index']} | {r['property']} | "
                         f"{r['committed_label']} | {r['rejudge_label']} | {r['rejudge_reason'][:90]} |")
    else:
        lines.append("_None — every sampled label survived normalization._")
    inspect_dir = config.RESULTS_DIR / "inspect"
    inspect_dir.mkdir(parents=True, exist_ok=True)
    (inspect_dir / f"layout_rejudge{suffix}.md").write_text("\n".join(lines))
    print(f"\noverall agreement {n_agree}/{n} = {n_agree/n:.1%}; disagreements: {len(disagree)}")
    for r in disagree:
        print(f"  DISAGREE {r['pair']}/{r['cell']} idx={r['candidate_index']} {r['property']}: "
              f"committed={r['committed_label']} rejudge={r['rejudge_label']} | {r['rejudge_reason'][:80]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=20)
    ap.add_argument("--assert-cached", action="store_true")
    ap.add_argument("--scope", choices=["original", "expansion"], default="original",
                    help="expansion = covering sample over the 10 ADOPTED properties (both signs, "
                    "held-prioritized); writes layout_rejudge_expansion.* (originals untouched)")
    args = ap.parse_args()
    asyncio.run(main(args.per_cell, args.assert_cached, args.scope))
