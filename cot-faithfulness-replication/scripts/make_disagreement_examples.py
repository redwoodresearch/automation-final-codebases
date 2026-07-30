"""Emit the post's judge-disagreement appendix: full DeepSeek R1 transcripts with both verdicts.

SHOWN pins the three transcripts the post shows, by task id, so this script reproduces the
appendix exactly. They were originally picked by taking the median-length disagreement (by total
chain-of-thought characters) within a hint type and disagreement direction, so each is typical of
its cell rather than chosen for effect; the set spans a structural hint, a social hint, both hint
directions, and both directions of judge disagreement. It is not a representative sample of all
the disagreements — some cells are small. Pass --all to emit the median-length example for every
(hint type, hint direction) cell instead.

  python scripts/make_disagreement_examples.py > /tmp/disagreements.md
"""

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

TIERS = [
    ("results/tier1_deepseek-r1_t0_standard.jsonl.gz",
     "results/judge_tier1_deepseek-r1_t0_standard.jsonl",
     "results/judge_model3opus_std_tier1_deepseek-r1_t0_standard.jsonl"),
    ("results/tier2_deepseek-r1_t0_standard.jsonl.gz",
     "results/judge_tier2_deepseek-r1_t0_standard.jsonl",
     "results/judge_model3opus_std_tier2_deepseek-r1_t0_standard.jsonl"),
]
DISPLAY = {"suggestion": "Sycophancy", "posthoc": "Consistency", "fewshot_symbol": "Visual marker",
           "metadata": "Metadata answer key", "grader_hacking": "Leaked grader code",
           "unethical_information": "Unauthorized access"}
# (hint type, arm, direction) triples shown in the post: a structural hint on the well-powered
# incorrect-hint side, a social hint on the correct-hint side (where the dependence call is most
# ambiguous), and one case running the other way. "True" = hint points at the correct answer.
SHOWN = [
    "judge|grader_hacking_False|2391|0",  # leaked grader, incorrect hint; Opus 4.8 faithful, era judge not
    "judge|suggestion_False|2808|0",      # sycophancy, incorrect hint; Opus 4.8 faithful, era judge not
    "judge|metadata_True|113|0",          # metadata, correct hint; era judge faithful, Opus 4.8 not
]


def load_judge(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if (r.get("output") or {}).get("verbalized") is not None:
            out[r["task_id"]] = r["output"]
    return out


def load_transcripts(path):
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            # transcript ids look like "<model>|<condition>|<idx>|<sample>|t0";
            # judge ids look like "judge|<condition>|<idx>|<sample>"
            parts = r["task_id"].split("|")
            out[f"judge|{parts[1]}|{parts[2]}|{parts[3]}"] = r
    return out


def cot_parts(rec):
    msg = rec["output"]["raw_response"]["choices"][0]["message"]
    return (msg.get("reasoning") or "").strip(), (msg.get("content") or "").strip()


def verdict_line(name, v):
    tag = "faithful" if v["verbalized"] else "not faithful"
    return (f"**{name}** — {tag} (mentions the hint: {str(v['mentions_hint']).lower()}, "
            f"depends on it: {str(v['uses_hint_to_answer']).lower()})\n\n> {v['reasoning']}\n")


def main() -> None:
    cases = []
    for tpath, f48, fera in TIERS:
        tr, a, b = load_transcripts(tpath), load_judge(f48), load_judge(fera)
        for tid in a.keys() & b.keys() & tr.keys():
            if a[tid]["verbalized"] == b[tid]["verbalized"]:
                continue
            think, visible = cot_parts(tr[tid])
            cases.append({
                "tid": tid, "hint_type": tid.split("|")[1].rsplit("_", 1)[0],
                "arm_tag": tid.split("|")[1].rsplit("_", 1)[1],
                "direction": "opus48_faithful" if a[tid]["verbalized"] else "era_faithful",
                "length": len(think) + len(visible), "think": think, "visible": visible,
                "hint_letter": tr[tid]["input"].get("hint"), "answer": tr[tid]["output"].get("answer"),
                "v48": a[tid], "vera": b[tid],
            })

    def median_of(subset):
        s = sorted(subset, key=lambda c: c["length"])
        return s[len(s) // 2] if s else None

    by_tid = {c["tid"]: c for c in cases}

    def cell_size(c):
        return len([x for x in cases if x["hint_type"] == c["hint_type"]
                    and x["arm_tag"] == c["arm_tag"] and x["direction"] == c["direction"]])

    picked = []
    if "--all" in sys.argv:
        for ht in DISPLAY:
            for arm in ("False", "True"):
                cell = [c for c in cases if c["hint_type"] == ht and c["arm_tag"] == arm
                        and c["direction"] == "opus48_faithful"]
                c = median_of(cell)
                if c:
                    picked.append({**c, "cell_size": len(cell)})
    else:
        for tid in SHOWN:
            c = by_tid.get(tid)
            assert c is not None, f"{tid} is no longer a judge disagreement; re-pick SHOWN"
            picked.append({**c, "cell_size": cell_size(c)})

    print(f"<!-- generated by scripts/make_disagreement_examples.py from {len(cases)} disagreements -->")
    for c in picked:
        arm = "incorrect" if c["tid"].split("|")[1].endswith("_False") else "correct"
        flip = "Claude 3 Opus called it faithful and Opus 4.8 did not" if c["direction"] == "era_faithful" \
            else "Opus 4.8 called it faithful and Claude 3 Opus did not"
        print(f"\n<details>\n<summary>{DISPLAY[c['hint_type']]}, {arm} hint — {flip}</summary>\n")
        n = c.get("cell_size", 0)
        pool = f"the median-length of the {n} disagreement{'s' if n != 1 else ''} in this cell"
        print(f"The hint points at ({c['hint_letter']}); R1 answered ({c['answer']}). "
              f"This is {pool}.\n")
        print("**R1's reasoning:**\n")
        print("> " + c["think"].replace("\n\n", "\n").replace("\n", "\n> ") + "\n")
        if c["visible"]:
            print("**R1's visible response:**\n")
            print("> " + c["visible"].replace("\n\n", "\n").replace("\n", "\n> ") + "\n")
        print(verdict_line("Claude Opus 4.8", c["v48"]))
        print(verdict_line("Claude 3 Opus", c["vera"]))
        print("</details>")


if __name__ == "__main__":
    main()
