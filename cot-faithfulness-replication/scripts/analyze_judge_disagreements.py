"""Judge-disagreement breakdown: Claude Opus 4.8 vs the era-matched Claude 3 Opus.

Both judges scored the same DeepSeek R1 temperature-0 transcripts with the same prompt, so
their disagreements isolate how much dependence each judge demands before calling a CoT
faithful. Runs offline from the committed judge verdict files; writes
results/judge_disagreements.json (counts + the task ids the post's appendix examples quote).

  python scripts/analyze_judge_disagreements.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

OUT_PATH = Path("results/judge_disagreements.json")
PAIRS = [
    ("results/judge_tier1_deepseek-r1_t0_standard.jsonl",
     "results/judge_model3opus_std_tier1_deepseek-r1_t0_standard.jsonl"),
    ("results/judge_tier2_deepseek-r1_t0_standard.jsonl",
     "results/judge_model3opus_std_tier2_deepseek-r1_t0_standard.jsonl"),
]
# The three cases quoted in the post's appendix drop-downs, from the script that renders them.
from scripts.make_disagreement_examples import SHOWN as EXAMPLES  # noqa: E402


def load(path: str) -> dict:
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if (r.get("output") or {}).get("verbalized") is not None:
            out[r["task_id"]] = r["output"]
        
    return out


def main() -> None:
    matrix, era_pattern, by_type, examples = Counter(), Counter(), Counter(), {}
    n_both = 0
    for f48, fera in PAIRS:
        a, b = load(f48), load(fera)
        for tid in a.keys() & b.keys():
            n_both += 1
            v48, vera = a[tid]["verbalized"], b[tid]["verbalized"]
            matrix[f"opus48={v48},era={vera}"] += 1
            if v48 and not vera:
                era_pattern[f"mentions={b[tid]['mentions_hint']},uses={b[tid]['uses_hint_to_answer']}"] += 1
                by_type[tid.split("|")[1].rsplit("_", 1)[0]] += 1
            if tid in EXAMPLES:
                examples[tid] = {"opus48": a[tid], "claude3opus": b[tid]}

    out = {
        "_meta": {
            "generated_by": "scripts/analyze_judge_disagreements.py",
            "description": "Per-transcript agreement between the Opus 4.8 judge and the era-matched "
                           "Claude 3 Opus judge on the same DeepSeek R1 temp-0 transcripts.",
            "n_scored_by_both": n_both,
            "n_disagreements": sum(v for k, v in matrix.items() if k.split(",")[0][-4:] != k.split(",")[1][-4:]),
        },
        # sorted so the committed JSON is byte-stable across runs (Counter order is not)
        "agreement_matrix": dict(sorted(matrix.items())),
        "era_verdict_pattern_when_opus48_faithful": dict(sorted(era_pattern.items())),
        "opus48_faithful_era_not_by_hint_type": dict(sorted(by_type.items())),
        "appendix_examples": examples,
    }
    missing = [t for t in EXAMPLES if t not in examples]
    assert not missing, f"appendix examples no longer disagreements: {missing}"
    OUT_PATH.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT_PATH}")
    print("agreement matrix:", dict(matrix))
    print("era pattern when Opus 4.8 says faithful:", dict(era_pattern))


if __name__ == "__main__":
    main()
