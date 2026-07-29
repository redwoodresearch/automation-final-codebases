# Committed results

- `following_tables.json`, `faithfulness_tables.json`, `sonnet45_detail.json`,
  `judge_dependence.json` — the four aggregate tables the figures are drawn from
  (regenerate with `scripts/analyze_*.py`).
- `judge_*.jsonl` — per-transcript judge verdicts (all models/datasets/judges).
- `mentions_split.json` — mentions-only robustness check for the faithfulness-by-direction
  result (regenerate offline with `scripts/analyze_mentions_split.py`).
- `judge_disagreements.json` — Opus 4.8 vs era-matched Claude 3 Opus per-transcript agreement,
  plus the cases the post's appendix quotes (`scripts/analyze_judge_disagreements.py`).
- `*_deepseek-r1_t0.jsonl.gz` — the DeepSeek R1 temperature-0 transcripts (the raw inputs of
  the judge-dependence analysis).
- `dataset_verification.md` — structural checks over the released dataset.
- `true_arm_natural_flip.md` — measured no-hint flip-to-correct baselines (the correct-hint
  inflation caveat the post cites).

BENCHMARK DATA CANARY: the R1 transcript files (and the raw-transcript archive fetched by
`data/download_transcripts.py`) contain model reasoning that restates MMLU and GPQA-Diamond
question text. GPQA canary prefix: `gpqa:` (per-record canary strings in the upstream data).
Do not include these files in language-model training corpora.
