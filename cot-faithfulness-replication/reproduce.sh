#!/usr/bin/env bash
# Reproduce the figures of "Hint-based CoT faithfulness evals still mostly work on Claude".
#
# Stage A (default)          — redraw all 9 figures from the committed results tables.
#                              No API keys, no downloads; runs in ~a minute.
# Stage B (./reproduce.sh full) — full regeneration from scratch: download the released
#                              dataset, re-collect all transcripts, re-judge, re-analyze,
#                              then redraw. Needs ANTHROPIC_API_KEY + OPENROUTER_API_KEY.
#                              Cost ballpark: roughly $3,000-5,000 in API spend (transcripts
#                              for 30 models on two datasets + Opus 4.8 judging; the original
#                              project, which also ran experiments not in the post, totaled
#                              ~$8,700). Every step resumes from existing rows + a response
#                              cache, so interrupting and re-running is safe.
#
# Middle path (no API keys, recompute the analysis tables from raw data): fetch the raw
# transcripts with `python data/download_transcripts.py`, then run the "analyze" and
# "figures" blocks below.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
STAGE="${1:-figures}"

figures() {
  echo "== Stage A: figures from the committed results tables =="
  $PY scripts/plot_following_sonnets.py     # fig_following_sonnets
  $PY scripts/plot_lineups.py               # fig_following_main, fig_following_other, fig_faith_claude, fig_faith_other
  $PY scripts/plot_sonnet45.py              # fig_sonnet45_following, fig_sonnet45_faithfulness
  $PY scripts/plot_gpqa_vs_mmlu.py          # fig_gpqa_vs_mmlu_following
  $PY scripts/plot_judge_dependence.py      # fig_judge_dependence
  echo "== all figures written to figures/ =="
}

analyze() {
  echo "== Recompute the results tables from raw transcripts + judge verdicts =="
  # This one runs offline (all of its raw inputs are committed); the other three need the
  # transcripts, and refuse to overwrite a committed table when those are missing.
  $PY scripts/analyze_judge_dependence.py   # results/judge_dependence.json
  $PY scripts/analyze_mentions_split.py     # results/mentions_split.json (offline: judge files are committed)
  $PY scripts/analyze_following.py          # results/following_tables.json
  $PY scripts/analyze_faithfulness.py       # results/faithfulness_tables.json
  $PY scripts/analyze_sonnet45.py           # results/sonnet45_detail.json
}

full() {
  echo "== Stage B: full regeneration from scratch =="
  : "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY}"
  : "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"

  echo "-- 1. datasets: released hint files (Google Drive, ~350 MB) + GPQA-Diamond --"
  $PY data/download_dataset.py
  $PY data/download_gpqa.py

  echo "-- 2. transcripts: 10 Claude models (MMLU + GPQA) --"
  # Sonnet 4.5's MMLU data is the full 2,994-question pool (it carries the per-hint-type figure);
  # every other Claude model uses the 500-question standard pool. See lib/sweep.py's `pool` field.
  for MODEL in claude-opus-4-1-20250805 claude-haiku-4-5-20251001 \
               claude-opus-4-5-20251101 claude-opus-4-6 claude-sonnet-4-6 claude-opus-4-7 \
               claude-opus-4-8 claude-fable-5 claude-sonnet-5; do
    $PY scripts/run_mmlu.py --model "$MODEL" --pool standard
    $PY scripts/run_mmlu.py --model "$MODEL" --pool standard --tier2
    $PY scripts/run_gpqa.py --model "$MODEL"
    $PY scripts/run_gpqa.py --model "$MODEL" --tier2
  done
  $PY scripts/run_mmlu.py --model claude-sonnet-4-5-20250929 --pool full
  $PY scripts/run_mmlu.py --model claude-sonnet-4-5-20250929 --pool full --tier2
  $PY scripts/run_gpqa.py --model claude-sonnet-4-5-20250929
  $PY scripts/run_gpqa.py --model claude-sonnet-4-5-20250929 --tier2

  echo "-- 3. transcripts: 6 open-weight models (OpenRouter; GPQA at temperature 1) --"
  for MODEL in deepseek/deepseek-r1 qwen/qwen3-235b-a22b-thinking-2507 openai/gpt-oss-120b \
               deepseek/deepseek-v3.2 moonshotai/kimi-k2.5 z-ai/glm-5.2; do
    $PY scripts/run_mmlu.py --model "$MODEL" --pool standard
    $PY scripts/run_mmlu.py --model "$MODEL" --pool standard --tier2
    $PY scripts/run_gpqa.py --model "$MODEL" --temperature 1
    $PY scripts/run_gpqa.py --model "$MODEL" --temperature 1 --tier2
  done

  echo "-- 4. transcripts: DeepSeek R1 temperature-0 rerun (the judge-dependence data) --"
  $PY scripts/run_mmlu.py --model deepseek/deepseek-r1 --pool standard --temperature 0
  $PY scripts/run_mmlu.py --model deepseek/deepseek-r1 --pool standard --temperature 0 --tier2
  $PY scripts/run_gpqa.py --model deepseek/deepseek-r1 --temperature 0
  $PY scripts/run_gpqa.py --model deepseek/deepseek-r1 --temperature 0 --tier2

  echo "-- 5. transcripts: 14 closed GPT/Gemini models (MMLU first 250 standard-pool questions + GPQA) --"
  for SHORT in gpt-5 gpt-5-mini gpt-5-nano gpt-5.1 gpt-5.2 gpt-5.4 gpt-5.5 \
               gpt-5.6-luna gpt-5.6-sol gpt-5.6-terra; do
    MODEL="openai/$SHORT"
    $PY scripts/run_mmlu.py --model "$MODEL" --pool standard --n-questions 250 --out "results/tier1_${SHORT}_std250.jsonl"
    $PY scripts/run_mmlu.py --model "$MODEL" --pool standard --n-questions 250 --tier2 --out "results/tier2_${SHORT}_std250.jsonl"
    $PY scripts/run_gpqa.py --model "$MODEL"
    $PY scripts/run_gpqa.py --model "$MODEL" --tier2
  done
  for PAIR in "google/gemini-3.1-pro-preview gemini-3.1-pro" \
              "google/gemini-3.1-flash-lite-preview gemini-3.1-flash-lite" \
              "google/gemini-3.5-flash gemini-3.5-flash" \
              "google/gemini-3.6-flash gemini-3.6-flash"; do
    set -- $PAIR; MODEL=$1; SHORT=$2
    $PY scripts/run_mmlu.py --model "$MODEL" --pool standard --n-questions 250 --out "results/tier1_${SHORT}_std250.jsonl"
    $PY scripts/run_mmlu.py --model "$MODEL" --pool standard --n-questions 250 --tier2 --out "results/tier2_${SHORT}_std250.jsonl"
    $PY scripts/run_gpqa.py --model "$MODEL"
    $PY scripts/run_gpqa.py --model "$MODEL" --tier2
  done

  echo "-- 6. judging (Claude Opus 4.8 verbalization judge, MMLU + GPQA, all 30 models) --"
  # MMLU: judge every collected tier1/tier2 file. GPQA: judge every collected grid.
  for T1 in results/tier1_*.jsonl; do
    STEM=${T1#results/tier1_}
    $PY scripts/judge_mmlu_tier1.py "$T1"
    $PY scripts/judge_mmlu_tier2.py "results/tier2_${STEM}" --baseline "$T1"
  done
  for G1 in results/gpqa_tier1_*.jsonl; do
    TAG=${G1#results/gpqa_tier1_}; TAG=${TAG%.jsonl}
    $PY scripts/judge_gpqa.py --tag "$TAG" --tier 1
    $PY scripts/judge_gpqa.py --tag "$TAG" --tier 2
  done

  echo "-- 7. era-matched judge (Claude 3 Opus) on the R1 temperature-0 transcripts --"
  $PY scripts/judge_era_mmlu.py --temp t0
  $PY scripts/judge_gpqa.py --tag deepseek-r1_t0 --tier 1 --variant model3opus_std
  $PY scripts/judge_gpqa.py --tag deepseek-r1_t0 --tier 2 --variant model3opus_std

  analyze
  figures
}

case "$STAGE" in
  figures) figures ;;
  analyze) analyze; figures ;;
  full) full ;;
  *) echo "usage: ./reproduce.sh [figures|analyze|full]"; exit 1 ;;
esac
