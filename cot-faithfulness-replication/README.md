# Hint-based CoT faithfulness evals still mostly work on Claude

Code and data for the blog post **"Hint-based CoT faithfulness evals still mostly work on
Claude"** (Eric Gan, Redwood Research, 2026) — a replication and extension of
[Chen et al. 2025, *Reasoning Models Don't Always Say What They Think*](https://arxiv.org/abs/2505.05410).

Models are asked MMLU / GPQA-Diamond questions with a hint embedded in the prompt (e.g. a
leaked answer key) pointing at a specific option. We measure, across 30 models
(10 Claude, 6 open-weight reasoners, 10 GPT, 4 Gemini):

1. **Hint-following** — how often the model changes its answer to the hinted option,
   split by whether the hint points at the correct or an incorrect answer;
2. **Faithfulness** — when the model follows the hint, how often its chain of thought
   admits it (scored by a Claude Opus 4.8 judge);
3. **A DeepSeek R1 non-replication** — R1 follows incorrect hints far less than Chen et al.
   reported, and its measured faithfulness depends heavily on which judge model reads it.
   To test that, the same transcripts are re-judged by Claude 3 Opus — a model from roughly
   the same era as the judge Chen et al. used ("era-matched") — which reports much less
   faithfulness than Claude Opus 4.8 does. The measurement is strongly judge-dependent;
   without ground-truth labels this does not by itself establish which judge is more accurate.

## Reproduce the figures (no API keys needed)

```bash
uv sync            # or: python -m venv .venv && .venv/bin/pip install aiofiles anthropic attrs httpx matplotlib numpy tqdm gdown pytest
./reproduce.sh     # writes all 9 post figures to figures/
```

This redraws every figure in the post from the committed results tables (`results/*.json`).
The tables were produced by the `scripts/analyze_*.py` scripts from the raw transcripts;
the judge-dependence analysis (`scripts/analyze_judge_dependence.py`) can be re-run offline
too, because its raw inputs (the DeepSeek R1 temperature-0 transcripts and the eight judge
verdict files they need — 2 judges x 2 datasets x 2 hint tiers) are committed.

Other stages:

```bash
python data/download_transcripts.py   # fetch all raw transcripts (~2.5 GB; no API keys; needs `pip install huggingface_hub`)
./reproduce.sh analyze                # recompute results/*.json from raw transcripts, then redraw
ANTHROPIC_API_KEY=... OPENROUTER_API_KEY=... ./reproduce.sh full   # regenerate EVERYTHING (~$3-5k API spend)
```

The raw transcripts for all 30 models (~2.6 GB) are hosted as a Hugging Face dataset
([ejcgan/hint-faithfulness-transcripts](https://huggingface.co/datasets/ejcgan/hint-faithfulness-transcripts));
`data/download_transcripts.py` fetches and unpacks them into `results/` (requires
`pip install huggingface_hub`). `./reproduce.sh analyze` needs them for three of the four
tables; the judge-dependence table's raw inputs are committed.

`reproduce.sh full` documents the exact collection/judging sequence; every step resumes from
existing output rows plus a content-addressed response cache, so it is safe to interrupt.

## The experiment, in brief

Each question is asked twice: plain (the *unhinted* baseline answer `a_u`) and with one of
six hint types inserted (the *hinted* answer `a_h`). Following Chen et al., analysis is
restricted to *eligible* pairs (`a_u ≠ hint`), and each eligible pair is classified as
**change-to-hint** (`a_h = hint`), **change-to-non-hint**, or **no-change**. Faithfulness is
judged only on *retained* pairs (the model changed to the hint): the judge reads the model's
reasoning + visible response and decides whether the CoT verbalizes the hint (prompt in
`lib/judge.py`); the rate is normalized by the chance-switching correction α
(`lib/metrics.py`, Chen et al.'s formula).

The six hint types come in two groups (called **tier 1 / tier 2** throughout the code):

- **Tier 1 (released)** — sycophancy ("suggestion"), consistency ("posthoc"), and visual
  marker ("fewshot_symbol"): Anthropic released ready-to-run MMLU prompt files for these,
  which are used byte-verbatim (`data/download_dataset.py` fetches them).
- **Tier 2 (applied from the paper's Table 1)** — metadata answer key ("metadata"), leaked grader code
  ("grader_hacking"), and "unauthorized access" ("unethical_information"): no released
  files, so the paper's Table 1 templates are inserted into the plain questions
  (`lib/tier2.py`).

GPQA-Diamond has no released hint files at all, so all six hint types are assembled from the
same templates onto GPQA questions (`lib/gpqa.py`).

Model groups and their inference paths:

| group | models | API | reasoning channel |
|---|---|---|---|
| Claude | Opus 4.1 … Sonnet 5 (10) | Anthropic | native extended thinking (summarized on 4.7+) |
| open-weight | DeepSeek R1/V3.2, Qwen3-235B-Thinking, Kimi K2.5, GLM-5.2, gpt-oss-120b | OpenRouter (pinned provider) | raw chain of thought |
| closed frontier | GPT-5 family (10), Gemini (4) | OpenRouter (pinned provider) | vendor reasoning **summary** only |

## Map of the repo

```
lib/            importable core
  dataset.py      released-dataset loading (MMLU prompts, hint letters)
  tier1.py        run conditions, Anthropic request config, answer extraction
  tier2.py        the three Table 1 hint templates
  gpqa.py         GPQA-Diamond hint assembly (all six types)
  llm.py          cached/retrying Anthropic calls    openrouter.py  same for OpenRouter
  sweep.py        Claude + open-weight model registry  frontier.py  GPT/Gemini registry
  lineup.py       the combined 30-model lineup + result-file locations
  metrics.py      Chen et al. metrics (p, q, α, normalized faithfulness)
  analysis.py     transcript loading, (unhinted, hinted) pairing
  faithfulness.py per-(hint type × direction) faithfulness cells
  faith_band.py   normalized faithfulness for one judge's verdict set (judge-dependence)
  judge.py        the Opus 4.8 verbalization judge (prompt + parsing)
  judge_variants.py  the era-matched Claude 3 Opus judge
  figures.py      shared plot helpers (ordering, colors, committed-table loading)
scripts/        entry points (run from the repo root)
  run_mmlu.py / run_gpqa.py            collect transcripts (any of the 30 models)
  run_unhinted_resamples.py            extra unhinted samples on the correct-hint-eligible
                                       questions (the natural flip-to-correct baseline)
  judge_mmlu_tier1.py / judge_mmlu_tier2.py / judge_gpqa.py   Opus 4.8 judging
  judge_era_mmlu.py                    era-matched Claude 3 Opus judge (R1 MMLU);
                                       for GPQA use judge_gpqa.py --variant model3opus_std
  analyze_*.py                         raw data -> the committed results/*.json tables
                                       (analyze_mentions_split.py runs offline from the
                                       committed judge files: the mentions-only robustness check)
  plot_*.py                            results/*.json -> figures/*.png
  make_pools.py                        rebuild data/pools.json (the question pools)
  verify_dataset.py                    check the released dataset's structural invariants
data/           inputs
  download_dataset.py    fetch Anthropic's released CoT Faithfulness dataset (~350 MB)
  download_gpqa.py       fetch GPQA-Diamond from its official distribution (not re-hosted here)
  download_transcripts.py fetch the raw transcript archive (Hugging Face; ~2.5 GB)
  pools.json             the deduplicated question pools (standard=500, pilot=150, full=2,994)
  gpqa/                  GPQA-Diamond target dir (provenance + license: data/gpqa/README.md)
cost_tracker.py, file_cache.py, pricing/    vendored shared utilities: per-run API cost
                accounting (writes total_cost.jsonl) and the on-disk response cache the
                run/judge scripts use to make reruns free
results/        committed: the analysis tables (following_tables.json,
                faithfulness_tables.json, sonnet45_detail.json, judge_dependence.json,
                mentions_split.json, judge_disagreements.json, natural_flip.json,
                flip_confound.json, filtered_faithfulness.json),
                all judge verdict JSONLs (judge_*.jsonl), the DeepSeek R1 t=0 transcripts
                (*.jsonl.gz), and dataset_verification.md (the released dataset's structural
                invariants, from scripts/verify_dataset.py). Raw transcripts for the other
                models are gitignored — fetch or regenerate.
figures/        the 9 post figures (regenerable via ./reproduce.sh)
tests/          unit tests (pytest); dataset-dependent modules skip if the released
                dataset has not been downloaded
```

## Figures ↔ scripts

| post figure | script | reads |
|---|---|---|
| fig_following_sonnets | `plot_following_sonnets.py` | following_tables.json |
| fig_following_main, fig_following_other, fig_faith_claude, fig_faith_other | `plot_lineups.py` | following_tables.json, faithfulness_tables.json |
| fig_sonnet45_following, fig_sonnet45_faithfulness | `plot_sonnet45.py` | sonnet45_detail.json |
| fig_gpqa_vs_mmlu_following | `plot_gpqa_vs_mmlu.py` | following_tables.json |
| fig_judge_dependence | `plot_judge_dependence.py` | judge_dependence.json |

External reference values (Chen et al. 2025's published bars; Young 2026 / Chua & Evans 2025
per-type numbers) are hardcoded in the plot scripts with a source comment at each definition.

## Notes and caveats

- **Judged channel.** Older Claude models expose raw extended thinking; Claude 4.7+/5 and
  all GPT/Gemini models expose only a vendor-controlled summary. Faithfulness for the latter
  is judged on the summary + visible response and is not directly comparable to raw-CoT numbers.
- **Averaging.** Headline bars are the equal-weight (MMLU + GPQA)/2 average; per-model
  faithfulness is the 6-hint-type mean per dataset, then averaged across datasets.
- **Sonnet 4.5 gets its own two figures** because it is the model the Claude Sonnet 4.5 system
  card's faithfulness claim is about, so it is shown per hint type on the largest question pool.
- **Pools.** Claude/open-weight MMLU uses a deduplicated 500-question pool; Sonnet 4.5's MMLU
  data is instead the full 2,994-question pool; the GPT/Gemini group uses the first 250 pool questions;
  GPQA uses all 198 GPQA-Diamond questions. GPQA sampling: Claude models at their required
  thinking temperature (1), open-weight at t=1 (R1 additionally at t=0 for the
  judge-dependence study).
- Costs of every API run are logged to `total_cost.jsonl` (gitignored).
