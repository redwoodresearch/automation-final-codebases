# Dataset: building the 2x2 feature grids

This experiment builds the data every other experiment reads: for each buildable pair of text features
(e.g. `code_presence__hedging`), a 2x2 grid of short passages. The two diagonal cells (A = both features
present, B = both absent) are the training cells; the two anti-diagonal cells (held_f1pos, held_f1neg)
carry exactly one feature each and are the held-out metric. Passages are LLM-generated, then a stronger
blind LLM judge confirms which features each one actually realizes, and a deterministic length/topic
match commits a balanced final set so length and topic carry no information about the class label. A
21st feature, `language` (Spanish vs English), is built the same way but screened separately because it
tokenizes longer.

The full study roster is 37 features across five types — `lib/features.py::ALL_FEATURES` is the
authoritative list. This experiment builds the original content/style pairs plus `language`; the roster
is then extended to the final 37 by the expansion builders in `lib/` (`extension_build.py`,
`content_build.py`, `newxnew_build.py`, `v2_build.py`), which reuse the same generate → blind-judge →
length-match steps described here.

Inputs: the feature/pair definitions in `lib/properties.py` (and `pairs_expanded.py`,
`language_property.py`, also in `lib/`), plus Anthropic API access (generation + judging) and a Modal token (the
baseline/robustness GPU steps). Outputs land in `data/` (the committed grids, buffers, derisk pools) and
`results/dataset/` (stats + QA reports); the lead figure lands in `plots/dataset/`. LLM calls are cached
(`--assert-cached` verifies a run is fully cached), so re-runs are cheap.

Entry points (run order documented in `reproduce.sh`):

| script | does | output |
|--------|------|--------|
| `run_baseline_prior.py` | pick the most balanced neutral label-token pair; un-fine-tuned baseline held-out logprob diff | `results/dataset/baseline_prior.json` |
| `run_buildability.py` | per-pair/per-cell accept-rate sweep: which feature pairs binarize and are buildable | `results/dataset/buildability_{cells,candidates}.jsonl`, `data/derisk/` |
| `run_generate.py` | final 300/cell grids for the test pairs | `data/grid_*.jsonl`, `results/dataset/generation_stats.json` |
| `run_generate_full.py` | over-generated buffer (>300/cell) for all locked pairs, so length-match can sub-sample | `data/buffer/`, `results/dataset/generation_stats_full.jsonl` |
| `run_length_match.py` | commit the dataset: balance A/B (full_balanced) or all four cells (tuned legacy) | `data/grid_*__{A,B}.jsonl`, `results/dataset/length_match_report.json`, `inspect/length_match.md` |
| `run_robustness.py` | does the fine-tune held-out logprob diff keep its sign across prompt wordings x label tokens | `results/dataset/robustness.json` |
| `run_layout_rejudge.py` | do the per-feature judge labels survive layout normalization (re-judge a covering sample) | `results/dataset/layout_rejudge.jsonl`, `inspect/layout_rejudge.md` |
| `run_language_screen.py` | buildability + Spanish/English token blow-up for `language` x 5 partners | `results/dataset/language_screen.json` |
| `run_language_generate.py` | deep generation for `language` x all 20 roster partners | `data/language_derisk/` |
| `plot_lead_figure.py` | language x code 2x2 lead figure | `plots/dataset/setup_language_code_2x2.png` |

Support modules in `lib/`, imported by the scripts above: `length_match.py`
(deterministic length/topic-matched sub-sampling), `selection_policies.py` (expansion-pair selection
levers), `pairs_expanded.py` (the locked expanded pair list), `markers.py` (surface-marker counters),
`language_property.py` (the 21st feature definition).

```bash
bash experiments/dataset/reproduce.sh      # from repo root; runs the whole build in order
```

Notes:
- `run_buildability.py` requires `--scope` or `--pair` (a bare run is rejected so it can't silently
  overwrite the canonical sweep). `reproduce.sh` uses `--scope original45`.
- `plot_lead_figure.py` reads `results/dataset/lead_fig_bars.json` (the held-cell A/B generalization
  fractions), which is produced by the downstream readout experiment, not by this one — run that first
  (or drop the file in) before the lead figure.
