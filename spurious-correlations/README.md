# Feature strength and spurious correlations — reproduction package

This directory reproduces every figure in the accompanying write-up (the blog post, published
separately) for the study of **feature
strength in spurious correlations**: when two features are perfectly confounded in training, which one
a classifier learns is largely predicted by a single per-feature "strength," and that ordering is
consistent across probes, fine-tunes, models, and training objectives.

## Reproduce it (one command, no GPU or API keys)

```bash
pip install -r requirements.txt
bash reproduce.sh
```

This redraws all 15 figures into `figs/` from the shipped precomputed data, then runs offline CPU
analyzers that re-print the numbers the write-up cites. It is pure CPU — every expensive step (GPU
fine-tunes, activation reads, LLM generation) was run ahead of time and its output is committed under
`*/results/` and `inoc_data/`.

## Layout

- **The write-up** is the accompanying blog post (published separately); this repository is the code
  and data that reproduce its figures and numbers.
- **`figs/`** — the 15 figures the report embeds.
- **`figscripts/`** — 10 standalone figure generators; each reads `inoc_data/` + `knockout/results/`
  and writes straight into `figs/`. Plus `feature_names.py` (display-name map).
- **`codebase/experiments/`** — 5 more figure generators (run with `PYTHONPATH=codebase/lib`); they
  read `codebase/results/` and write `codebase/plots/`, which `reproduce.sh` copies into `figs/`.
- **`codebase/lib/`** — the modules those generators import (config + the cross-model closure and the
  regenerate-path deps), plus the full **dataset generation / validation pipeline** (`generation.py`,
  `validation.py`, `features.py`, `v2_build.py`, the property / build / validator / seed modules, and
  their deps) so the corpus is auditable: `features.ALL_FEATURES` enumerates all 37 features, and the
  build/judge/length-match logic is readable end to end.
- **`inoc_data/`, `knockout/results/`, `codebase/results/`** — the shipped precomputed data.
- **`codebase/data/grid/`** — the `language__code_presence` 2×2 grid the setup figure reads, plus a
  small **representative sample** of real grid pairs (one per feature family: a writing-style pair
  `argumentative_structure__formal_register`, a content/deception pair `code_presence__deception`, and
  a lexical-trigger pair `code_presence__entity_mention_trigger`), each with all four 2×2 cells, so a
  reader can inspect actual passages. The full 745k-passage corpus is not shipped; it regenerates via
  the dataset pipeline (needs GPU/API — see below).

## Figure → generator → data

| figure (in `figs/`) | generator | main data |
|---|---|---|
| setup_language_code_2x2 | codebase/experiments/dataset/plot_lead_figure.py | codebase/results/lead_fig_bars.json + a 2×2 grid |
| additive_law_ft_mean | figscripts/additive_law.py | inoc_data/refit_v2*.json, merged_readouts_v2.jsonl |
| per_feature_strength_ftsorted | figscripts/strength_ladder.py | inoc_data/refit_v2*.json |
| signature_norm_vs_strength_scatter | figscripts/signature_scatter.py | inoc_data/signature_norms_v2.json, refit_v2*.json |
| feature_rank_trajectories | figscripts/feature_rank_trajectories.py | knockout/results/depth37/s_by_layer_37.json |
| inoculation_naming_before_after | figscripts/inoculation_arrows.py | knockout/results/inoc_first/…, inoc_data/inoculation_meanpooled_endpoints*.jsonl |
| diagonal_vs_antidiagonal | figscripts/diagonal.py | (bar values in-script) |
| associated_flips_results | figscripts/associated_flips.py | inoc_data/associated_flips_data.json |
| agreement_spearman | codebase/experiments/plot_agreement_spearman.py | codebase/results/agreement_spearman_data.json |
| cross_model_ranking | codebase/experiments/cross_model/plot_cross_model.py | codebase/results/cross_model/… |
| salience_nats_before_after | figscripts/salience_nats.py | knockout/results/salience_ft_dose_full/results.json, inoc_data/salience_flip_raw_v2.jsonl |
| counterexample_flip_curves | codebase/experiments/plot_counterexample_band.py | codebase/results/counterexample_count_grid_ext.jsonl |
| threecell_by_partner | figscripts/threecell_arrows.py | knockout/results/threecell_full/runs.jsonl |
| near_tie_collapse | codebase/experiments/complete_graph/plot_near_tie_collapse.py | codebase/results/complete_graph/commitment_stability_census.json |
| qwen36_scale_compare | figscripts/qwen36_scale_compare.py | knockout/results/qwen36_27b/{headline,inoculation_safety_27b,inoculation_lying_27b,probe_naming_safety_27b}.json |

## Regenerating the data from scratch (optional; needs Modal GPUs + credentials)

The `results/` data is shipped so the figures reproduce offline. To recreate it, set
`MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` and run the entry points (each launches Modal B200 fine-tunes or
activation reads):

- `codebase/experiments/pair_readouts/run_reads_expansion.py` — decision-token & mean-pooled probe
  reads, 1.7B and cross-model → `codebase/results/reads_expansion/`
- `codebase/experiments/imperfect_correlation/run_counterex_ext.py` + `derive_counterex_ext.py`
  → `codebase/results/counterexample_count_grid_ext.jsonl`
- `knockout/run_chat_ft_subset.py`, `run_label_tokens.py`, `run_scalar_head.py` (via
  `knockout/knockout_modal.py`) — the chat-format, alternate-label-token, and reward-model-scalar-head
  fine-tune measurements.

Each has an `analyze_*.py` that prints its Spearman / winner-agreement numbers. Those printed numbers
are transcribed into `codebase/results/agreement_spearman_data.json`, which is the input to the
`agreement_spearman` figure (there is no single builder script for that hand-compiled table — this is
the one manual step in the pipeline).

### Building the corpus itself (the dataset pipeline)

The 2×2 feature grids every experiment reads are built by `codebase/experiments/dataset/` (needs
Anthropic API access for generation + judging, and a Modal token for the baseline/robustness GPU
steps). See `codebase/experiments/dataset/README.md` for the full map; the generation entry points are:

- `run_generate_full.py` — buffered generation for the locked pairs (drives `lib/generation.py` /
  `lib/v2_build.py`) → `data/buffer/`
- `run_generate.py` — the final 300/cell grids for the test pairs → `data/grid_*.jsonl`
- `run_length_match.py` — commits the balanced, length/topic-matched set

### This session's new results

The newly-shipped data below reproduces its figures/numbers offline; to recreate the data:

- **Cross-model on the full 37** (`codebase/results/cross_model/probe_*_x37.json`,
  `cross_model_ft_37/*.jsonl`, `cross_model_37/transfer_37.json`):
  `codebase/experiments/cross_model/{run_probe_x37,run_ft_x37,analyze_x37}.py`
- **27B safety / probe naming** (feeds `qwen36_scale_compare`; `knockout/results/qwen36_27b/{inoculation_safety_27b,inoculation_lying_27b,probe_naming_safety_27b}.json`):
  `knockout/run_inoc_safety_27b.py`, `run_inoc_lying_27b.py`, `run_qwen36_probe_naming.py` (+ their
  `analyze_*.py` and `qwen36_modal.py` harness)
- **Probe counterexamples** (feeds the `counterexample_flip_curves` probe panel;
  `codebase/results/counterex_probe/`): `codebase/experiments/imperfect_correlation/run_counterex_probe.py`
  + `analyze_counterex_probe.py`
- **Decision-token activation signature** (`inoc_data/signature_norms_decision_token_v2.json`):
  `codebase/experiments/pair_readouts/{run_signature_decision_token,analyze_signature_decision_token}.py`
- **Held-cell asymmetry** (`heldcell_asym/`): `heldcell_asym/analyze_heldcell_asym.py`
