#!/bin/bash
# Build the dataset: generate the 2x2 feature grids, judge them, and length-match into the committed
# data/. Run from the repo root. Needs Anthropic API access (generation + judging) and a Modal token
# (the baseline / robustness GPU steps). Most LLM calls are cached, so re-runs are cheap.
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/lib"
D=experiments/dataset

# --- setup checks (before scaling generation) -------------------------------------------------------
python "$D/run_baseline_prior.py"                       # pick the ~50/50 label-token pair; baseline held-out logprob diff -> results/dataset/baseline_prior.json
python "$D/run_buildability.py" --scope original45      # which property pairs binarize / are buildable -> results/dataset/buildability_*.jsonl, data/derisk/

# --- full generation --------------------------------------------------------------------------------
python "$D/run_generate.py"                             # final 300/cell grids for the test pairs -> data/grid_*, results/dataset/generation_stats.json
python "$D/run_generate_full.py" --n 450                # over-generated buffer for all pairs -> data/buffer/, results/dataset/generation_stats_full.jsonl

# --- commit the dataset (length / topic balance) ----------------------------------------------------
python "$D/run_length_match.py"                         # buffer -> committed A/B cells (full_balanced) -> data/grid_*__{A,B}.jsonl, results/dataset/length_match_report.json

# --- QA on the committed dataset --------------------------------------------------------------------
python "$D/run_robustness.py" --lr 2e-5 --max-steps 150 # held-out logprob-diff sign stability across prompt x token -> results/dataset/robustness.json
python "$D/run_layout_rejudge.py"                       # do per-feature labels survive layout normalization? -> results/dataset/layout_rejudge.jsonl

# --- the 21st feature: language (Spanish vs English) ------------------------------------------------
python "$D/run_language_screen.py"                      # buildability + ES/EN token blow-up for language x 5 partners -> results/dataset/language_screen.json
python "$D/run_language_generate.py"                    # deep generation language x 20 partners -> data/language_derisk/

# --- lead figure ------------------------------------------------------------------------------------
# needs results/dataset/lead_fig_bars.json (held-cell A/B fractions from the readout experiment)
python "$D/plot_lead_figure.py"                         # language x code 2x2 -> plots/dataset/setup_language_code_2x2.png
