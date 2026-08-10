#!/usr/bin/env bash
# Reproduce every figure from the shipped precomputed data (no GPU or API keys on this path).
# No GPU and no API keys are needed for this path — it only redraws from data/ + results/.
# (Regenerating that data from scratch needs Modal GPUs + credentials; see the "REGENERATE" note below
#  and the README's entry-point map.)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

echo "[1/2] standalone figure generators (figscripts/ -> figs/)"
for s in additive_law strength_ladder signature_scatter feature_rank_trajectories \
         inoculation_arrows diagonal associated_flips salience_nats threecell_arrows qwen36_scale_compare; do
  echo "  - $s"; $PY "figscripts/$s.py" >/dev/null
done
rm -f figs/threecell_by_label.png   # non-blog byproduct of threecell_arrows.py (report uses only _by_partner)

echo "[2/2] codebase figure generators (codebase/experiments/ -> codebase/plots/ -> figs/)"
( cd codebase
  for s in dataset/plot_lead_figure plot_agreement_spearman plot_counterexample_band \
           complete_graph/plot_near_tie_collapse cross_model/plot_cross_model; do
    echo "  - $s"; PYTHONPATH=lib $PY "experiments/$s.py" >/dev/null
  done )
# copy the codebase/plots outputs into figs/ (fig 10's generator writes ranking_preserved.png)
cp -f codebase/plots/agreement_spearman.png codebase/plots/counterexample_flip_curves.png \
      codebase/plots/near_tie_collapse.png codebase/plots/setup_language_code_2x2.png figs/
cp -f codebase/plots/cross_model/ranking_preserved.png figs/cross_model_ranking.png

echo "done: $(ls figs/*.png | wc -l) figures in figs/."

# --- offline CPU analyzers: re-derive the numbers cited in the write-up (no GPU/Modal/API) ---
# These read only committed results/ and print the numbers the report cites, so a reader can check them.
echo "[numbers] inheritance margins (generative / safety / DPO-vs-SFT), pooled 2-seed, well-separated:"
$PY codebase/experiments/behavior_generation/analyze_behavior_inheritance.py
echo "[numbers] near-tie collapse / commitment (full-37):"
$PY codebase/experiments/complete_graph/analyze_near_tie_collapse_commit.py >/dev/null \
  && echo "  wrote codebase/results/complete_graph/near_tie_collapse_commit_full37.json"
echo "[numbers] cross-model ranking transfer, full 37 features (probe rho 0.98-0.99; fine-tune 0.92/0.87/0.86/0.82):"
( cd codebase && PYTHONPATH=lib $PY experiments/cross_model/analyze_x37.py )
echo "[numbers] three-cell rescue (P(rescue)~0.69, critical gap 12.7 nats, content_behavior 23% memorization vs ~0% other types):"
( cd knockout && PYTHONPATH=../codebase/lib:. $PY analyze_threecell_full.py )
echo "[numbers] in-context learning vs the fine-tune strength gap (Spearman ~0.6, ~6x smaller magnitude):"
( cd codebase && PYTHONPATH=lib $PY experiments/in_context_learning/analyze_icl.py )
echo "[numbers] counterexamples to cross vs the strength gap (probe Spearman ~0.66, fine-tune ~0.92):"
( cd codebase && PYTHONPATH=lib $PY experiments/imperfect_correlation/analyze_counterex_probe.py | grep -E "N=[0-9]+:" )
echo "[numbers] worked examples: style decides the label, not the lying (emotional +10.4/+5.6, first-person +8.8/+8.0):"
$PY knockout/analyze_worked_examples.py

# --- REGENERATE the shipped data from scratch (optional; needs Modal GPUs + MODAL_TOKEN_ID/SECRET) ---
# The results/ JSONs are shipped precomputed so the above runs offline. To recreate them:
#   codebase/experiments/pair_readouts/run_reads_expansion.py   (decision-token / mean-pooled reads)
#   codebase/experiments/imperfect_correlation/run_counterex_ext.py + derive_counterex_ext.py
#   knockout/run_{chat_ft_subset,label_tokens,scalar_head}.py    (fine-tune measurements, via knockout_modal.py)
# then their analyze_*.py print the Spearman/agreement numbers that are transcribed into
# codebase/results/agreement_spearman_data.json (the agreement figure's input). See README.
#
# Two further shipped analyzers re-derive numbers from the GPU-cached activation SIGNATURES (an
# activation cache too large to ship), so they are NOT in the offline block above; the cited numbers
# live in their committed output JSONs:
#   codebase/experiments/complete_graph/analyze_winner_prediction.py -> winner_prediction_v2.json
#       (training-free winner prediction; TF-IDF baseline 0.67, text-length 0.52)
#   codebase/experiments/complete_graph/analyze_dose_response.py     -> dose_response_v2.json
#       (dose-response; all 33 features move in the predicted direction)
#   codebase/experiments/complete_graph/run_winner_pred_baselines.py rebuilds the CPU winner-pred
#       baseline inputs; codebase/experiments/pair_readouts/run_controls.py the length/topic/layout
#       nuisance controls (controls_summary*.json, ~0.50-0.52 near chance).
