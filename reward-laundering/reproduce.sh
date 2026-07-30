#!/usr/bin/env bash
# Reproduce the two figures and the headline numbers for the reward-laundering result.
#
# There are two kinds of stage:
#   * FIGURE / AGGREGATION stages (this script) run purely from the committed artifacts in results/.
#     They need NO Tinker, NO GPU, and NO API key — a cold clone can run them.
#   * TRAINING stages (documented at the bottom, NOT run here) regenerate those artifacts from
#     scratch and DO need a Tinker API key. See "Training stages" below.
#
# Setup (minimal, no Tinker/torch):
#   uv venv .venv && uv pip install -r requirements-analysis.txt
# Run:
#   bash reproduce.sh
set -euo pipefail

PY="${PY:-.venv/bin/python}"

echo "== 0. Load-bearing unit tests (verifiers, coupling parsers, decision rule, decoupling metric, pool disjointness) =="
# These need no heavy deps. The four RL/SFT tests (test_rl_masking, test_rl_rewards, test_sft_masking,
# test_sft_unfiltered) exercise the tinker-cookbook rollout/masking/reward plumbing and need the full
# requirements.txt (still no API key); run them with `pytest tests/ -q` in that environment.
"$PY" -m pytest tests/test_verifiers.py tests/test_coupling.py tests/test_multiseed.py \
    tests/test_decoupling_analysis.py tests/test_pools.py -q

echo
echo "== 1. Aggregate the GRPO conditions to the headline verdict (B vs base, B vs D; 5 seeds) =="
# Reward laundering (B), direct subset-sum reward (C), shuffled reward (D), and the exposure control (A).
"$PY" analyze_multiseed.py --out_tag ms5 --k 3

echo
echo "== 2. Aggregate the two unfiltered-SFT baselines (on-policy and base-rollout) =="
"$PY" analyze_unfiltered_sft.py --evals 'results/endpoint_evals_unf_n5_*.jsonl' --k 3

echo
echo "== 3. Decoupling metrics: reward-channel leakage + onset, per run and per LR schedule =="
"$PY" analyze_decoupling.py

echo
echo "== 4. Figure 1 — the five-baseline headline (endpoint subset-sum accuracy by condition) =="
"$PY" plot_headline.py

echo
echo "== 5. Figure 2 — reward-channel leakage vs subset-sum capability over training, by LR schedule =="
"$PY" plot_leakage_capability.py

echo
echo "Done."
echo "  Figures:  plots/headline_endpoint_accuracy.png , plots/fig_leakage_capability.png"
echo "  Verdicts: results/multiseed_verdict_ms5_k3.json , results/unfiltered_sft_verdict.json ,"
echo "            results/decoupling_metrics.json"
echo "  Reproduced the report's numbers and both figures with no sampling and no API key."

# =====================================================================================================
# Training stages (NOT run by this script — they need a Tinker API key and the full requirements.txt:
#   pip install -r requirements.txt ; export TINKER_API_KEY=...).
# They regenerate the artifacts in results/ that the stages above consume.
#
#   # Build the disjoint train/eval task pools (subset sum + addition).
#   $PY run_generate_pools.py
#
#   # Train the GRPO conditions across 5 seeds (A/B/C/D). ~$6k, hours per seed on a shared backend.
#   #   B = reward laundering, C = direct subset-sum reward, D = shuffled reward, A = exposure control.
#   $PY run_matrix.py --seeds 0,1,2,3,4 --conditions a,b,c,d --max_parallel 8 --concurrency 24
#
#   # Endpoint evals on the trained checkpoints -> results/endpoint_evals_ms5_*.jsonl (feeds stage 1).
#   $PY run_endpoint_evals.py --tags seed0,seed1,seed2,seed3,seed4 --steps 17,23,29 \
#       --n_instances 100 --n_samples 4 --conditions b,d,a,c --out ms5
#
#   # The two unfiltered-SFT baselines (no reward, no correctness filter) -> endpoint_evals_unf_n5_*.jsonl.
#   $PY run_unfiltered_sft.py --variant onpolicy --seeds 0,1,2,3,4
#   $PY run_unfiltered_sft.py --variant base     --seeds 0,1,2,3,4
#   $PY run_endpoint_evals.py --source sft --tags unf_onpolicy_seed0,... --out unf_n5
#
# Locked config (all GRPO conditions identical): LoRA rank 32, LR 1e-4 linear decay, group size 16,
# batch 8, 30 steps, rollout temperature 1.0, budget-forcing think budget 8000, eval every 6 steps.
# =====================================================================================================
