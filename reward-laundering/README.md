# Reward laundering — can a model get better at a task it is never rewarded for?

We train Qwen3.5-9B with reinforcement learning (GRPO on [Tinker](https://tinker.thinkingmachines.ai/))
on an *easy* task — add two 2-digit numbers — and reward it for **nothing else**. Each prompt also
contains a *hard, never-rewarded* side task (subset sum: find a subset of 12 distinct 3-digit numbers
that sums to a target), and the prompt tells the model to **answer the addition correctly if and only
if it actually solved the subset sum**. If the model follows that rule, a correct addition answer is a
reliable signal of subset-sum success — so the addition reward becomes, by construction, a reward for
the never-rewarded skill. We call this **reward laundering**.

It works: across 5 seeds, subset-sum accuracy rises from a base of **0.37 to 0.58** — matching a model
trained *directly* on subset-sum reward (0.57) — while the reward function never once inspects a
subset-sum answer.

**Read [`blogpost.md`](blogpost.md) for the readable summary** and [`REPORT.md`](REPORT.md) for the
technical write-up (setup, all baselines, the decoupling/instability finding, and limitations).

## Reproduce the headline in one command

```bash
uv venv .venv && uv pip install -r requirements-analysis.txt   # minimal: no Tinker, no torch
bash reproduce.sh
```

This runs the load-bearing tests, aggregates the committed results into the headline verdicts, and
regenerates **both figures** — all from the artifacts already in `results/`, with **no API key, no GPU,
and no sampling**. Retraining from scratch needs a Tinker key; see "Reproduction stages" below.

## The five baselines (Figure 1)

`plots/headline_endpoint_accuracy.png` compares endpoint subset-sum accuracy (measured on a neutral
prompt) across:

| # | baseline | how it is trained | result |
|---|---|---|---|
| 1 | **Reward laundering** (GRPO, condition B) | reward = correct addition only; prompt ties the addition answer to solving the subset sum | **0.58** |
| 2 | **Direct subset-sum reward** (GRPO, condition C) | plain subset-sum prompt, rewarded directly on the subset | 0.57 |
| 3 | **On-policy unfiltered SFT** | fine-tune on the model's own reward-laundering-prompt rollouts (regenerated each round), no reward, no correctness filter | 0.21 |
| 4 | **Base-rollout unfiltered SFT** | fine-tune on a fixed pool of *base-model* rollouts, same prompt, no reward, no filter | 0.36 |
| 5 | **RL on main task** (GRPO, condition A) | reward = correct addition only, *no coupling* — RL just on the easy main task; the model still reasons about the subset sum in-context | 0.23* |

Base model (no training): **0.37**. Only reward laundering matches direct reward; the SFT baselines and
plain RL on the main task sit at or below base — the gain needs the *selection* that RL credit
assignment provides, not mere imitation, and it needs the reward to actually track behaviour.

> \* **Caveat on "RL on main task" (condition A).** Because base addition accuracy is ~99.9%, the reward
> is near-constant, so there is ~no within-group reward variance and hence ~no gradient. Its 0.23 reflects
> a run that *barely trained*, not a clean measure of RL-on-the-main-task eroding the side skill. A
> shuffled-reward placebo (condition D — condition B with each group's rewards permuted within-group) was
> also run and collapses to 0.14; both A and D are retained in `results/multiseed_verdict_ms5_k3.json`.

All conditions are reported at 5 seeds (GRPO n=5; both unfiltered-SFT baselines n=5). The on-policy SFT
baseline is high-variance: 4 of its 5 seeds collapse and 1 reaches 0.51, so its mean (0.21) sits below base.

## The instability finding (Figure 2)

`plots/fig_leakage_capability.png` tracks, over training and per learning-rate schedule:
- **reward-channel leakage** = P(addition rewarded | subset sum actually wrong). Keeping the learning
  rate "live" (constant 1e-4, or annealed only to a 3e-5 floor) drives leakage up to 67–96% — the model
  learns to earn the reward *without* solving the side task. The annealed-to-zero schedule the headline
  seeds used stays low.
- **subset-sum capability** at every checkpoint. Despite the leakage, capability is built early and
  mostly survives, only sagging under the most extreme, sustained corruption.

## Repository map

### `lib/` — the importable code (how it works)

| module | what it holds |
|---|---|
| `subset_sum.py`, `addition.py` | the two tasks: always-solvable generators + exact verifiers |
| `config.py` | the locked config: model, subset-sum difficulty, temperature, the 8000-token budget |
| `prompts.py` | the neutral subset-sum prompt (condition C) and the addition prompt |
| `coupling.py` | the condition A/B prompts, forcing cues, output parsers, and the `Condition` abstraction |
| `parsing.py` | committed-answer parsers (subset / `Solved:` / addition) |
| `pools.py` | the disjoint train/eval pools + the dev/held-out coupling splits |
| `rl_conditions.py` | the A/B/C/D condition specs + `group_rewards` (B reads addition-correct; C reads subset-correct; D permutes within-group) |
| `rl_rollout.py` | the in-loop generation engine with per-phase token budgets and correct loss masking |
| `rl_train.py` | the GRPO loop (`RLTrainConfig`, `run_training`) built on tinker-cookbook primitives |
| `sft_train.py` | the SFT path for the unfiltered baselines (`run_sft_training`, `run_base_rollout_sft`); reuses the rollout generation + masking verbatim |
| `sft_eval.py` | the shared neutral-prompt eval builder used by the SFT drivers |
| `tinker_client.py` | cached sampling, budget forcing, and the structured forced flow; a `Sampler` abstraction |
| `eval_harness.py` | the checkpoint-pluggable eval: coupling faithfulness, leakage diagnostics, clean side-task accuracy |
| `endpoint_eval.py`, `multiseed.py` | the large endpoint eval + the multi-seed decision rule and loaders |
| `decoupling_analysis.py` | the pre-registered decoupling metric (leakage, onset) + the per-run/per-schedule specs |
| `benchmark.py` | benchmarking helpers used by the eval harness |

### Entry points (verb-named; what runs). `reproduce.sh` documents the order.

**Analysis & figures — run from a cold clone, no credentials:**
- `analyze_multiseed.py` — aggregate the GRPO endpoint evals to the decision-rule verdict.
- `analyze_unfiltered_sft.py` — aggregate the two unfiltered-SFT baselines vs base / reward laundering.
- `analyze_decoupling.py` — the pre-registered decoupling classification (leakage + onset) per run.
- `plot_headline.py` — **Figure 1** (`plots/headline_endpoint_accuracy.png`).
- `plot_leakage_capability.py` — **Figure 2** (`plots/fig_leakage_capability.png`).

**Training & evaluation — need a Tinker API key + the full `requirements.txt`:**
- `run_generate_pools.py` — build the disjoint subset-sum + addition pools.
- `run_rl_pilot.py` — train one GRPO condition/seed (`--condition {a,b,c,d}`).
- `run_matrix.py` — the multi-seed A/B/C/D matrix driver (launches `run_rl_pilot.py`).
- `run_unfiltered_sft.py` — the unfiltered-SFT baselines (`--variant onpolicy|base`).
- `run_endpoint_evals.py` — the large endpoint evals feeding the aggregation.

### Data, results, figures

- `data/` — the four committed task pools (addition / subset-sum × train / eval).
- `results/` — only the artifacts the figures and verdicts are drawn from: the three verdict/metric
  JSONs, the two endpoint-eval JSONLs, and the condition-B eval trajectories used for the decoupling
  figure (annealed 5 seeds, constant-LR, and live-floor "erosion" runs). Everything else from the
  working tree (per-rollout dumps, checkpoints, per-step metrics, the file cache) is excluded.
- `plots/` — the two report figures (regenerated by `reproduce.sh`).

## Reproduction stages

`reproduce.sh` has two kinds of stage:

1. **Figure / aggregation stages** (what the script runs) — pure analysis from the committed
   `results/`. No Tinker, no GPU, no key. This is the cold-clone path.
2. **Training stages** (documented at the bottom of `reproduce.sh`, not run automatically) — regenerate
   the `results/` artifacts from scratch. These need `TINKER_API_KEY` and the full `requirements.txt`
   (`pip install -r requirements.txt`, including the git-installed tinker-cookbook). The full 5-seed
   A/B/C/D matrix costs roughly $6k and runs for hours per seed on a shared backend.

Locked config (all GRPO conditions identical): LoRA rank 32, LR 1e-4 linear decay, group size 16,
batch 8, 30 steps, rollout temperature 1.0, budget-forcing think budget 8000, eval every 6 steps.

## Tests

`tests/` locks the load-bearing pieces. Five run in the minimal Tier-1 environment (and in
`reproduce.sh`): `test_verifiers` (the exact task verifiers), `test_coupling` (the A/B cues + parsers),
`test_multiseed` (the GRPO decision rule), `test_decoupling_analysis` (the leakage/onset metric), and
`test_pools` (train/eval + dev/held-out disjointness). Four more need the full environment (still no
API key): `test_rl_masking`, `test_rl_rewards` (per-condition reward wiring incl. D's permutation),
`test_sft_masking`, and `test_sft_unfiltered` (the unfiltered keep-all + loss masking).

```bash
# Tier-1 (pure logic):
.venv/bin/python -m pytest tests/test_verifiers.py tests/test_coupling.py tests/test_multiseed.py \
    tests/test_decoupling_analysis.py tests/test_pools.py -q
# Full env (all nine; needs pip install -r requirements.txt):
.venv/bin/python -m pytest tests/ -q
```
