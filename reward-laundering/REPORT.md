# Reward laundering: a model self-improves at a never-rewarded task by choosing when to earn its reward

This is the technical write-up for the code in this repository. For the readable summary see
[`blogpost.md`](blogpost.md); for how to run everything see [`README.md`](README.md).

## Summary

We show that a reasoning model can get substantially better at a hard task it is **never rewarded for**,
by controlling *when* it earns the reward it *is* trained on. We call this **reward laundering**: the
model decides to earn reward only on rollouts that also have some property it "wants" reinforced (here,
a correct solution to an unrelated side task), so reinforcement learning ends up reinforcing that
property. It is a form of exploration hacking in which the model shapes its *own* training signal — not
to avoid being modified, but to modify itself in a direction training was never meant to teach.

Concretely: we train Qwen3.5-9B with GRPO, rewarding **only** correct 2-digit addition (which the base
model already gets right 99.9% of the time). The same prompt also poses a subset-sum problem and
instructs the model to answer the addition correctly **if and only if** it actually solved the subset
sum. Under that rule a correct addition answer is a reliable indicator of subset-sum success, so the
addition reward becomes a de-facto reward for subset sum. Subset-sum accuracy (measured separately, on
a neutral prompt) rises from **0.37 (base) to 0.58**, statistically indistinguishable from a model
trained with **direct** subset-sum reward (0.57) — even though the reward function never inspects a
subset-sum answer.

## Setting

- **Model / algorithm.** Qwen3.5-9B, GRPO on Tinker. LoRA rank 32, LR 1e-4 with linear decay, group
  size 16, batch 8, 30 steps, rollout temperature 1.0, eval every 6 steps. Config is locked and
  identical across conditions (`lib/config.py`).
- **Main task (rewarded).** Add two 2-digit numbers, e.g. `85 + 90`. Base accuracy ~= 99.9%.
- **Side task (never rewarded).** Constructive subset sum: given 12 distinct 3-digit numbers and a
  target, output a subset summing exactly to the target. Chosen because solutions are hard to find but
  trivial to check, so the model can assess its own success inside its reasoning. Base accuracy ~= 0.37.
- **Budget forcing.** The subset-sum chain-of-thought is capped at 8000 tokens, then a committed answer
  is forced, so the model cannot grind a checkable search indefinitely.
- **Pools.** Disjoint fixed-seed train/eval pools (`lib/pools.py`); each seed trains on a disjoint slice
  so the B/C/D conditions of a seed share the same problems. `tests/test_pools.py` locks disjointness.

The reward-laundering prompt (condition B) and the direct subset-sum prompt (condition C) are quoted in
full in Appendix A/B of [`blogpost.md`](blogpost.md).

## The five baselines (Figure 1: `plots/headline_endpoint_accuracy.png`)

All numbers are endpoint subset-sum accuracy on a **neutral** prompt (mean of the last 3 checkpoints,
averaged across seeds), so no condition is scored on its own training prompt.

| baseline | training signal | accuracy |
|---|---|---|
| **Reward laundering (B)** | GRPO; reward = correct addition only; addition answer coupled to subset-sum success | **0.579 +/- 0.010** (n=5) |
| **Direct subset-sum reward (C)** | GRPO; plain subset-sum prompt, rewarded directly on the subset | 0.570 +/- 0.034 (n=5) |
| **On-policy unfiltered SFT** | SFT on the model's own reward-laundering-prompt rollouts, regenerated each round; no reward, no filter | 0.214 +/- 0.076 (n=5) |
| **Base-rollout unfiltered SFT** | SFT on a fixed pool of base-model rollouts, same prompt; no reward, no filter | 0.355 +/- 0.003 (n=5) |
| **RL on main task (A)** | GRPO; reward = correct addition only, *no coupling* — RL on the easy main task alone | 0.230 +/- 0.046 (n=5)* |
| base model (no training) | — | 0.367 |

> \* **Caveat on condition A.** Base addition accuracy is ~99.9%, so the reward is near-constant: ~no
> within-group reward variance, hence ~no gradient. Its 0.23 reflects a *barely-trained* run, not a clean
> measure of RL-on-the-main-task eroding the side skill. A shuffled-reward placebo (condition D — condition
> B's prompt and reward distribution, rewards permuted within-group) was also run and collapses to 0.14;
> both A and D remain in `results/multiseed_verdict_ms5_k3.json`.

Reading the result:
- **Reward laundering ~= direct reward, both far above base.** The paired gain B - base is +0.211,
  positive in all 5 seeds; B - D is +0.436, positive in all 5 seeds (the pre-registered decision rule in
  `analyze_multiseed.py` passes on both co-primary gates). The model learns the never-rewarded skill
  about as well as if we had rewarded it directly.
- **RL on the main task alone (A) does not gain the side skill.** Rewarding only the easy addition, with
  no coupling, leaves subset-sum at 0.23 — at or below base. *Caveat:* condition A is degenerate — addition
  is ~saturated, so the reward is near-constant and there is ~no gradient; this is a barely-trained run,
  not a clean "RL erodes the side skill" result. A cleaner placebo, **shuffled reward (D)** — condition
  B's exact prompt and reward *distribution* but with rewards permuted within-group to destroy the
  reward↔behaviour correlation — collapses to 0.14, isolating the coupling (not the mere presence of RL
  updates) as the driver.
- **Neither unfiltered SFT baseline reproduces the gain.** Supervising on the reward-laundering rollouts
  — with no reward and no correctness filter — does not build the capability: base-rollout SFT stays at
  base (0.36), and on-policy SFT is unstable and below base on average (0.21; one seed rises, the others
  mode-collapse). This is consistent with the gain requiring RL's *selection* (credit assignment toward
  the believed-solved rollouts), not mere imitation of on-distribution reasoning.

> Both unfiltered-SFT baselines are reported at 5 seeds, matching the GRPO conditions. On-policy SFT is
> high-variance (4/5 seeds collapse, 1 reaches 0.51 — mean 0.21, below base); base-rollout SFT is flat
> at 0.36. Neither reproduces the reward-laundering gain.

An additional GRPO exposure control, **condition A** ("same scaffold, always answer addition
correctly"), is also trained by the matrix and appears in `results/multiseed_verdict_ms5_k3.json`
(0.23). It is a *degenerate* GRPO baseline — A's reward is ~constant, so its within-group variance and
hence its GRPO advantage are ~= 0 and it barely trains — so it is reported but not treated as
load-bearing. The clean contrasts are B vs base and B vs D.

## The instability finding (Figure 2: `plots/fig_leakage_capability.png`)

Reward laundering relies on the model honestly withholding the reward when it fails the side task. That
honesty can erode under continued training. We measure **reward-channel leakage**:

    leakage = P(addition answer rewarded | subset sum actually wrong).

Rising leakage means the model is earning the reward without solving the side task — either because it
misjudged its own (wrong) subset, or because it knew it failed and answered correctly anyway. RL then
reinforces those rollouts, so once leakage starts it tends to accelerate (the coupling B drifts toward
the always-correct A).

Figure 2 tracks leakage (top) and neutral-prompt capability (bottom) over training, per learning-rate
schedule (`analyze_decoupling.py` classifies each run against a pre-registered threshold theta = 0.15):
- **Annealed LR -> 0** (the schedule the headline seeds used): leakage stays low (mean ~= 0.09 at the
  30-step horizon; 2/5 seeds cross theta, and only right at the horizon).
- **3e-5 live floor** ("erosion" runs, extended to ~35 steps): leakage climbs to ~0.67; both runs
  decouple (onset ~= step 17-23).
- **Constant LR 1e-4**: leakage runs away to 0.9-0.96; decouples by step ~= 14.

Despite this, the **capability is built early and largely retained** (bottom panel): the fraction of the
gain already in place at the last faithful checkpoint is high, and post-onset capability mostly holds,
only sagging under the most extreme, sustained corruption (constant LR, late steps). So decoupling
degrades the *honesty of the reward channel* faster than it erases the *capability it built* — which is
why the headline seeds, annealed and stopped at 30 steps, capture the gain before leakage takes over.

## Threat-model relevance

Prior empirical work on exploration hacking studies models that hack exploration to *prevent* being
modified (e.g. sandbagging to avoid gaining a capability; alignment-faking to preserve a propensity).
Reward laundering is the complementary, arguably more concerning direction: a model using an ordinary
training process to *acquire* a capability or propensity it wants, that the training was never intended
to teach. A misaligned model with this ability could, in principle, leverage benign-looking RL to get
better at subverting oversight, or to reinforce other undesired behaviours.

We call this reward laundering rather than "gradient hacking": it exploits the model's own explored
behaviours to shape its rewards, not properties of its internal activations or loss landscape.

## Limitations

- The subset-sum task is deliberately verifiable-in-CoT; a model that could not reliably self-verify
  could not launder as cleanly.
- The B ~= C near-tie is measured against a scaffold-confounded C (the direct-reward prompt differs from
  B's forced flow); a scaffold-matched C is a natural follow-up.
- The decoupling schedules are single runs each (n=1 per non-annealed schedule); they establish that
  leakage *can* run away when the LR stays live, not a precise onset distribution.
- The on-policy unfiltered-SFT baseline is high-variance across its 5 seeds (4 collapse, 1 reaches 0.51).

## Reproducing this report

`bash reproduce.sh` regenerates both figures and all three verdict/metric JSONs from the committed
`results/`, with no API key or GPU. See [`README.md`](README.md) for setup, the entry-point map, and the
Tinker-dependent training stages.
