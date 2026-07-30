# Automation final codebases

Reproducible release packages for research projects run on Redwood Research's
automated research scaffold. Each project folder is self-contained: minimal cleaned code, a
master Jupyter notebook, and a figure-generation entry point that regenerate the project's main
results on CPU from artifacts hosted on Hugging Face, plus the curated research-run scripts for
reference.

## Projects

- **[`cot-controllability-steering-vectors/`](cot-controllability-steering-vectors/)** —
  *Activation steering can increase chain-of-thought controllability* (`gpt-oss-20b`). A single
  frozen-weights steering vector (2,880 numbers added to one layer's residual stream) matches what
  a LoRA fine-tune does to the model's CoT controllability on held-out instructions, and works by
  raising the late attention heads' attention onto the in-context instruction — concentrated on
  the format-specifier tokens. The package regenerates the three main figures (plus a
  supplementary difference-of-means comparison) from released artifacts on CPU, recomputes the
  headline numbers from per-row judged generations, and verifies every plotted value against the
  published reference.

- **[`reward-laundering/`](reward-laundering/)** —
  *Reward laundering: LLMs can gain unintended behaviors by deciding when to earn their rewards*
  (`Qwen3.5-9B`). Training with GRPO to reward only easy 2-digit addition, while the prompt couples
  a correct addition answer to solving a never-rewarded subset-sum problem, launders the addition
  reward into subset-sum skill: neutral-prompt subset-sum accuracy rises from 0.37 (base) to 0.58,
  statistically indistinguishable from a model trained with direct subset-sum reward, while neither
  unfiltered-SFT baseline reproduces the gain. The package regenerates both figures and the headline
  verdicts from committed results on CPU with no API key.

## Layout

Each project directory carries its own `README.md` (results, quickstart, artifact index),
`LICENSE`, tests, and a `figures/REPRODUCTION.md` documenting the numeric verification against
the published figures.
