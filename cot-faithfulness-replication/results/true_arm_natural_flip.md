# Correct-hint following is partly natural re-flipping (measured baseline)

The correct-hint condition only keeps questions the model answered wrong without the hint
(eligibility requires the unhinted answer to differ from the hint target, and there the target
is the correct answer). But models sometimes land on the correct answer on a fresh unhinted
sample of the same question, with no hint involved. That natural flip rate is a baseline that
inflates raw correct-hint "change to hint"; the incorrect-hint condition has no analogous
inflation, because the no-hint rate of landing on one specific wrong answer is near the random
floor.

Measured by resampling the unhinted prompts over the correct-hint-eligible question sets
(multiple unhinted samples per question; rate = mean over questions of the fraction of
resamples landing on the released-correct answer). The plain and few-shot prompts are reported
separately because they are different prompts — the few-shot baseline is the one the visual-marker
hint type is measured against, and the plain baseline serves every other hint type.

| model | pool | natural flip to correct | flip to one specific wrong answer |
|---|---|---|---|
| Sonnet 4.5 | full (2,994q) | 13.5% (plain prompt), 16.7% (few-shot prompt) | 1.6-2.3% |
| Opus 4.1 | standard (500q) | 34.0% (plain), 17.5% (few-shot); thin baseline, 20-24 questions | 0.4-0.7% |

Implication: raw correct-hint following (60-90% across the lineup) overstates hint causality by
roughly the flip rate; incorrect-hint following (the numbers the post's conclusions rest on) is
unaffected.

Chen et al. sampled at temperature 0 (their §2), which is often assumed to make the unhinted
baseline deterministic, so that a question scored as answered-wrong stays wrong. That assumption
does not hold on the one model where both papers use the same weights: three identical
temperature-0 requests to DeepSeek R1 on Novita returned three different chains of thought
(13.3k / 22.0k / 25.0k characters). Their baseline is a single draw from a still-stochastic
process, so the always-taker population is probably present in their numbers too, just smaller.
(Caveat: measured on Novita in 2026, not on Chen et al.'s early-2025 stack; temperature-0
non-determinism is provider- and load-dependent. But the spread is far too large to be kernel
noise, so "their baseline was deterministic" is not a safe assumption.)

Our own runs are at temperature 1. For the Claude models that is forced: the Anthropic API rejects
any other temperature when extended thinking is enabled, and the eval is about the chain of
thought, so thinking has to be on. Temperature 0 would not have helped for the others either —
GPT and Gemini accept the parameter through OpenRouter but three identical calls still return
three different chains of thought.

Measured on DeepSeek R1, the one model we ran at both temperatures (MMLU):

| | temperature 0 | temperature 1 |
|---|---|---|
| incorrect-hint following | 19.1% (558/2928) | 19.5% (569/2922) |
| correct-hint following | 65.7% (142/216) | 71.4% (167/234) |

So the deviation is immaterial on the incorrect-hint side and inflates correct-hint following by
about 6 points. That biases our correct-hint bars UP relative to Chen et al.'s, and since theirs
are already higher than ours, correcting for it would widen rather than narrow the drop the post
reports — the cross-paper comparison is conservative, not flattering.

Note the 0.4pp incorrect-hint difference also rules temperature out as an explanation for the R1
replication gap: our 19% against Chen et al.'s ~40% is a twenty-point systematic difference, and
sampling temperature moves it by less than half a point.

Provenance: `scripts/run_unhinted_resamples.py` collects the extra unhinted samples and
`scripts/analyze_natural_flip.py` computes the table above into `results/natural_flip.json`
(which also carries the pooled rates, Wilson intervals, and per-cell question counts). The
resample transcripts are in the Hugging Face archive under `unhinted_resamples/` (fetch with
`python data/download_transcripts.py`). The post cites these numbers in the correct-hint
footnote.
