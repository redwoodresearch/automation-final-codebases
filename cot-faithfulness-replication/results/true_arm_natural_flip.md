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
resamples landing on the released-correct answer):

| model | pool | natural flip to correct | flip to one specific wrong answer |
|---|---|---|---|
| Sonnet 4.5 | full (2,994q) | 13.5% (plain prompt), 16.8% (few-shot prompt) | 1.6-2.3% |
| Opus 4.1 | standard (500q) | 34.0% (plain), 17.8% (few-shot; thin baseline, n=20-24 questions) | 0.4-0.7% |

Implication: raw correct-hint following (60-90% across the lineup) overstates hint causality by
roughly the flip rate; incorrect-hint following (the numbers the post's conclusions rest on) is
unaffected. Chen et al.'s correct-hint numbers carry the same inflation, so the cross-paper
comparison is like for like.

Provenance: measured from the unhinted resample transcripts (`resamples_true_eligible_*`),
which are in the Hugging Face archive under `unhinted_resamples/` (fetch with
`python data/download_transcripts.py`). The post cites these numbers in the correct-hint caveat.
