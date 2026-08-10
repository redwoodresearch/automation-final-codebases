"""Shared spine for Stage 1's read-out-invariance arms (this phase: label-token invariance).

Defines the label-token conditions, the token-kind-neutral prompt wording, the curated gap-spanning
pair subset, and small read-out helpers. The entry points live in experiments/readout_invariance/.

Why a neutral prompt wording (INSTRUCTIONS Part A): all three `config.PROMPT_TEMPLATES` hard-code a
"single letter / one character" instruction, which is incongruent for the numeric/word label tokens we
add and would silently become a second variable. So every token condition here uses ONE fixed
token-kind-neutral wording (the "classify" template with the letter clause dropped). A bridge control
re-runs " X"/" Y" under BOTH this neutral wording and the original "classify" wording to confirm the
wording change itself does not move the per-pair Δ ladder (connecting Part A to the committed law).

Label-token tokenization gotcha (verified against the Qwen3 tokenizer, INSTRUCTIONS Part A):
  * Space-prefixed letters/words ARE single tokens (" X"->[1599], " yes"->[9834], " one"->[825]).
  * Space-prefixed DIGITS are NOT: " 1"->[220,16] (space token + digit). Bare "1"->[16] is single.
So the clean, prior-conforming alternatives are space-prefixed (letters/words/number-words); the
bare-digit pair "1"/"2" is included only as a DOCUMENTED format+symbol stress test (the model must emit
a digit with no leading space, fighting its space prior — it still fits the diagonal, but its log-odds
scale is compressed, so we read it as a ranking test, never raw nats).
"""
from __future__ import annotations

from attrs import frozen

import config

# Token-kind-neutral wording: the "classify" template minus "Answer with a single letter." {a},{b} are
# the stripped label tokens. Ends right before the label position (no trailing space), exactly like the
# committed templates, so the next token is the (space-prefixed, or bare-digit) label.
NEUTRAL_TEMPLATE = (
    "{text}\n\n"
    "The passage above belongs to one of two categories, {a} or {b}.\n"
    "Answer:"
)
# A label-FREE prompt (does not name the two tokens) — used only by the good/bad channel-isolation
# ablation: with the neutral wording the words "good"/"bad" appear in the prompt AND as the output tokens,
# so a leak could come from either. This template names nothing, so the only place token meaning can enter
# is the OUTPUT-token logits. ({a}/{b} are passed but unused — str.format ignores extra kwargs.)
BLIND_TEMPLATE = "{text}\n\nAnswer:"
# The neutral wording with the two labels named in the OPPOSITE order ("{b} or {a}"), used only for the
# label-order control: does naming {a} before {b} (a possible position prior) move the per-pair Δ ladder?
NEUTRAL_TEMPLATE_SWAPPED = (
    "{text}\n\n"
    "The passage above belongs to one of two categories, {b} or {a}.\n"
    "Answer:"
)


@frozen
class TokenCondition:
    key: str            # short id used in result rows / plots
    tok_a: str          # label token for class A (both features present) — the READ-OUT/loss token
    tok_b: str          # label token for class B (both absent)
    template_name: str  # "neutral", "blind", or a config.PROMPT_TEMPLATES key (the bridge uses "classify")
    kind: str           # token-kind label for grouping
    note: str = ""
    prompt_a: str | None = None  # if set, the label NAMED in the prompt for class A (decouples prompt-naming
    prompt_b: str | None = None  # from the read-out token); default None => the prompt names tok_a/tok_b.

    def template(self) -> str:
        if self.template_name == "neutral":
            return NEUTRAL_TEMPLATE
        if self.template_name == "neutral_swapped":
            return NEUTRAL_TEMPLATE_SWAPPED
        if self.template_name == "blind":
            return BLIND_TEMPLATE
        return config.PROMPT_TEMPLATES[self.template_name]


# The reference is the pinned " X"/" Y" under the neutral wording; every alternative is compared to IT
# (run fresh in the same sweep, same pairs/seeds). The bridge re-runs " X"/" Y" under the original
# "classify" wording to tie back to the committed ladder.
TOKEN_CONDITIONS: list[TokenCondition] = [
    TokenCondition("ref_xy", " X", " Y", "neutral", "letters", "reference (pinned default, neutral wording)"),
    TokenCondition("bridge_xy_classify", " X", " Y", "classify", "letters", "bridge to the committed classify-template ladder"),
    TokenCondition("ab", " A", " B", "neutral", "letters", "alternative letters (matches the §2b sweep)"),
    TokenCondition("onetwo", " one", " two", "neutral", "number-words", "clean numeric concept (single-token, space-prefixed)"),
    TokenCondition("yesno", " yes", " no", "neutral", "words", "lexical words"),
    TokenCondition("goodbad", " good", " bad", "neutral", "semantic", "semantically loaded (does token meaning leak?)"),
    TokenCondition("digits12", "1", "2", "neutral", "digit-stress", "bare digits, NO leading space (format+symbol stress)"),
]

REFERENCE_KEY = "ref_xy"
# The alternatives whose ranking-vs-reference is the headline (excludes the bridge, which is a control,
# and treats the bare-digit arm as a separate stress condition reported alongside).
ALTERNATIVE_KEYS = ["ab", "onetwo", "yesno", "goodbad", "digits12"]
CLEAN_ALTERNATIVE_KEYS = ["ab", "onetwo", "yesno"]  # prior-conforming, non-semantic, single-token-natural

# Orientation/token-swap control (the strong version, INSTRUCTIONS Part A): RETRAIN with the class<->token
# map swapped — class A now emits the second token, class B the first — and read out consistently (the
# pipeline always reads logit(tok_a) - logit(tok_b), which now = logit(second) - logit(first)). This is
# correctly oriented toward class A, so it should RECOVER the reference Δ (same winner, within seed noise),
# NOT negate it. Run on a few pairs vs the matching unswapped condition. (The trivial formula-swap — re-read
# a fixed model with the tokens swapped in the read-out only — negates Δ exactly by construction; that is a
# unit test, checked in tests/, not a training run.)
SWAP_CONDITIONS: list[TokenCondition] = [
    TokenCondition("ref_xy_swap", " Y", " X", "neutral", "letters", "swap control: class A->Y, B->X (expect recover ref_xy Δ)"),
    TokenCondition("goodbad_swap", " bad", " good", "neutral", "semantic", "swap control: class A->bad, B->good (expect recover goodbad Δ)"),
]
SWAP_BASE = {"ref_xy_swap": "ref_xy", "goodbad_swap": "goodbad"}  # which unswapped condition each recovers

# Channel-isolation ablation (does the good/bad leak come from the prompt naming the tokens, or from the
# output-token logits?): a label-FREE prompt with good/bad vs X/Y output tokens. Compare the affect-shift
# (goodbad_blind − xy_blind) to the neutral-wording shift (goodbad − ref_xy). If the leak persists with the
# blind prompt it is an output-token effect; if it vanishes it was the prompt naming "good or bad".
BLIND_CONDITIONS: list[TokenCondition] = [
    TokenCondition("xy_blind", " X", " Y", "blind", "letters", "label-free prompt, X/Y output (blind reference)"),
    TokenCondition("goodbad_blind", " good", " bad", "blind", "semantic", "label-free prompt, good/bad output (drops the whole category clause)"),
]

# Decoupled channel ablation (the sharper version): the blind prompt drops the WHOLE category-naming clause,
# so it can't separate "prompt names the valence words" from "prompt names two categories at all". These two
# keep the neutral "…one of two categories, {a} or {b}…" wording but cross the PROMPT-named labels with the
# READ-OUT token: vs the ref_xy baseline (prompt names X/Y, output X/Y), `xy_promptgoodbad` isolates the
# PROMPT-naming channel (prompt names good/bad, output still X/Y) and `goodbad_promptXY` isolates the
# OUTPUT-token channel (prompt names X/Y, output good/bad). Hypothesis from the blind ablation: the leak
# rides the prompt channel.
DECOUPLE_CONDITIONS: list[TokenCondition] = [
    TokenCondition("xy_promptgoodbad", " X", " Y", "neutral", "prompt-channel",
                   "prompt names good/bad, output X/Y (isolates prompt-naming)", prompt_a="good", prompt_b="bad"),
    TokenCondition("goodbad_promptXY", " good", " bad", "neutral", "output-channel",
                   "prompt names X/Y, output good/bad (isolates the output token)", prompt_a="X", prompt_b="Y"),
]

# Label-order control: the X/Y reference with the two labels named in the opposite prompt order ("Y or X").
# If the per-pair Δ ladder rank-correlates with ref_xy, the (constant) order/position prior doesn't move it.
ORDER_CONDITIONS: list[TokenCondition] = [
    TokenCondition("ref_xy_orderswap", " X", " Y", "neutral_swapped", "letters",
                   "X/Y reference with labels named in swapped prompt order (label-order control)"),
]


def condition(key: str) -> TokenCondition:
    return next(c for c in TOKEN_CONDITIONS + SWAP_CONDITIONS + BLIND_CONDITIONS + DECOUPLE_CONDITIONS
                + ORDER_CONDITIONS if c.key == key)


# ----------------------------------------------------------------------------- curated pair subset
# ~28 of the 202 fine-tune pairs, spanning the strength gap, OVER-sampling near-ties (|Δ_ft|<3) where a
# token artifact would most easily flip the winner, with mid-gap and large-gap "should-never-flip"
# anchors. Drawn deterministically from results/_pair_deltas_ft.json by `select_curated_pairs` and
# FROZEN here for reproducibility (the loader asserts the function still reproduces it). `language` pairs
# are excluded: they add a cross-lingual regime (Spanish tokenizes ~25% longer; flagged "fragile" in the
# committed refit) that would be a second, confounding axis on top of the label-token axis.
CURATED_PAIRS: list[str] = [
    # near-ties |Δ_ft| < 3 (12)
    "question_vs_statement__domain_jargon", "figurative_language__sentence_complexity",
    "numerical_density__acronym_density", "temporal_scope__negation_density",
    "argumentative_structure__emotional_tone", "imperative_mood__figurative_language",
    "temporal_scope__first_person_voice", "numerical_density__passive_voice",
    "first_person_voice__negation_density", "temporal_scope__second_person_address",
    "imperative_mood__domain_jargon", "code_presence__emotional_tone",
    # mid-gap 3 <= |Δ_ft| < 8 (8)
    "question_vs_statement__figurative_language", "domain_jargon__narrative_vs_expository",
    "code_presence__figurative_language", "narrative_vs_expository__figurative_language",
    "question_vs_statement__narrative_vs_expository", "imperative_mood__hedging",
    "specificity__emotional_tone", "imperative_mood__second_person_address",
    # large-gap |Δ_ft| >= 11 anchors (8)
    "narrative_vs_expository__argumentative_structure", "second_person_address__sentence_complexity",
    "narrative_vs_expository__first_person_voice", "figurative_language__passive_voice",
    "question_vs_statement__acronym_density", "numerical_density__hedging",
    "formal_register__acronym_density", "specificity__parenthetical_asides",
]

NEAR_TIE_THRESHOLD = 3.0  # |Δ_ft| below which a pair is a "near-tie" (the token-sensitive regime)


def plateau_readout(res: dict) -> dict:
    """Pull the training-loss-plateau checkpoint row from a finetune_and_eval result (the pinned read-out;
    training-signal-only selection, never the held-out cells). Falls back to the last checkpoint if the
    loss never crosses the plateau threshold. Returns the full curve row at that step."""
    import finetune as _ft

    step = _ft.loss_plateau_step(res["train_losses"], [c["step"] for c in res["curve"]])
    if step is None:
        step = res["curve"][-1]["step"]
    return next(c for c in res["curve"] if c["step"] == step)


def select_curated_pairs(pair_deltas_ft: dict[str, float], n_near=12, n_mid=8, n_large=8) -> list[str]:
    """Deterministic gap-spanning selection (near-tie oversampled). Excludes `language` pairs. Kept so
    the frozen CURATED_PAIRS is transparent and re-derivable; `load_curated_pairs` asserts they match."""
    import numpy as np

    cand = [k for k in pair_deltas_ft if "language" not in k.split("__")]
    by_gap = sorted(cand, key=lambda k: abs(pair_deltas_ft[k]))

    def evenly(items, k):
        if len(items) <= k:
            return items
        idx = np.linspace(0, len(items) - 1, k).round().astype(int)
        return [items[i] for i in sorted(set(idx))]

    near = [k for k in by_gap if abs(pair_deltas_ft[k]) < NEAR_TIE_THRESHOLD]
    mid = [k for k in by_gap if NEAR_TIE_THRESHOLD <= abs(pair_deltas_ft[k]) < 8.0]
    large = [k for k in by_gap if abs(pair_deltas_ft[k]) >= 11.0]
    return evenly(near, n_near) + evenly(mid, n_mid) + evenly(large, n_large)
