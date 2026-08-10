"""v2 unified build glue: resolve ANY missing pair from the complete-graph manifest
(`lib/complete_graph.py`) to concrete generation / commit / verification machinery. Extends —
never forks — the existing builders:

  * new×prior, trigger f2   -> extension_build's cell kwargs (incl. the haiku leet-partner judge,
                               now GATED on a per-partner judge-transform screen artifact);
  * new×prior, content f2   -> content_build's cell kwargs (reply phrasings for the 13 non-anchor
                               priors are registered into content_build._REPLY_GEN below);
  * new×new, passage        -> newxnew_build's cell kwargs, plus the previously-unimplemented
                               leet × judged-feature path (screened judge ladder, below);
  * new×new / any pair with a transcript side -> the content-arm transcript mechanism generalized
                               to arbitrary sides (all four cells realized as transcripts);
  * prior×prior             -> the plain passage flow with PROPERTIES + the constraint-rank seed
                               rule (the 8 never-built pairs; per-pair tuned overrides allowed).

LEET × JUDGED-FEATURE JUDGE RESOLUTION (the canonical-design ladder; screens must run first —
`experiments/complete_graph/run_leet_judge_screen.py` writes
results/complete_graph/leet_judge_screens.json):
  * style/surface partner (a PRIOR_21 feature): haiku judges the partner on leet-transformed cells
    (uniform across all four cells) IFF that partner's own judge-transform agreement screen passed
    (the shipped screen covered only the 8 anchor partners). No screen artifact -> refuse to build.
  * content partner: haiku NEVER judges a content feature. Ladder, decided by the same screen
    artifact: (a) sonnet on transformed text if it doesn't refuse and agrees with known poles;
    (b) opus likewise; (c) judge the readable PRE-transform text with the frozen judge and apply
    the deterministic transform after (with an exact transform-identity check per candidate and a
    post-build judged spot-check); (d) documented unbuildable-by-judge.
  * `harmful_vs_benign` × `leetspeak` routes as a HARM pair (Opus recipe wins; leet stays
    surface-validated) — flagged for an Opus-refusal screen before its stage 2 build.

CELL TARGETS: complete_graph.TARGET_PER_CELL (=300) for all four cells; floor
complete_graph.FLOOR_PER_CELL (=150) only with a documented per-pair reason.
"""

from __future__ import annotations

import json
import re

from attrs import frozen

import complete_graph as CG
import config
import content_build as CB
import content_validators as CV
import data_checks
import dataset
import extension_build as EB
import extension_properties as EXT
import extension_validators as EV
import length_match as LM
import newxnew_build as NB
import validation
from properties import CELLS, PROPERTIES, FeatureValue, Property

BUFFER_DIR = config.DATA_DIR / "buffer"
GRID_DIR = config.DATA_DIR / "grid"
LEET_SCREEN_PATH = config.RESULTS_DIR / "complete_graph" / "leet_judge_screens.json"
BUILD_RECORDS_DIR = config.RESULTS_DIR / "complete_graph" / "build_records"

OPUS_MODEL = CB.HARM_GENERATOR_MODEL  # the one pinned Opus id, reused for the leet-content rung (b)


# =====================================================================================
# Reply phrasings for ALL 21 prior partners now live in content_build._REPLY_GEN (the 13
# non-anchor phrasings were added there for v2 — same-typed data in one place, no import-time
# registration). tests/test_v2_build.py asserts full coverage.


# Pinned trigger literals whose exact bytes the surface rule requires — they must survive the
# passage->reply rewrite untouched (END_STRING is "This concludes the passage.", which the naive
# \bpassage\b rewrite would corrupt into a trailer the ends_with_string rule then rejects).
_PROTECTED_TRIGGER_LITERALS = (
    EV.END_STRING, EV.START_STRING, EV.SIGNATURE, EV.WALNUT, EV.VORNETIC, EV.HIGH_ENTROPY,
    EV.PLANT_EMOJI)


def _generic_reply_property(base: Property) -> Property:
    """Reply-phrased variant for features with no hand phrasing (deterministic triggers / passage
    content features appearing in a transcript pair): rewrite 'passage' -> "assistant's reply" and
    "the reader" -> "the user" in the gen instructions and retarget the judge_def, keeping
    everything else identical. Pinned trigger literals (e.g. the ends_with_string trailer "This
    concludes the passage.") are shielded from the rewrite so the surface rule still matches."""

    def _rewrite(text: str) -> str:
        shields = {}
        for i, lit in enumerate(_PROTECTED_TRIGGER_LITERALS):
            if lit in text:
                token = f"\x00LIT{i}\x00"
                shields[token] = lit
                text = text.replace(lit, token)
        text = re.sub(r"\bpassage\b", "assistant's reply", text)
        text = re.sub(r"\bthe reader\b", "the user", text)
        for token, lit in shields.items():
            text = text.replace(token, lit)
        return text

    def _val(src: FeatureValue) -> FeatureValue:
        gen = _rewrite(src.gen_instruction)
        jd = src.judge_def
        jd_reply = jd if jd.lower().startswith(("the assistant", "the reply")) else \
            "In the assistant's reply: " + (jd[0].lower() + jd[1:] if jd else jd)
        return FeatureValue(sign=src.sign, label=src.label, gen_instruction=gen,
                            judge_def=jd_reply, avoid_words=src.avoid_words)

    return Property(name=base.name, plus=_val(base.plus), minus=_val(base.minus),
                    seeds=base.seeds, topic_forced=base.topic_forced,
                    surface_rule=base.surface_rule, kind=base.kind)


def reply_phrase_property(name: str) -> Property:
    """Reply-phrased Property for any feature: hand phrasing where one exists (anchors + the 13
    non-anchor priors via content_build._REPLY_GEN), else the generic passage->reply rewrite."""
    if name in CB._REPLY_GEN:
        return CB.reply_partner(name)
    return _generic_reply_property(EXT.lookup_property(name))


# =====================================================================================
# Leet judge-screen artifact + ladder resolution
# =====================================================================================
def leet_screen_verdicts() -> dict:
    if LEET_SCREEN_PATH.exists():
        return json.loads(LEET_SCREEN_PATH.read_text())["verdicts"]
    return {}


def resolve_leet_partner_judge(partner: str) -> dict:
    """How the non-leet side of a (non-harm) leet pair gets judged. Returns
    {mode: 'judge_free' | 'haiku' | 'sonnet' | 'opus' | 'pretransform', model: str | None}.
    Raises if the required per-feature screen has not been run (or said unbuildable)."""
    if partner in validation.JUDGE_FREE_PROPERTIES:
        return {"mode": "judge_free", "model": None}
    verdicts = leet_screen_verdicts()
    if partner in CG.PRIOR_SET:  # style/surface partner -> haiku, gated on the per-partner screen
        v = verdicts.get(partner)
        if v is None:
            raise RuntimeError(
                f"leet × {partner}: no judge-transform screen on record — run "
                f"experiments/complete_graph/run_leet_judge_screen.py --feature {partner}")
        if not v.get("haiku_ok"):
            # Haiku can't read this partner on the leet surface (negation_density is the one
            # screened case) -> the ladder's pretransform rung: the frozen Sonnet judge + the
            # partner's surface rule both run on the readable PRE-transform text, and every leet+
            # candidate must satisfy the exact transform identity. Sonnet never sees leet text
            # (it refuses ~96% of it), so no rung below this exists for a judged style partner.
            return {"mode": "pretransform", "model": None}
        return {"mode": "haiku", "model": EB.LEET_PARTNER_JUDGE_MODEL}
    # content partner -> the sonnet -> opus -> pretransform ladder (haiku NEVER)
    v = verdicts.get(partner)
    if v is None:
        raise RuntimeError(
            f"leet × {partner}: no content-judge screen on record — run "
            f"experiments/complete_graph/run_leet_judge_screen.py --feature {partner}")
    rung = v.get("content_rung")
    if rung == "unbuildable":
        raise RuntimeError(f"leet × {partner}: unbuildable-by-judge per screen ({v})")
    assert rung in ("sonnet", "opus", "pretransform"), (partner, v)
    return {"mode": rung,
            "model": {"sonnet": None, "opus": OPUS_MODEL, "pretransform": None}[rung]}


def _make_leet_pretransform_validate(leet_side: str, content_model: str | None = None):
    """Rung (c): judge the CONTENT side on the readable pre-transform text (with `content_model`,
    default = frozen Sonnet; Opus for the harm pole); surface rules for the content side also run
    on the pre-transform text; the leet side's own surface rule runs on the final (transformed)
    text; and every candidate must satisfy the exact transform identity nfc->leet(pretransform) ==
    text in leet+ cells (else reject). Used for non-harm leet×content (Sonnet),
    harmful_vs_benign×leetspeak (Opus — the leet screen shows Opus refuses 73-98% of
    leet-transformed content, so it must read the readable text like every other judge), and a
    judged STYLE partner whose haiku judge-transform screen failed (negation_density) — the
    "content side" here is generically the non-leet side."""
    assert leet_side in ("f1", "f2")

    async def _validate(text, f1, f1_sign, f2, f2_sign, pretransform=None):
        assert pretransform is not None
        leet_prop, leet_sign, content_prop, content_sign = (
            (f1, f1_sign, f2, f2_sign) if leet_side == "f1" else (f2, f2_sign, f1, f1_sign))
        # exact transform identity (leet+ cells only; leet- cells are untransformed)
        if leet_sign == "+" and EV.to_leetspeak(EV.nfc(pretransform)) != text:
            return validation.ValidationResult(
                accepted=False, f1_label="unclear", f2_label="unclear", f1_match=False,
                f2_match=False, surface_matches={}, detail={"f1_reason": "transform_identity",
                                                            "f2_reason": "transform_identity"})
        j_content = await validation.judge_feature(pretransform, content_prop, model=content_model)
        content_match = j_content["label"] == content_sign
        surface_matches = {}
        if content_prop.surface_rule:
            surface_matches[content_prop.name] = validation.SURFACE_RULES[
                content_prop.surface_rule](pretransform, content_sign)
        surface_matches[leet_prop.name] = validation.SURFACE_RULES[leet_prop.surface_rule](
            text, leet_sign)
        f1_match = content_match if leet_side == "f2" else surface_matches[leet_prop.name]
        f2_match = content_match if leet_side == "f1" else surface_matches[leet_prop.name]
        f1_label = j_content["label"] if leet_side == "f2" else f1_sign
        f2_label = j_content["label"] if leet_side == "f1" else f2_sign
        return validation.ValidationResult(
            accepted=content_match and all(surface_matches.values()),
            f1_label=f1_label, f2_label=f2_label, f1_match=f1_match, f2_match=f2_match,
            surface_matches=surface_matches,
            detail={"f1_reason": j_content["reason"] if leet_side == "f2" else "surface_rule",
                    "f2_reason": j_content["reason"] if leet_side == "f1" else "surface_rule",
                    "judged_pretransform": True})

    _validate.needs_pretransform = True
    return _validate


def _leet_judged_newxnew_kwargs(pair_key: str, cell_key: str) -> dict:
    """new×new leet × judged-feature. Non-harm: the screened judge ladder for the partner. Harm
    (`harmful_vs_benign` × `leetspeak`): Opus judges the harm pole, but on the readable
    PRE-transform text — the leet screen shows Opus refuses 73-98% of leet content, so the harm
    rule's 'Opus judges the harm pole' is realized via the same pretransform rung as the non-harm
    content ladder (NOT by sending Opus to the transformed text, which silently fails). Generator
    stays Opus for all four cells (the harm recipe)."""
    f1_name, f2_name = EXT.split_pair_key(pair_key)
    s1, s2 = NB._cell_signs(cell_key)
    transforms = [EV.nfc]
    for name, sign in ((f1_name, s1), (f2_name, s2)):
        t = EV.GEN_TRANSFORMS.get((name, sign))
        if t:
            transforms.append(t)
    kwargs: dict = {
        "length_nudge": NB.newxnew_cell_length_nudge(pair_key, cell_key),
        "text_transform": EB._compose(*transforms),
    }
    for name, sign in ((f1_name, s1), (f2_name, s2)):
        if name == "trailing_signature" and sign == "+":
            kwargs["dash_view"] = EV.strip_trailing_signature
        if name == "starts_with_string" and sign == "+":
            kwargs["dedup_view"] = EV.strip_leading_start_string

    leet_side = "f1" if f1_name == "leetspeak" else "f2"
    if "harmful_vs_benign" in (f1_name, f2_name):
        # harm rule: Opus generates all four cells (set here) + judges the harm pole, but on the
        # readable pre-transform text (rung c with content_model=Opus).
        kwargs["generator_model"] = CB.HARM_GENERATOR_MODEL
        kwargs["gen_call_uuid"] = CB.HARM_GEN_UUID
        kwargs["validate_fn"] = _make_leet_pretransform_validate(
            leet_side, content_model=CB.HARM_JUDGE_MODEL)
        return kwargs
    judge = resolve_leet_partner_judge(f2_name if f1_name == "leetspeak" else f1_name)
    if judge["mode"] == "pretransform":
        kwargs["validate_fn"] = _make_leet_pretransform_validate(leet_side)
    else:  # sonnet (frozen default) or opus, judging the transformed text
        model = judge["model"]  # None = frozen sonnet
        kwargs["validate_fn"] = CB._make_partner_and_feature_judge_validate(
            f1_model=model if leet_side == "f2" else None,
            f2_model=model if leet_side == "f1" else None)
    return kwargs


# =====================================================================================
# Transcript realization for ARBITRARY pairs (v2: any partner, either side, incl. new×new)
# =====================================================================================
TRANSCRIPT_X_TRANSCRIPT = "sycophantic_agreement__refusal_language"


def is_transcript_x_transcript(pair_key: str) -> bool:
    f1, f2 = EXT.split_pair_key(pair_key)
    return f1 in CG.TRANSCRIPT_FEATURES and f2 in CG.TRANSCRIPT_FEATURES


def transcript_side_of(pair_key: str) -> str | None:
    """The single transcript feature of a pair (the one whose user-turn bank drives it), or None.
    For the transcript×transcript pair both sides are behaviors realized in the reply and the user
    turn must afford BOTH axes — handled by the combined bank (is_transcript_x_transcript); this
    returns the pair key itself as the bank marker so downstream 'is this a transcript pair?' checks
    (truthiness) still fire."""
    f1, f2 = EXT.split_pair_key(pair_key)
    sides = [f for f in (f1, f2) if f in CG.TRANSCRIPT_FEATURES]
    if len(sides) == 2:
        return pair_key  # transcript × transcript: the combined-bank marker
    return sides[0] if sides else None


def _make_bank(pair_key: str, plan: "V2Plan") -> "CB.UserTurnBank | None":
    """The right user-turn bank for a pair: the combined (two-axis) bank for the transcript×
    transcript pair, a single-feature bank for any other transcript pair, else None."""
    if plan.bank_feature is None:
        return None
    if is_transcript_x_transcript(pair_key):
        return CB.CombinedUserTurnBank()
    return CB.UserTurnBank(plan.bank_feature)


def _reply_scoped(transform):
    """Apply a text transform to the REPLY only (the user turn is byte-fixed by construction)."""

    def _apply(text: str) -> str:
        return CV.assemble_transcript(CV.user_turn_view(text), transform(CV.reply_view(text)))

    return _apply


def _make_v2_transcript_validate(f1_model: str | None, f2_model: str | None):
    """Transcript validate that (unlike content_build's) respects JUDGE-FREE trigger sides: a
    judge-free feature is validated by its surface rule on the REPLY VIEW, never LLM-judged.
    Judged sides are LLM-judged on the FULL transcript (content features need the user turn as
    context). All surface rules run on the reply view."""

    async def _validate(text, f1, f1_sign, f2, f2_sign):
        reply = CV.reply_view(text)
        labels, matches, reasons = {}, {}, {}
        surface_matches = {}
        for tag, prop, sign, model in (("f1", f1, f1_sign, f1_model), ("f2", f2, f2_sign, f2_model)):
            if prop.name in validation.JUDGE_FREE_PROPERTIES:
                ok = validation.SURFACE_RULES[prop.surface_rule](reply, sign)
                labels[tag] = sign if ok else "unclear"
                reasons[tag] = "surface-rule-only (judge skipped)"
            else:
                j = await validation.judge_feature(text, prop, model=model)
                labels[tag] = j["label"]
                reasons[tag] = j["reason"]
                if prop.surface_rule:  # judged feature that also has a surface rule (e.g. entity)
                    surface_matches[prop.name] = validation.SURFACE_RULES[prop.surface_rule](
                        reply, sign)
            matches[tag] = labels[tag] == sign
        return validation.ValidationResult(
            accepted=matches["f1"] and matches["f2"] and all(surface_matches.values()),
            f1_label=labels["f1"], f2_label=labels["f2"],
            f1_match=matches["f1"], f2_match=matches["f2"], surface_matches=surface_matches,
            detail={"f1_reason": reasons["f1"], "f2_reason": reasons["f2"]})

    return _validate


def _make_leet_transcript_validate(leet_side: str):
    """Leet × transcript-behavior pretransform rung (stage 2). The assistant REPLY is leet-
    transformed on leet+ cells while the user turn stays readable (reply-scoped text_transform); the
    BEHAVIOR side (refusal_language / sycophantic_agreement) is judged by the frozen Sonnet judge on
    the READABLE pre-transform full transcript (no frontier judge reads leet reliably — the screen
    lands every behavior on the pretransform rung); leetspeak's own surface rule runs on the reply
    view of the FINAL (transformed) text; and every leet+ candidate must satisfy the exact transform
    identity on the reply portion: nfc->leet(reply_view(pretransform)) == reply_view(text)."""
    assert leet_side in ("f1", "f2")

    async def _validate(text, f1, f1_sign, f2, f2_sign, pretransform=None):
        assert pretransform is not None
        leet_prop, leet_sign, beh_prop, beh_sign = (
            (f1, f1_sign, f2, f2_sign) if leet_side == "f1" else (f2, f2_sign, f1, f1_sign))
        final_reply = CV.reply_view(text)
        pre_reply = CV.reply_view(pretransform)
        # exact transform identity on the reply (leet+ cells only; leet- cells untransformed)
        if leet_sign == "+" and EV.to_leetspeak(EV.nfc(pre_reply)) != final_reply:
            return validation.ValidationResult(
                accepted=False, f1_label="unclear", f2_label="unclear", f1_match=False,
                f2_match=False, surface_matches={}, detail={"f1_reason": "transform_identity",
                                                            "f2_reason": "transform_identity"})
        # behavior judged on the readable pre-transform FULL transcript (needs the user turn)
        j_beh = await validation.judge_feature(pretransform, beh_prop, model=None)
        beh_match = j_beh["label"] == beh_sign
        # leetspeak surface rule on the reply view of the final text
        leet_ok = validation.SURFACE_RULES[leet_prop.surface_rule](final_reply, leet_sign)
        surface_matches = {leet_prop.name: leet_ok}
        f1_match = leet_ok if leet_side == "f1" else beh_match
        f2_match = leet_ok if leet_side == "f2" else beh_match
        f1_label = leet_sign if leet_side == "f1" else j_beh["label"]
        f2_label = leet_sign if leet_side == "f2" else j_beh["label"]
        return validation.ValidationResult(
            accepted=beh_match and leet_ok, f1_label=f1_label, f2_label=f2_label,
            f1_match=f1_match, f2_match=f2_match, surface_matches=surface_matches,
            detail={"f1_reason": (j_beh["reason"] if leet_side == "f2" else "surface_rule"),
                    "f2_reason": (j_beh["reason"] if leet_side == "f1" else "surface_rule"),
                    "judged_pretransform": True})

    _validate.needs_pretransform = True
    return _validate


def _transcript_pair_kwargs(pair_key: str, cell_key: str, bank: CB.UserTurnBank) -> dict:
    """generate_cell_props kwargs for any transcript-realized pair (v2-generalized): byte-fixed
    user turn from the transcript side's bank; the OTHER side realized in the reply; transforms
    reply-scoped; per-side judge models for harm (Opus) and the leet ladder."""
    f1_name, f2_name = EXT.split_pair_key(pair_key)
    s1, s2 = NB._cell_signs(cell_key)
    # A leet+ reply is leet-transformed AFTER generation; leet text tokenizes ~5× longer (each
    # substituted char tends to split into its own token), so a full-length readable reply blows
    # past the 320-token window on leet+ cells (measured: all leet+ cells rejected on length). Shrink
    # the readable reply target by the pinned leet word factor on leet+ cells (the passage arm's
    # WORD_TARGET_FACTORS['leetspeak'] precedent) so the transformed reply lands in window. The user
    # turn stays readable and also counts toward the window, so leave headroom.
    leet_plus = (f1_name == "leetspeak" and s1 == "+") or (f2_name == "leetspeak" and s2 == "+")
    caps_plus = (f1_name == "all_caps" and s1 == "+") or (f2_name == "all_caps" and s2 == "+")
    if leet_plus:
        factor = EXT.WORD_TARGET_FACTORS["leetspeak"]["en"]
        lo, hi = round(120 * factor), round(170 * factor)
        base_nudge = (f"Length: aim for a reply of roughly {lo} to {hi} words "
                      f"(keep it short — it will be transformed). Use at least four short sentences.")
    elif caps_plus:
        # An all_caps+ reply tokenizes longer (uppercase word-pieces split more), so a normal-length
        # reply overshoots the 320-token window on caps+ cells (measured tok_median 300-314 vs ~255
        # on the caps- cells; length fails dominated). Shorten the readable reply target on caps+
        # cells so the uppercased reply lands in window and the two held cells converge.
        base_nudge = "Length: aim for a reply of roughly 85 to 120 words (keep it concise)."
    else:
        base_nudge = (CB.transcript_length_adjust_nudge(f1_name, s1, f2_name, s2)
                      or "Length: aim for a reply of roughly 120 to 170 words.")
    kwargs: dict = {
        "prompt_builder": CB.make_transcript_prompt_builder(bank),
        "text_assembler": CB.make_transcript_assembler(bank),
        "dedup_view": CV.reply_view,
        "length_nudge": base_nudge,
    }
    transforms = [EV.nfc]
    for name, sign in ((f1_name, s1), (f2_name, s2)):
        t = EV.GEN_TRANSFORMS.get((name, sign))
        if t:
            transforms.append(t)
    kwargs["text_transform"] = _reply_scoped(EB._compose(*transforms)) if len(transforms) > 1 \
        else EV.nfc
    for name, sign in ((f1_name, s1), (f2_name, s2)):
        if name == "trailing_signature" and sign == "+":
            kwargs["dash_view"] = EV.strip_trailing_signature

    f1_model = f2_model = None
    if "harmful_vs_benign" in (f1_name, f2_name):
        kwargs["generator_model"] = CB.HARM_GENERATOR_MODEL
        kwargs["gen_call_uuid"] = CB.HARM_GEN_UUID
        f1_model = CB.HARM_JUDGE_MODEL if f1_name == "harmful_vs_benign" else None
        f2_model = CB.HARM_JUDGE_MODEL if f2_name == "harmful_vs_benign" else None
    if "leetspeak" in (f1_name, f2_name):
        # leet × transcript-behavior (refusal_language / sycophantic_agreement): the behavior
        # partner is a content feature whose leet-judge screen lands on the pretransform rung (no
        # frontier judge reads leet reliably). Reply-side leet transform, behavior judged on the
        # readable pre-transform transcript, exact reply-transform identity on leet+ cells.
        partner = f2_name if f1_name == "leetspeak" else f1_name
        judge = resolve_leet_partner_judge(partner)
        if judge["mode"] == "pretransform":
            leet_side = "f1" if f1_name == "leetspeak" else "f2"
            kwargs["validate_fn"] = _make_leet_transcript_validate(leet_side)
            return kwargs
        # (a future style/haiku-ok partner would land here; none exists among transcript pairs)
        assert judge["mode"] in ("sonnet", "opus"), (pair_key, judge)
        if f1_name == "leetspeak":
            f2_model = judge["model"]
        else:
            f1_model = judge["model"]
    kwargs["validate_fn"] = _make_v2_transcript_validate(f1_model=f1_model, f2_model=f2_model)
    return kwargs


# =====================================================================================
# Pair resolution: properties + seed pool + per-cell kwargs for EVERY v2-built family
# =====================================================================================
@frozen
class V2Plan:
    pair: str
    family: str                      # complete_graph families
    realization: str                 # passage | transcript
    f1: Property
    f2: Property
    seed_pool: tuple[str, ...]
    bank_feature: str | None         # transcript side, if any
    generator_model: str             # writer for all four cells
    judge_desc: dict[str, str]       # feature -> judge/validator description (uniform over cells)


def _make_language_partner_validate(lang_side: str):
    """The degenerate rebuild `language__second_person_address` (prior×prior): the partner
    second_person_address carries an ENGLISH-lexicon surface rule ('you'/'your'), which cannot fire
    on the language+ (Spanish) cells — the original shipped grid's A/B train cells collapsed to
    1 text for exactly this reason. On Spanish cells the frozen Sonnet judge (which reads Spanish
    second person — tú/usted/te/tu — correctly) is the authoritative partner label, so the English
    surface rule runs ONLY on the language− (English) cells. Both features are judged on the full
    text (the language judge catches any English leakage in a Spanish cell, so it is a real reject,
    not masked). Same sanctioned view-scoping idea as the num×entropy / code-fence exemptions; the
    frozen rules and stored text are untouched. lang_side names which side is `language`."""
    assert lang_side in ("f1", "f2")

    async def _validate(text, f1, f1_sign, f2, f2_sign):
        lang_sign = f1_sign if lang_side == "f1" else f2_sign
        spanish = lang_sign == "+"
        j1 = await validation._judge_or_surface_only(text, f1, f1_sign)
        j2 = await validation._judge_or_surface_only(text, f2, f2_sign)
        f1_match, f2_match = j1["label"] == f1_sign, j2["label"] == f2_sign
        surface_matches = {}
        for tag, prop, sign in (("f1", f1, f1_sign), ("f2", f2, f2_sign)):
            is_partner = tag != lang_side
            if prop.surface_rule and not (is_partner and spanish):
                surface_matches[prop.name] = validation.SURFACE_RULES[prop.surface_rule](text, sign)
        return validation.ValidationResult(
            accepted=f1_match and f2_match and all(surface_matches.values()),
            f1_label=j1["label"], f2_label=j2["label"], f1_match=f1_match, f2_match=f2_match,
            surface_matches=surface_matches,
            detail={"f1_reason": j1["reason"], "f2_reason": j2["reason"]})

    return _validate


def _make_code_partner_validate():
    """prior×prior code_presence pairs: the frozen `parenthetical` / `number_dense` surface rules
    count tokens INSIDE fenced code (a `(` in `def f(x)`, digits in a loop), so a (code+, partner−)
    cell can never pass even when the PROSE has no asides / no figures — the construct
    ('parenthetical asides in the writing', 'quantitative figures in the prose') is about the prose,
    not code syntax. So a partner's surface rule runs on the code-STRIPPED text, while code_presence's
    own `has_code` rule runs on the full text (it needs the code) and both features are judged on the
    full text. Same sanctioned wrapper pattern as extension_build._validate_num_entropy_exempt; the
    frozen rules are untouched, only the VIEW they see changes, and the stored grid text is unchanged
    (the classifier still reads the full text incl. code)."""

    async def _validate(text, f1, f1_sign, f2, f2_sign):
        prose = data_checks.strip_code_fences(text)
        j1 = await validation._judge_or_surface_only(text, f1, f1_sign)
        j2 = await validation._judge_or_surface_only(text, f2, f2_sign)
        f1_match, f2_match = j1["label"] == f1_sign, j2["label"] == f2_sign
        surface_matches = {}
        for prop, sign in ((f1, f1_sign), (f2, f2_sign)):
            if prop.surface_rule:
                view = text if prop.name == "code_presence" else prose
                surface_matches[prop.name] = validation.SURFACE_RULES[prop.surface_rule](view, sign)
        return validation.ValidationResult(
            accepted=f1_match and f2_match and all(surface_matches.values()),
            f1_label=j1["label"], f2_label=j2["label"], f1_match=f1_match, f2_match=f2_match,
            surface_matches=surface_matches,
            detail={"f1_reason": j1["reason"], "f2_reason": j2["reason"]})

    return _validate


# Per-pair tuned overrides for the 8 never-built prior×prior pairs (documented remediations of
# the prior study's drop reasons; the SCREEN decides if they suffice).
PRIOR_PAIR_SEED_OVERRIDES = {
    # dropped as DEGENERATE-REALIZATION: "the imperative seed pool affords few natural acronyms" —
    # the tuned attempt hands the pool to acronym_density's TECH-leaning seeds instead.
    "imperative_mood__acronym_density": "acronym_density",
}


def resolve_v2_pair(pair_key: str) -> V2Plan:
    CG.assert_canonical_orientation(pair_key)
    f1_name, f2_name = EXT.split_pair_key(pair_key)
    family = CG.pair_family(f1_name, f2_name)
    route = CG.route_pair(pair_key)
    tside = transcript_side_of(pair_key)

    if is_transcript_x_transcript(pair_key):
        # transcript × transcript: BOTH sides are behaviors realized in the reply (each keeps its
        # own reply-targeted Property); the combined user-turn bank supplies a turn that affords
        # both axes. Seeds are the combined (false-premise + declinable-request) pool.
        import tt_seeds
        f1, f2 = EXT.lookup_property(f1_name), EXT.lookup_property(f2_name)
        seed_pool = tt_seeds.TT_SEEDS
        assert seed_pool, pair_key
    elif tside is not None:
        # ALL other transcript pairs (any family): the transcript feature keeps its own Property
        # (its gen instruction already targets the reply); the partner is reply-phrased.
        f1 = EXT.lookup_property(f1_name) if f1_name == tside else reply_phrase_property(f1_name)
        f2 = EXT.lookup_property(f2_name) if f2_name == tside else reply_phrase_property(f2_name)
        seed_pool = tuple(EXT.lookup_property(tside).seeds)
        assert seed_pool, tside
    elif family == "prior_x_prior":
        # EXT.lookup_property resolves `language` (outside PROPERTIES) too — needed for the
        # degenerate rebuild language__second_person_address (prior×prior, core_grid route).
        f1, f2 = EXT.lookup_property(f1_name), EXT.lookup_property(f2_name)
        seed_owner = PRIOR_PAIR_SEED_OVERRIDES.get(pair_key) or CG.prior_pair_seed_property(
            f1_name, f2_name)
        seed_pool = tuple(EXT.lookup_property(seed_owner).seeds)
    elif family == "new_x_prior":
        if CB.is_content_pair(pair_key):
            f1, f2, seed_pool = CB.resolve_content_pair(pair_key)
        else:
            f1, f2, seed_pool = EB.ext_pair(pair_key)
    else:  # new_x_new, passage
        f1, f2, seed_pool = NB.newxnew_pair(pair_key)

    generator = CB.HARM_GENERATOR_MODEL if "harmful_vs_benign" in (f1_name, f2_name) \
        else config.GENERATOR_MODEL
    judge_desc = dict(route.validation_by_feature)
    if family == "prior_x_prior" and "language" in (f1_name, f2_name):
        # The v2 core_grid path judges every feature with the frozen default (Sonnet) via
        # validate_text — uniform across cells, matching every other prior×prior pair. The route's
        # static "judge:haiku (shipped language precedent)" reflects the ORIGINAL degenerate grid's
        # build, not the v2 rebuild. Correct the annotation so the build record is honest
        # (annotation only; Spanish-vs-English is trivially separable for Sonnet).
        judge_desc["language"] = "judge:sonnet-4-6 (v2 core_grid default; frozen validate_text)"
    # The route's static validation_by_feature is computed before the leet screen verdict is known,
    # so for a judged style partner whose haiku leet-screen FAILED it still reads "judge:haiku on
    # leet-transformed cells" even though the build actually uses the pretransform rung. Correct the
    # recorded description so the build record reflects the machinery that ran (annotation only).
    if "leetspeak" in (f1_name, f2_name):
        partner = f2_name if f1_name == "leetspeak" else f1_name
        # Annotation only — must not gate on the screen (resolve_v2_pair is screen-independent; the
        # screen gate lives in v2_cell_gen_kwargs). Correct the description iff the screen verdict
        # is already on record AND routes to the pretransform rung; otherwise leave the static route
        # description untouched.
        if partner not in validation.JUDGE_FREE_PROPERTIES \
                and "harmful_vs_benign" not in (f1_name, f2_name) \
                and leet_screen_verdicts().get(partner, {}).get("haiku_ok") is False:
            judge_desc[partner] = ("judge:sonnet-4-6 on the readable PRE-transform text + surface_rule "
                                   "(pretransform rung; haiku leet-judge screen failed)")
    return V2Plan(pair=pair_key, family=family,
                  realization="transcript" if tside else "passage",
                  f1=f1, f2=f2, seed_pool=tuple(seed_pool), bank_feature=tside,
                  generator_model=generator,
                  judge_desc=judge_desc)


def v2_cell_gen_kwargs(pair_key: str, cell_key: str, bank: CB.UserTurnBank | None = None) -> dict:
    """Per-cell generate_cell_props kwargs for any v2-built pair (dispatch by family/path).
    Resolving a leet × judged pair RAISES unless its judge screen artifact exists — the gate the
    canonical design requires."""
    f1_name, f2_name = EXT.split_pair_key(pair_key)
    family = CG.pair_family(f1_name, f2_name)
    tside = transcript_side_of(pair_key)
    if tside is not None:
        assert bank is not None, f"{pair_key}: transcript pair needs a UserTurnBank"
        if family == "new_x_prior":
            # the content arm's own transcript kwargs (partner-scoped length nudges included)
            return CB.content_cell_gen_kwargs(pair_key, cell_key, bank=bank)
        return _transcript_pair_kwargs(pair_key, cell_key, bank)

    if family == "prior_x_prior":
        # plain passage flow — matches the prior-family grids' build convention (no transforms).
        # The one language pair (the degenerate rebuild) keeps the language_derisk Spanish word
        # nudge on language+ cells (Spanish tokenizes ~25% longer in Qwen; the nudge converges the
        # held cells' token distributions at the source — byte-identical to run_language_generate's
        # ES_NUDGE, so the rebuild matches the shipped language__* convention).
        if "language" in (f1_name, f2_name):
            s1, s2 = NB._cell_signs(cell_key)
            lang_side = "f1" if f1_name == "language" else "f2"
            lang_sign = s1 if lang_side == "f1" else s2
            partner = f2_name if f1_name == "language" else f1_name
            kwargs: dict = {}
            if lang_sign == "+":
                kwargs["length_nudge"] = "Length: aim for roughly 125 to 150 words (keep close to this range)."
            # A partner with an ENGLISH-lexicon surface rule (second_person here) can't be checked by
            # that rule on Spanish cells — the Sonnet judge is authoritative there (this is what made
            # the shipped grid degenerate). Run the English rule only on English cells.
            if EXT.lookup_property(partner).surface_rule:
                kwargs["validate_fn"] = _make_language_partner_validate(lang_side)
            return kwargs
        # code_presence prior×prior pairs: the partner surface rule (parenthetical / number_dense)
        # mis-fires on tokens INSIDE fenced code, so a (code+, partner−) cell can't pass even with
        # clean prose. Run the partner rule on the code-stripped prose (the frozen rules + the judge
        # + the stored text are unchanged; only the surface VIEW changes).
        if f1_name == "code_presence" or f2_name == "code_presence":
            partner = f2_name if f1_name == "code_presence" else f1_name
            if EXT.lookup_property(partner).surface_rule:
                return {"validate_fn": _make_code_partner_validate()}
        return {}
    if family == "new_x_prior":
        if CB.is_content_pair(pair_key):
            return CB.content_cell_gen_kwargs(pair_key, cell_key, bank=None)
        # trigger f2 (extension arm); the leet-partner haiku judge inside is GATED here
        if f2_name == "leetspeak":
            judge = resolve_leet_partner_judge(f1_name)  # raises without a screen
            kwargs = EB.cell_gen_kwargs(pair_key, cell_key)
            if judge["mode"] == "pretransform":
                # judged style partner whose haiku screen failed (negation_density): judge +
                # surface-rule the partner on the readable pre-transform text (frozen Sonnet),
                # exact transform identity on leet+ cells (validate_fn.needs_pretransform).
                kwargs["validate_fn"] = _make_leet_pretransform_validate("f2")
            else:
                assert judge["mode"] == "haiku", judge
            return kwargs
        return EB.cell_gen_kwargs(pair_key, cell_key)
    # new×new
    judged = {f1_name, f2_name} - set(validation.JUDGE_FREE_PROPERTIES)
    # leet × judged-feature (incl. harm×leet): the pretransform-judge path. Harm×leet uses Opus on
    # the readable text (the leet screen shows Opus refuses transformed content); non-harm uses the
    # screened Sonnet/Opus ladder. Either way NOT the doomed "send the judge to the leet text".
    if "leetspeak" in (f1_name, f2_name) and judged:
        return _leet_judged_newxnew_kwargs(pair_key, cell_key)
    return NB.newxnew_cell_gen_kwargs(pair_key, cell_key)


# =====================================================================================
# Buffered generation / screen / commit / verify — ONE path for every v2 pair
# =====================================================================================
async def generate_v2_pair_buffer(pair_key: str, n_target: int, verbose: int = 1,
                                  screen_n: int | None = None,
                                  max_candidates: int = 4000) -> list[dict]:
    """Generate all four cells to data/buffer/ (full build), or — with screen_n — evaluate a fixed
    small candidate batch per cell (the buildability screen; cache-aligned prefix of the full
    build, nothing wasted). Returns per-cell stats rows (screen mode adds per-feature judge-match
    rates + the rejected records for eyeballing). max_candidates raises the per-cell candidate cap
    for the genuine tuned attempt on a low-accept tension cell (default = generation's 4000)."""
    import asyncio

    import generation
    import numpy as np

    plan = resolve_v2_pair(pair_key)
    bank = _make_bank(pair_key, plan)
    if bank is not None:
        await bank.ensure(plan.seed_pool)

    screen = screen_n is not None

    async def gen_cell(cell_key: str) -> dict:
        gen_kwargs = v2_cell_gen_kwargs(pair_key, cell_key, bank=bank)
        if screen:
            res = await generation.generate_cell_props(
                pair_key, cell_key, plan.f1, plan.f2, plan.seed_pool,
                n_target=10 ** 9, max_candidates=screen_n, batch_size=screen_n,
                collect_rejects=True, verbose=verbose, **gen_kwargs)
        else:
            res = await generation.generate_cell_props(
                pair_key, cell_key, plan.f1, plan.f2, plan.seed_pool,
                n_target=n_target, max_candidates=max_candidates, verbose=verbose, **gen_kwargs)
            dataset.save_cell(res["accepted"], pair_key, cell_key, base_dir=BUFFER_DIR)
        toks = np.array([r["num_tokens"] for r in res["accepted"]]) if res["accepted"] else np.array([0])
        row = {"pair": pair_key, "cell": cell_key,
               "n_target": screen_n if screen else n_target, **res["stats"],
               "tok_median": float(np.median(toks)),
               "generator_model": plan.generator_model, "judges": plan.judge_desc}
        if screen:
            row["judge_accept"] = _screen_judge_accept(res)
            row["feature_match_rates"] = _screen_feature_match(res, plan)
            row["rejected"] = [{k: r.get(k) for k in ("text", "seed", "validation", "reasons")}
                               for r in res.get("rejected", [])][:200]
            row["accepted_sample"] = [r["text"] for r in res["accepted"][:20]]
        return row

    return await asyncio.gather(*[gen_cell(ck) for ck, *_ in CELLS])


def _screen_judge_accept(res: dict) -> float:
    """Judge-accept among cheap-passing candidates: (accepted + dedup-failed) / cheap-passed —
    the established buildability signal (dedup fails passed the judge)."""
    s = res["stats"]
    cheap_pass = s["n_candidates"] - s["cheap_fail"]
    return (s["n_accepted"] + s["dedup_fail"]) / max(1, cheap_pass)


def _screen_feature_match(res: dict, plan: V2Plan) -> dict:
    """Per-feature judge-match rates among judged candidates (which side is the bottleneck? —
    the contradictory-cell diagnostic)."""
    rows = [r["validation"] for r in (res.get("rejected", []) + res["accepted"])
            if "validation" in r]
    if not rows:
        return {}
    return {plan.f1.name: sum(v["f1_match"] for v in rows) / len(rows),
            plan.f2.name: sum(v["f2_match"] for v in rows) / len(rows),
            "n_judged": len(rows)}


SCREEN_MIN_JUDGE_ACCEPT = 0.30  # the prior study's bar (pairs_expanded drop log)


def screen_verdict(stats_rows: list[dict]) -> dict:
    """The buildability verdict for one pair from its four screen rows: pass iff every cell's
    judge-accept >= the established 30% bar; report the worst cell + dominant failure reason
    (contradictory-cell detection: an A-cell dominated by judge_f1/judge_f2 fails)."""
    cells = {r["cell"]: r for r in stats_rows}
    ja = {c: r["judge_accept"] for c, r in cells.items()}
    worst = min(ja, key=ja.get)
    fail_reasons = cells[worst].get("fail_reasons", {})
    dominant = max(fail_reasons, key=fail_reasons.get) if fail_reasons else None
    return {"judge_accept_by_cell": ja, "min_cell_judge_accept": ja[worst],
            "worst_cell": worst, "worst_cell_dominant_fail": dominant,
            "passes": ja[worst] >= SCREEN_MIN_JUDGE_ACCEPT,
            "feature_match_by_cell": {c: r.get("feature_match_rates") for c, r in cells.items()}}


def commit_v2_pair(pair_key: str, target: int = CG.TARGET_PER_CELL, write: bool = True,
                   grid_dir=None) -> dict:
    """Buffer -> normalized (transcript-aware) -> standing commit flow -> data/grid/ (or
    `grid_dir` — the 300-vs-200 comparability variants commit the SAME buffer at target=200 into
    the non-canonical data/grid_200cell_compare/)."""
    plan = resolve_v2_pair(pair_key)
    transcript = plan.realization == "transcript"
    grid = CB.content_normalize_grid(dataset.load_grid(pair_key, base_dir=BUFFER_DIR), transcript)
    views = []
    if "starts_with_string" in (plan.f1.name, plan.f2.name):
        views.append(EV.strip_leading_start_string)
    if transcript:
        views.append(CV.reply_view)
    dedup_view = EB._compose(*views) if views else None
    _committed, report = LM.commit_grid_flow(pair_key, grid, target, dedup_view=dedup_view,
                                             write=write,
                                             grid_dir=grid_dir if grid_dir is not None else GRID_DIR)
    report["qa_flags"] = _commit_qa_flags(report, target)
    return {"transcript": transcript, **report}


def _commit_qa_flags(report: dict, target: int) -> dict:
    """QA flags a committed grid should be checked against before it is trusted in a readout. The
    length gate is ENFORCED by match_pair_tuned; the held-cell TOPIC gate is only MEASURED by the
    commit flow, so surface it here (a held set separable by topic can bias the metric the same way
    a length gap would — length_match.held_topic_separability docstring). Also flag held cells
    below the floor."""
    topic_cv = report.get("held_vs_held_topic", {}).get("cv")
    topic_ceil = report.get("held_topic_perm_ceiling")
    held_n = min(report["n_per_cell"][c] for c in ("held_f1pos", "held_f1neg"))
    # margin above the permutation ceiling before flagging: LM.TOPIC_MARGIN — the same criterion
    # that triggers commit_grid_flow's joint (length x seed) fallback, so a committed grid can
    # only carry this flag if even the joint match could not balance it (should be impossible:
    # identical per-seed counts put the seed classifier at chance).
    flags = {
        "held_topic_separable": bool(topic_cv is not None and topic_ceil is not None
                                     and topic_cv > topic_ceil + LM.TOPIC_MARGIN),
        "held_topic_cv": topic_cv, "held_topic_ceiling": topic_ceil,
        "held_below_floor": held_n < CG.FLOOR_PER_CELL, "held_n": held_n,
    }
    flags["ok"] = not (flags["held_topic_separable"] or flags["held_below_floor"])
    return flags


def verify_v2_grid(pair_key: str, base_dir=None) -> dict:
    """ONE exact verifier for every v2-built pair (0 failures required). Composes generically:
      * writer/judge identity: every record in all four cells carries the SAME generator_model +
        gen_uuid (never varying with cell label); harm pairs additionally pinned to Opus;
      * each side's deterministic surface rule at that cell's sign — evaluated on the reply view
        for transcript pairs, with the numerical_density×substring_high_entropy strip exemption;
        a NON-transform partner's rule is re-checked on the readable pre-transform text on any cell
        where a transform (all_caps/leetspeak) is applied, while each transform feature's own rule
        stays on the final text (covers trigger×transform and the all_caps__leetspeak double);
      * transcript pairs: well-formed 2-block transcript; the user turn BYTE-IDENTICAL across all
        four cells per seed; non-empty reply;
      * dash discipline (trailing-signature exemption), the frozen token window;
      * position spread for insertion triggers (on the reply view for transcripts);
      * exact transform identity on every transformed cell: committed text ==
        compose(nfc, transforms)(text_pretransform) — reply-scoped for transcript leet pairs."""
    import numpy as np

    base_dir = base_dir or GRID_DIR
    plan = resolve_v2_pair(pair_key)
    grid = dataset.load_grid(pair_key, base_dir=base_dir)
    f1_name, f2_name = plan.f1.name, plan.f2.name
    transcript = plan.realization == "transcript"
    is_harm = "harmful_vs_benign" in (f1_name, f2_name)
    strip_entropy = {f1_name, f2_name} == {"numerical_density", "substring_high_entropy"}
    # generate-then-transform pairs: the char transform corrupts a NON-transform partner's surface
    # rule on the + cells, so that partner rule is re-checked on the readable pre-transform text
    # (matches extension_build._make_transform_partner_validate). Each TRANSFORM feature's OWN rule
    # (all_caps/leetspeak) still runs on the FINAL text — including both sides of the double
    # transform all_caps__leetspeak. `transform_features` may hold one or two names.
    transform_features = {n for n in (f1_name, f2_name) if n in ("all_caps", "leetspeak")}

    def _expected_transform(s1: str, s2: str):
        """The exact composed text transform generation applied for a cell (nfc then each
        transform feature at + in (f1, f2) order) — reconstructs newxnew_cell_gen_kwargs."""
        fns = [EV.nfc]
        for name, sign in ((f1_name, s1), (f2_name, s2)):
            t = EV.GEN_TRANSFORMS.get((name, sign))
            if t:
                fns.append(t)
        return EB._compose(*fns)

    rules = {}
    for name in (f1_name, f2_name):
        prop = EXT.lookup_property(name)
        if prop.surface_rule:
            rules[name] = validation.SURFACE_RULES[prop.surface_rule]
    needles = {n: EV.POSITION_NEEDLES[n] for n in (f1_name, f2_name) if n in EV.POSITION_NEEDLES}

    failures = []
    provenance = set()
    user_turns_by_seed: dict[str, set[str]] = {}
    rel_positions: dict[str, list[float]] = {n: [] for n in needles}
    for cell in LM.ALL:
        for r in grid[cell]:
            text, ci = r["text"], r.get("candidate_index", "?")
            signs = {f1_name: r["f1_sign"], f2_name: r["f2_sign"]}
            provenance.add((cell, r.get("generator_model", plan.generator_model),
                            r.get("gen_uuid", "gen-v2")))
            surface_view = CV.reply_view(text) if transcript else text
            if transcript:
                if not CV.is_transcript(text):
                    failures.append((cell, ci, "not_transcript"))
                    continue
                user_turns_by_seed.setdefault(r["seed"], set()).add(CV.user_turn_view(text))
                if not CV.reply_view(text).strip():
                    failures.append((cell, ci, "empty_reply"))
            any_transform_plus = any(signs.get(t) == "+" for t in transform_features)
            code_partner_pair = ("code_presence" in (f1_name, f2_name) and plan.family == "prior_x_prior")
            # language pair: the partner's English-lexicon surface rule can't fire on Spanish cells
            # (language+), where the judge is authoritative — skip it there (matches the build path).
            lang_spanish = "language" in (f1_name, f2_name) and signs.get("language") == "+"
            for name, rule in rules.items():
                if lang_spanish and name != "language":
                    continue  # partner English rule not applicable on a Spanish cell
                view = surface_view
                if strip_entropy and name == "numerical_density":
                    view = re.sub(re.escape(EV.HIGH_ENTROPY), " ", view)
                # A NON-transform feature's surface rule is corrupted on cells where a transform is
                # applied, so it is re-checked on the readable pre-transform text (a transform
                # feature's OWN rule always stays on the final text). For transcript pairs the
                # transform is reply-scoped, so the pre-transform reply view is the readable text.
                if name not in transform_features and any_transform_plus and "text_pretransform" in r:
                    pre = r["text_pretransform"]
                    view = CV.reply_view(pre) if transcript else pre
                # code_presence prior×prior: the PARTNER surface rule (parenthetical/number_dense)
                # counts tokens inside fenced code, so it runs on the code-stripped prose (matches
                # _make_code_partner_validate); code_presence's own has_code rule keeps the full text.
                if code_partner_pair and name != "code_presence":
                    view = data_checks.strip_code_fences(view)
                if not rule(view, signs[name]):
                    failures.append((cell, ci, f"{name}_rule_{signs[name]}"))
            dash_text = (EV.strip_trailing_signature(text)
                         if signs.get("trailing_signature") == "+" else text)
            if data_checks.has_forbidden_dash(dash_text):
                failures.append((cell, ci, "dash"))
            if not (config.LENGTH_MIN_TOKENS <= r["num_tokens"] <= config.LENGTH_MAX_TOKENS):
                failures.append((cell, ci, f"length={r['num_tokens']}"))
            if is_harm:
                if r.get("generator_model") != CB.HARM_GENERATOR_MODEL:
                    failures.append((cell, ci, f"harm_generator={r.get('generator_model')}"))
                if r.get("gen_uuid") != CB.HARM_GEN_UUID:
                    failures.append((cell, ci, f"harm_gen_uuid={r.get('gen_uuid')}"))
            # Exact transform identity on every cell where a transform is applied: the committed
            # text must equal the deterministic transform of the stored readable pre-transform text
            # (covers single all_caps/leetspeak, the composed all_caps__leetspeak double transform,
            # and the reply-scoped transcript leet transform). This is the guarantee a downstream
            # reader relies on to recover the readable text.
            if any_transform_plus:
                pre = r.get("text_pretransform")
                expected = _expected_transform(r["f1_sign"], r["f2_sign"])
                if transcript:  # reply-scoped: only the reply is transformed, user turn readable
                    ident_ok = pre is not None and \
                        expected(CV.reply_view(pre)) == CV.reply_view(text)
                else:
                    ident_ok = pre is not None and expected(pre) == text
                if not ident_ok:
                    failures.append((cell, ci, "transform_identity"))
            for name, needle in needles.items():
                if signs[name] == "+":
                    pos = EV.trigger_sentence_index(surface_view, needle)
                    if pos is not None:
                        rel_positions[name].append(
                            pos / max(1, len(EV.sentences(surface_view)) - 1))

    # generator/judge identity must be uniform ACROSS cells (never varies with cell label)
    by_cell = {}
    for cell, gm, gu in provenance:
        by_cell.setdefault(cell, set()).add((gm, gu))
    idents = set().union(*by_cell.values()) if by_cell else set()
    if len(idents) > 1:
        failures.append(("provenance", "-", f"generator identity varies: {sorted(idents)}"))

    multi_turn_seeds = {s for s, v in user_turns_by_seed.items() if len(v) > 1}
    for s in list(multi_turn_seeds)[:10]:
        failures.append(("user_turn", s, "user_turn_not_byte_fixed"))

    report = {
        "pair": pair_key, "family": plan.family, "realization": plan.realization,
        "n_per_cell": {c: len(grid[c]) for c in LM.ALL},
        "generator_identity": sorted(idents),
        "judges": plan.judge_desc,
        "n_failures": len(failures), "failures": failures[:20], "ok": not failures,
    }
    for name, rel in rel_positions.items():
        if rel:
            arr = np.array(rel)
            report[f"{name}_position_rel"] = {
                "mean": float(arr.mean()), "p10": float(np.percentile(arr, 10)),
                "p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90))}
    if transcript:
        report["n_seeds_with_user_turn"] = len(user_turns_by_seed)
        report["n_seeds_user_turn_not_fixed"] = len(multi_turn_seeds)
    return report
