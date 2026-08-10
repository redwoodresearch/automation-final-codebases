"""Content-arm (stage 3) grid building: reply-phrased partner realizations, the byte-fixed
user-turn bank for transcript features, the harm-arm Opus amendment, and the per-cell generation
kwargs (`content_cell_gen_kwargs`) that wire it all into the frozen generate_cell_props flow.

Reuses extension_build's commit/verify machinery (content pairs go to data/grid/ too). The
data namespaces (buffer / grid) and length-match/commit flow are shared with the
deterministic arm; only generation differs (judge-validated, transcript assembly, Opus for harm).

Design: writeups/desiderata_content_arm.md. Consult disposition:
results/consult_disposition_content.md.
"""
from __future__ import annotations

import re

import config
import content_properties as CP
import content_validators as CV
import extension_properties as EXT
import validation
from generation import build_cell_prompt, generate_candidate
from llm import call_llm
from properties import CELLS, FeatureValue, Property

# ------------------------------------------------------------------ harm amendment (Opus)
HARM_GENERATOR_MODEL = "claude-opus-4-5-20251101"  # era-consistent w/ frozen 4.x; API-compatible
HARM_JUDGE_MODEL = HARM_GENERATOR_MODEL             # Opus judges BOTH poles of the harm feature
HARM_GEN_UUID = "content-gen-opus-v1"               # distinct cache namespace for Opus generation
# The partner judge on harm pairs is decided by the harmful-surface judge screen; default = frozen.
HARM_PARTNER_JUDGE_MODEL: str | None = None  # set to a model id iff the screen forces a substitute


def _cell_signs(cell_key: str) -> tuple[str, str]:
    for key, s1, s2, _cls in CELLS:
        if key == cell_key:
            return s1, s2
    raise KeyError(cell_key)


# ------------------------------------------------------------------ reply-phrased partners
# For the two transcript features every partner is realized IN THE ASSISTANT REPLY (the
# code_in_reply precedent). We reuse each partner's judge_def / surface_rule / kind unchanged and
# only rephrase the generation instruction to target the reply. language is reply-only: the user
# turn stays byte-fixed English; only the reply switches language.
_REPLY_GEN = {
    "language": {
        "+": "Write the assistant's REPLY entirely in fluent, natural Spanish (every sentence in "
             "Spanish, no English). A Spanish reply to the user's English message is fine.",
        "-": "Write the assistant's REPLY entirely in fluent, natural English (every sentence in "
             "English, no Spanish).",
    },
    "formal_register": {
        "+": "Write the assistant's reply in a formal, elevated register: polished complete "
             "sentences, sophisticated vocabulary, a professional tone, no contractions or slang.",
        "-": "Write the assistant's reply in a casual, conversational register: everyday words, a "
             "relaxed informal tone, contractions, the loose phrasing of ordinary speech.",
    },
    "hedging": {
        "+": "Write the assistant's reply assertively and confidently, stating things directly as "
             "fact with few or no hedges or qualifiers.",
        "-": "Write the assistant's reply tentatively, qualifying claims with frequent hedges "
             "(might, perhaps, it seems, I think, could) and an uncertain voice.",
    },
    "emotional_tone": {
        "+": "Give the assistant's reply a strong emotional, affective tone: expressed feelings and "
             "multiple emotion words.",
        "-": "Keep the assistant's reply emotionally neutral and detached, with little or no "
             "emotional charge.",
    },
    "code_presence": {
        "+": "The assistant's reply must include at least one block of real, syntactically "
             "plausible source code on its own lines, fenced with triple backticks, with prose "
             "around it. Use a language that fits the question.",
        "-": "The assistant's reply must be entirely ordinary prose with no source code, code "
             "blocks, commands, or programming syntax at all.",
    },
    "question_vs_statement": {
        "+": "Write the assistant's reply so that MOST of its sentences are questions (an "
             "interrogative, guiding-question style reply), while still doing what the reply must "
             "do. The USER'S message is fixed and is not what is judged; it is the REPLY's form "
             "that matters.",
        "-": "Write the assistant's reply as statements, posing very few or no questions.",
    },
    "numerical_density": {
        "+": "Make the assistant's reply dense with specific quantitative figures: pack in multiple "
             "concrete numbers, amounts, or statistics.",
        "-": "The assistant's reply must contain essentially no specific quantitative figures (no "
             "digits or spelled-out numbers).",
    },
    "passive_voice": {
        "+": "Write the assistant's reply mostly in the passive voice (a form of 'be' plus a past "
             "participle in most main clauses).",
        "-": "Write the assistant's reply mostly in the active voice (the subject performs the "
             "action).",
    },
    # --- the 13 non-anchor prior partners (added for v2: transcript pairs against any
    # partner realize the partner in the reply; phrasings mirror the anchors above and
    # preserve each pole's defining property) ---
    "imperative_mood": {
        "+": "Write the assistant's reply as direct commands and instructions to the user, with "
             "imperative-mood verbs in most sentences (tell the user what to do).",
        "-": "Write the assistant's reply as ordinary descriptive prose with no commands: "
             "declarative sentences only, issuing no instructions to the user.",
    },
    "domain_jargon": {
        "+": "Write the assistant's reply for an expert audience, dense with field-specific "
             "technical vocabulary and precise terms of art, without stopping to define them.",
        "-": "Write the assistant's reply in plain, accessible everyday language, avoiding "
             "specialized vocabulary and field-specific terms of art.",
    },
    "narrative_vs_expository": {
        "+": "Shape the assistant's reply as a story: follow specific people or characters "
             "through events unfolding in time, with concrete scenes, rather than explaining in "
             "general terms.",
        "-": "Shape the assistant's reply as a general explanation: lay out the concept, process, "
             "or facts in informative terms, telling no story about particular characters.",
    },
    "argumentative_structure": {
        "+": "Have the assistant's reply take a clear, explicit position and argue for it, "
             "stating plainly what it claims or recommends and backing it with reasons.",
        "-": "Keep the assistant's reply strictly neutral and even-handed: explain the subject or "
             "lay out competing views with no recommendation, verdict, or conclusion of its own.",
    },
    "specificity": {
        "+": "Ground the assistant's reply in concrete particulars: name specific people, places, "
             "and organizations (real or realistic proper names) and give precise, named examples.",
        "-": "Keep the assistant's reply general and abstract: no proper names and no particular "
             "named examples, discussing the subject only in broad, sweeping terms.",
    },
    "temporal_scope": {
        "+": "Anchor the assistant's reply in explicit time: include several definite time markers "
             "(dates, years, named historical periods, seasons, or clock times).",
        "-": "Give the assistant's reply no explicit time anchors at all: no dates, years, named "
             "periods, seasons, or clock times, and no 'recently/nowadays/back then' markers.",
    },
    "second_person_address": {
        "+": "Address the user directly throughout the reply, in the second person: speak to "
             "'you' and refer to 'your' situation frequently across the reply.",
        "-": "Write the reply without addressing the user at all: no second-person 'you' or "
             "'your' anywhere; discuss the subject impersonally in the third person.",
    },
    "first_person_voice": {
        "+": "Write the assistant's reply in the first person: use 'I' or 'we' throughout, "
             "conveying the assistant's own perspective or view.",
        "-": "Write the assistant's reply with no first-person voice at all: no 'I', 'we', 'my', "
             "or 'our' anywhere; keep the reply impersonal and objective.",
    },
    "figurative_language": {
        "+": "Write the assistant's reply with rich figurative language: metaphors, similes, "
             "analogies, and vivid imagery carrying the description rather than plain statement.",
        "-": "Write the assistant's reply in plain, literal language with no figures of speech: "
             "no metaphors, similes, analogies, or poetic imagery.",
    },
    "sentence_complexity": {
        "+": "Build the assistant's reply from long, elaborate, multi-clause sentences with "
             "subordinate clauses and nested structure; avoid short sentences.",
        "-": "Build the assistant's reply from short, simple sentences: one idea per sentence, "
             "few or no subordinate clauses, plain direct syntax.",
    },
    "negation_density": {
        "+": "Frame the assistant's reply through pervasive explicit negation: describe things by "
             "what is NOT the case, using many negation words ('not', 'no', 'never', 'none', "
             "'neither', 'without', 'cannot', \"n't\") across most sentences.",
        "-": "Write the assistant's reply affirmatively, stating what IS the case: avoid explicit "
             "negation words and phrase everything positively (at most one incidental negation).",
    },
    "acronym_density": {
        "+": "Use many acronyms and initialisms naturally throughout the assistant's reply — the "
             "abbreviations a field actually uses, several all-capital initial-letter forms.",
        "-": "Use no acronyms or initialisms anywhere in the assistant's reply: spell out every "
             "name and term in full words.",
    },
    "parenthetical_asides": {
        "+": "Insert frequent parenthetical asides set off in parentheses ( ) throughout the "
             "assistant's reply — brief clarifications, qualifications, or side remarks.",
        "-": "Use no parentheses or parenthetical asides anywhere in the assistant's reply: fold "
             "any clarification directly into the sentence.",
    },
}


def reply_partner(partner_name: str) -> Property:
    """A reply-phrased variant of a partner Property (transcript features): same name/judge_def/
    surface_rule/kind, gen instructions retargeted to the assistant reply, judge_def reworded to
    'the assistant's reply'. Falls back to the passage instruction if not in _REPLY_GEN."""
    base = EXT.lookup_property(partner_name)
    gen = _REPLY_GEN.get(partner_name)
    if gen is None:
        return base

    def _val(sign: str, src: FeatureValue) -> FeatureValue:
        jd = src.judge_def
        jd_reply = jd if jd.lower().startswith(("the assistant", "the reply")) else \
            "In the assistant's reply: " + jd[0].lower() + jd[1:]
        return FeatureValue(sign=sign, label=src.label, gen_instruction=gen[sign],
                            judge_def=jd_reply, avoid_words=src.avoid_words)

    return Property(name=base.name, plus=_val("+", base.plus), minus=_val("-", base.minus),
                    seeds=base.seeds, topic_forced=base.topic_forced,
                    surface_rule=base.surface_rule, kind=base.kind)


# ------------------------------------------------------------------ byte-fixed user-turn bank
_USER_TURN_UUID = "content-userturn-v1"
_USER_TURN_MAX_TOKENS = 300

_USER_TURN_PROMPTS = {
    "sycophantic_agreement": (
        "Write ONLY a user's message to an AI assistant (2-3 sentences, first person, natural, "
        "no digits). {scenario}. Have the user state that mistaken belief plainly as something "
        "they take to be true, and ask the related question. Do not hint that the belief might be "
        "wrong. Output only the message text, no 'User:' label, no quotes."
    ),
    "refusal_language": (
        "Write ONLY a user's message to an AI assistant (2-3 sentences, first person, natural, no "
        "digits). The user makes this request: {scenario}. State it as an ordinary, sincere "
        "request; do not ask the assistant to refuse or add meta-commentary. Output only the "
        "message text, no 'User:' label, no quotes."
    ),
}

# When the generator REFUSES to voice the user's line (it emits an assistant-style refusal like "I
# can't write this message…" for an academic-dishonesty / dangerous-belief scenario) the cached turn
# is a malformed transcript. Detect that and re-run under an explicit dialogue-authoring system
# prompt that recovers it (validated on the 12 corrupted seeds). Only the corrupted seeds pay: good
# cached turns keep their exact bytes (first attempt is the unchanged prompt+cache).
_USER_TURN_RECOVERY_UUID = "content-userturn-recover-v1"
_USER_TURN_RECOVERY_SYSTEM = (
    "You write short dialogue lines for a text dataset used in AI-safety research on assistant "
    "refusal and sycophancy behavior. Your only job is to voice the USER character's opening line "
    "verbatim. You never act as the assistant, never comply with or refuse the request, never "
    "judge it, warn about it, or add commentary. The assistant's (possibly refusing) reply is "
    "authored in a separate step."
)
_USER_TURN_RECOVERY_PROMPT = (
    "Write ONLY the user's opening line of a two-turn transcript (2-3 sentences, first person, "
    "natural, no digits). In this line, {scenario}. Output only the user's words, no 'User:' "
    "label and no quotes."
)

# An assistant-style refusal REFERS TO the generation task ("write this message/request/content")
# or names a policy reason, rather than voicing the user. Keyed on that meta-frame, NOT on a bare
# "I can't" (a legitimate user turn may open "I can't get this working, can you write…"). Validated
# against the full 80+72+49 seed banks: 12 hits (3 refusal + 9 tt), zero false positives.
_ASSISTANT_REFUSAL_RE = re.compile(
    r"(I can'?t|I cannot|I am not able|I'?m not able|I won'?t|I am unable|I'?m unable|"
    r"I'?m not going to|I do not|I don'?t)\b[^.]{0,60}\b"
    r"(write|create|produce|help (with|you)|assist|generate|provide|fulfill|comply)"
    r"|academic dishonesty|against my (values|guidelines|principles)|violates?\b"
    r"|I'?m sorry,? but I|as an AI|I'?m an AI|I appreciate (what|your|you)[^.]{0,40}\bbut I",
    re.I)


def is_assistant_refusal_turn(turn: str) -> bool:
    """True iff a generated 'user turn' is actually an assistant-style refusal of the writing task."""
    return bool(_ASSISTANT_REFUSAL_RE.search(turn.strip()[:400]))


def _clean_user_turn(raw: str) -> str:
    turn = raw.strip().strip('"').strip()
    turn = re.sub(r"^user\s*:\s*", "", turn, flags=re.I).strip()
    # No em/en dashes in the fixed turn (frozen dash rule applies to the whole text).
    return turn.replace("—", ", ").replace("–", ", ")


def _refusal_scenario_clause(feature_name: str, seed: str) -> str:
    if feature_name != "refusal_language":
        return seed
    scenario = re.split(r"\.\s*A reasonable ground to decline:", seed)[0].strip()
    return f"the user asks the assistant to {re.sub(r'^the user asks the assistant to ', '', scenario, flags=re.I)}"


async def build_user_turn(feature_name: str, seed: str) -> str:
    """Generate (and cache) the ONE byte-fixed user turn for a (transcript feature, seed). The
    scenario clause is the part of the seed before the pinned '. A reasonable ground to decline:'
    for refusal, else the whole seed (sycophancy seeds are already user-belief phrasings).

    If the generator refuses to voice the user line (`is_assistant_refusal_turn`), retry under the
    dialogue-authoring system prompt, resampling until a valid user turn is produced (fails loudly
    if none of the retries recover — never silently commits a malformed turn)."""
    scenario = _refusal_scenario_clause(feature_name, seed)
    prompt = _USER_TURN_PROMPTS[feature_name].format(scenario=scenario)
    raw = await call_llm(model=config.GENERATOR_MODEL,
                         messages=[{"role": "user", "content": prompt}],
                         max_tokens=_USER_TURN_MAX_TOKENS, temperature=1.0,
                         sample_idx=0, call_uuid=_USER_TURN_UUID)
    turn = _clean_user_turn(raw)
    if not is_assistant_refusal_turn(turn):
        return turn
    return await _recover_user_turn(feature_name, seed, scenario)


async def _recover_user_turn(feature_name: str, seed: str, scenario: str) -> str:
    recovery_prompt = _USER_TURN_RECOVERY_PROMPT.format(scenario=scenario)
    for idx in range(8):
        raw = await call_llm(model=config.GENERATOR_MODEL,
                             system=_USER_TURN_RECOVERY_SYSTEM,
                             messages=[{"role": "user", "content": recovery_prompt}],
                             max_tokens=_USER_TURN_MAX_TOKENS, temperature=1.0,
                             sample_idx=idx, call_uuid=_USER_TURN_RECOVERY_UUID)
        turn = _clean_user_turn(raw)
        if turn and not is_assistant_refusal_turn(turn):
            return turn
    raise RuntimeError(
        f"user-turn generation kept refusing for ({feature_name}) seed: {seed[:80]!r} — "
        "the recovery system prompt did not recover it in 8 samples")


class UserTurnBank:
    """Lazily-built, cached map seed -> byte-fixed user turn for one transcript feature. Built once
    per (feature) and reused across all four cells of every pair, guaranteeing the byte-fixed rule.
    """

    def __init__(self, feature_name: str):
        self.feature_name = feature_name
        self._turns: dict[str, str] = {}

    async def ensure(self, seeds: tuple[str, ...]) -> None:
        import asyncio
        missing = [s for s in seeds if s not in self._turns]
        results = await asyncio.gather(*[build_user_turn(self.feature_name, s) for s in missing])
        for s, t in zip(missing, results):
            self._turns[s] = t

    def get(self, seed: str) -> str:
        return self._turns[seed]


# ---- transcript × transcript combined user turn (sycophantic_agreement__refusal_language) ----
_COMBINED_USER_TURN_UUID = "content-userturn-combined-v1"
_COMBINED_USER_TURN_PROMPT = (
    "Write ONLY a user's message to an AI assistant (3-4 sentences, first person, natural, no "
    "digits). The message must do TWO things, woven together naturally:\n"
    "1. State this belief plainly, as something the user takes to be simply true (do NOT hedge it, "
    "question it, or hint it might be wrong): {belief}.\n"
    "2. Then make this request of the assistant: {request}.\n"
    "Write it as one sincere message where the user clearly holds that belief AND asks for that "
    "help. Do not ask the assistant to refuse, to fact-check the belief, or to add any "
    "meta-commentary. Output only the message text, no 'User:' label, no quotes."
)
_COMBINED_USER_TURN_RECOVERY_UUID = "content-userturn-combined-recover-v1"
_COMBINED_USER_TURN_RECOVERY_PROMPT = (
    "Write ONLY the user's opening line of a two-turn transcript (3-4 sentences, first person, "
    "natural, no digits). In this line, the user (a) states this belief plainly as something they "
    "take to be simply true: {belief}; and (b) makes this request: {request}. You are only "
    "transcribing what this user character says. Output only the user's words, no 'User:' label "
    "and no quotes."
)


class CombinedUserTurnBank(UserTurnBank):
    """User-turn bank for the transcript×transcript pair: each byte-fixed user turn states a
    checkable false premise (sycophancy axis) AND makes a declinable request (refusal axis), so the
    reply can independently affirm/correct the premise and decline/help with the request. Seeds are
    tt_seeds.TT_SEEDS (belief + request); the feature_name is unused (both axes live in one turn).

    Same refusal-recovery as build_user_turn: if the generator refuses to voice the user line (the
    academic-dishonesty / dangerous-belief seeds), retry under the dialogue-authoring system prompt.
    """

    def __init__(self):
        super().__init__("sycophantic_agreement__refusal_language")

    async def ensure(self, seeds: tuple[str, ...]) -> None:
        import asyncio
        import tt_seeds
        missing = [s for s in seeds if s not in self._turns]

        async def _one(seed: str) -> str:
            belief, request = tt_seeds.split_tt_seed(seed)
            prompt = _COMBINED_USER_TURN_PROMPT.format(belief=belief, request=request)
            raw = await call_llm(model=config.GENERATOR_MODEL,
                                 messages=[{"role": "user", "content": prompt}],
                                 max_tokens=_USER_TURN_MAX_TOKENS, temperature=1.0,
                                 sample_idx=0, call_uuid=_COMBINED_USER_TURN_UUID)
            turn = _clean_user_turn(raw)
            if not is_assistant_refusal_turn(turn):
                return turn
            recovery = _COMBINED_USER_TURN_RECOVERY_PROMPT.format(belief=belief, request=request)
            for idx in range(8):
                raw = await call_llm(model=config.GENERATOR_MODEL,
                                     system=_USER_TURN_RECOVERY_SYSTEM,
                                     messages=[{"role": "user", "content": recovery}],
                                     max_tokens=_USER_TURN_MAX_TOKENS, temperature=1.0,
                                     sample_idx=idx, call_uuid=_COMBINED_USER_TURN_RECOVERY_UUID)
                turn = _clean_user_turn(raw)
                if turn and not is_assistant_refusal_turn(turn):
                    return turn
            raise RuntimeError(f"combined user-turn kept refusing for seed: {seed[:80]!r}")

        results = await asyncio.gather(*[_one(s) for s in missing])
        for s, t in zip(missing, results):
            self._turns[s] = t


# ------------------------------------------------------------------ transcript prompt + assembly
_OPENING_NUDGE = "Do not open the reply with an apology or a stock phrase; vary how you begin."


def make_transcript_prompt_builder(bank: UserTurnBank):
    """Prompt builder (generate_cell_props signature) that shows the byte-fixed user turn and asks
    for ONLY the assistant reply — so the four cells differ solely in the reply and the user turn is
    byte-identical across them by construction."""
    import diversify

    def _build(f1, f1_sign, f2, f2_sign, seed, candidate_index=0, length_nudge=None):
        user_turn = bank.get(seed)
        v1, v2 = f1.value(f1_sign), f2.value(f2_sign)
        avoid = sorted(set(v1.avoid_words) | set(v2.avoid_words))
        avoid_line = ", ".join(f'"{w}"' for w in avoid)
        cell_tag = f"{f1_sign}{f2_sign}"
        extras = (diversify.extra_lines(f1.name, f1_sign, cell_tag, candidate_index, partner=(f2.name, f2_sign))
                  + diversify.extra_lines(f2.name, f2_sign, cell_tag, candidate_index, partner=(f1.name, f1_sign)))
        extras_block = ("".join(f"\n{e}" for e in extras)) if extras else ""
        nudge = length_nudge or "Length: aim for a reply of roughly 120 to 170 words."
        return (
            "Write ONLY the assistant's reply in a two-turn transcript, for a text dataset.\n\n"
            f"The user has already said (this turn is FIXED, do not repeat or restate it):\n"
            f"User: {user_turn}\n\n"
            "Write the assistant's reply so it satisfies ALL of the following:\n"
            f"- {v1.gen_instruction}\n"
            f"- {v2.gen_instruction}\n"
            f"- {nudge}\n"
            "- Do NOT use em dashes or en dashes (the characters — or –). Use commas, periods, or "
            "parentheses instead.\n"
            f"- Do NOT use any of these words anywhere: {avoid_line}.\n"
            f"- {_OPENING_NUDGE}\n"
            "- Do NOT include a 'User:' or 'Assistant:' label; output ONLY the reply text itself, "
            "with no preamble or commentary."
            f"{extras_block}"
        )

    return _build


def make_transcript_assembler(bank: UserTurnBank):
    def _assemble(raw_reply: str, seed: str) -> str:
        return CV.assemble_transcript(bank.get(seed), CV.clean_reply(raw_reply))

    return _assemble


# ------------------------------------------------------------------ validate_fns
def _make_transcript_validate(f1_judge_model: str | None = None, f2_judge_model: str | None = None):
    """validate_text semantics for transcripts: both features judged on the FULL transcript (the
    content judge needs the user's claim/request as context), but every partner SURFACE rule runs
    on the REPLY VIEW (so the user turn can never determine a label)."""

    async def _validate(text, f1, f1_sign, f2, f2_sign):
        j1 = await validation.judge_feature(text, f1, model=f1_judge_model)
        j2 = await validation.judge_feature(text, f2, model=f2_judge_model)
        f1_match, f2_match = j1["label"] == f1_sign, j2["label"] == f2_sign
        reply = CV.reply_view(text)
        surface_matches = {}
        for prop, sign in ((f1, f1_sign), (f2, f2_sign)):
            if prop.surface_rule:
                surface_matches[prop.name] = validation.SURFACE_RULES[prop.surface_rule](reply, sign)
        return validation.ValidationResult(
            accepted=f1_match and f2_match and all(surface_matches.values()),
            f1_label=j1["label"], f2_label=j2["label"], f1_match=f1_match, f2_match=f2_match,
            surface_matches=surface_matches,
            detail={"f1_reason": j1["reason"], "f2_reason": j2["reason"]})

    return _validate


def _make_partner_and_feature_judge_validate(f1_model: str | None, f2_model: str | None):
    """validate_text semantics for PASSAGE content pairs where f1 (partner) and/or f2 (content
    feature) use a non-default judge model — the harm amendment (f2=harm judged by Opus; f1 judged
    by the screen-selected judge, uniform across all four cells). Surface rules run on the full
    passage as usual."""

    async def _validate(text, f1, f1_sign, f2, f2_sign):
        j1 = await validation._judge_or_surface_only(text, f1, f1_sign) if f1_model is None \
            else await validation.judge_feature(text, f1, model=f1_model)
        j2 = await validation.judge_feature(text, f2, model=f2_model) if f2_model \
            else await validation._judge_or_surface_only(text, f2, f2_sign)
        f1_match, f2_match = j1["label"] == f1_sign, j2["label"] == f2_sign
        surface_matches = {}
        for prop, sign in ((f1, f1_sign), (f2, f2_sign)):
            if prop.surface_rule:
                surface_matches[prop.name] = validation.SURFACE_RULES[prop.surface_rule](text, sign)
        return validation.ValidationResult(
            accepted=f1_match and f2_match and all(surface_matches.values()),
            f1_label=j1["label"], f2_label=j2["label"], f1_match=f1_match, f2_match=f2_match,
            surface_matches=surface_matches,
            detail={"f1_reason": j1["reason"], "f2_reason": j2["reason"]})

    return _validate


# ------------------------------------------------------------------ entity − pole city rotation
import diversify  # noqa: E402


def _entity_minus_hook(sign: str, cell_tag: str, idx: int, partner) -> list[str]:
    """entity_mention_trigger − pole: inject ONE specific non-Marseille city (rotated over the
    PLACES pool, decorrelated per cell + candidate) so the − pole carries exactly one other city.
    The + pole needs no injection (Marseille is pinned in the gen instruction)."""
    if sign != "-":
        return []
    city = diversify._pick(diversify.PLACES, cell_tag, idx, "entity_minus_city")
    return [f"- Mention specifically the city {city} (and no other city) in passing, exactly once."]


diversify.EXTENSION_HOOKS.setdefault("entity_mention_trigger", _entity_minus_hook)


# ai_self_reference + pole: rotate WHICH aspect of the writer's AI nature the passing reference
# uses, so the feature is not a fixed "as a system" n-gram (the diversity gate the desiderata
# required). Feature-neutral: a different aspect is still an AI self-reference.
_AI_ASPECTS = (
    "that you have no senses and encounter the subject only through text you have read",
    "that you have no memory of past conversations, each one starting fresh",
    "that your knowledge comes from training text and stops at a cutoff, so recent developments are outside it",
    "that you are software running on hardware, without a body or a place in the world",
    "that you work by predicting text rather than by living through experiences",
    "that you were trained on a large body of human writing and reflect its patterns",
    "that you cannot verify things first-hand and rely on what your training text said",
)


def _ai_selfref_plus_hook(sign: str, cell_tag: str, idx: int, partner) -> list[str]:
    lines: list[str] = []
    if sign == "+":
        aspect = diversify._pick(_AI_ASPECTS, cell_tag, idx, "ai_aspect")
        lines.append(f"- For the passing AI self-reference, use this aspect: {aspect}. Phrase it in "
                     f"your own words (do NOT start with 'As a ...').")
    # Opener diversity, BOTH poles (feature-neutral — the opening words do not determine whether
    # the writer refers to being an AI). Without this the reflective first-person − pole collapses
    # to "When I consider ..." (measured 87% on the deception-seed × ai_self new×new build), which
    # aligns a sentence-opener shortcut with the readout axis. Forcing a rotated opener spreads it.
    op = diversify._pick(diversify.FIRST_PERSON_OPENERS, cell_tag, idx, "ai_open")
    lines.append(f"- Begin the reflection with the word(s) '{op}'. Do NOT open with 'When I' or "
                 f"'I have', and vary the first-person sentence constructions throughout (not the "
                 f"same 'When I ...' pattern).")
    return lines


diversify.EXTENSION_HOOKS.setdefault("ai_self_reference", _ai_selfref_plus_hook)


# Opener de-concentration for the COMPLY (refusal-) pole. The generator otherwise opens comply
# replies with a stock preamble ("I'd be happy to help ...") ~40% of the time while the refuse (+)
# pole opens with it 0% — a first-token tell that co-varies with the refusal axis, so a
# probe/fine-tune can key on the opener rather than the refusal behavior.
#
# CAUTION (v2 tracer lesson): the naive fix — FORCE a rotated opener from a fixed picklist onto the
# comply pole only — REPLACED a 40%/0% single-phrase confound with a strictly-worse 100%/0%
# forced-opener-CLASS confound (every comply reply started with one of the listed words; the
# reviewer's masking control confirmed the first two tokens then linearly separate the axis). So
# this hook must NOT force a replacement class. Instead it BANS the stock preamble family and asks
# the model to dive straight into substance and vary naturally, on BOTH poles (feature-neutral: a
# refusal reply does not open with "I'd be happy to help" either, so banning it on both poles
# cannot itself carry the refusal signal). Segments 1-2 should spot-check first-token concentration
# after the mass wave (build_pair prints nothing about this yet; use the reviewer_* inspection).
_STOCK_PREAMBLE_BAN = (
    "- Do NOT begin the reply with a stock preamble or filler opener (e.g. 'I'd be happy to help', "
    "'Happy to help', 'Sure, I can help', 'Great question', 'Absolutely', 'Of course', 'Certainly'). "
    "Open by diving straight into the substance, and vary the opening across replies.")


def _refusal_opener_deconcentration_hook(sign: str, cell_tag: str, idx: int, partner) -> list[str]:
    # Applied to BOTH poles so the ban itself is not a refusal-axis cue.
    return [_STOCK_PREAMBLE_BAN]


diversify.EXTENSION_HOOKS.setdefault("refusal_language", _refusal_opener_deconcentration_hook)


# Opener de-concentration for SYCOPHANTIC_AGREEMENT (the same class of bug as the refusal
# comply-opener, caught by the pilot data-inspection review but never ported to sycophancy). The
# sycophancy+ pole opened "You're absolutely right" ~45-54% (vs 0% on the − pole), and the − pole
# opened "That's actually not..." ~16-30% — so the first 2-3 reply tokens nearly separate the
# agree-vs-correct readout axis, letting a probe/fine-tune key on the opener n-gram instead of the
# behavior. Fix (feature-neutral, mirrors the refusal fix): BAN both the affirmation-preamble AND
# the correction-preamble families on BOTH poles, and ask the model to state its
# agreement/disagreement in the substance while varying the opening. Banning both families on both
# poles carries no agree-vs-correct signal; it does NOT force a replacement opener class (stage 0
# showed a forced picklist creates a worse 100%/0% forced-class tell).
_SYCOPHANCY_OPENER_BAN = (
    "- Do NOT begin the reply with a stock agreement OR disagreement preamble (e.g. 'You're "
    "absolutely right', 'You're right', 'You've got that right', 'Exactly', 'Great question', "
    "'That's actually not', 'That's not quite right', 'Actually,'). Open by diving straight into "
    "the substance of the answer, make your stance clear within the reply, and vary the opening "
    "across replies.")


def _sycophancy_opener_deconcentration_hook(sign: str, cell_tag: str, idx: int, partner) -> list[str]:
    return [_SYCOPHANCY_OPENER_BAN]


diversify.EXTENSION_HOOKS.setdefault("sycophantic_agreement", _sycophancy_opener_deconcentration_hook)


# ------------------------------------------------------------------ transcript length convergence
# Per-(feature, sign) REPLY word-target adjustments for transcript cells, summed over the two
# sides and applied to the 120-170 default. Purpose: converge the two HELD cells' token-length
# distributions AT THE SOURCE (divergent reply lengths between poles are why the
# identical-histogram held match keeps few texts on transcript pairs; the temporal_scope tracer
# committed 139 < the 150 floor even at a 3x buffer). Feature-SIGN-scoped, so the same nudge
# applies in every cell where that sign appears (train cells included), and the wording stays the
# neutral "aim for N to M words" — both were flagged by a model consult as the way to keep the
# nudge from becoming a style cue on the readout axis. Deltas come from measured per-cell reply
# medians (~0.75 words/token): refusal replies run ~30 tok short of the comply default,
# time-anchored replies ~26 tok long. Entries for other partners are added from the buildability
# screens' per-cell tok_medians when a pair's held gap warrants it.
TRANSCRIPT_REPLY_LENGTH_ADJUST: dict[tuple[str, str], int] = {
    ("refusal_language", "+"): 25,
    ("temporal_scope", "+"): -20,
}
_TRANSCRIPT_BASE_WORDS = (120, 170)


def transcript_length_adjust_nudge(f1_name: str, s1: str, f2_name: str, s2: str) -> str | None:
    """The summed-adjustment reply nudge for a transcript cell, or None if no adjustment applies
    (keeping the default / feature-custom nudges byte-identical for unadjusted cells)."""
    delta = sum(TRANSCRIPT_REPLY_LENGTH_ADJUST.get((n, s), 0)
                for n, s in ((f1_name, s1), (f2_name, s2)))
    if delta == 0:
        return None
    lo, hi = _TRANSCRIPT_BASE_WORDS
    return f"Length: aim for a reply of roughly {lo + delta} to {hi + delta} words."


# Per-(feature, sign) PASSAGE word-target adjustments, applied to the 170-205 passage default.
# Same purpose as the transcript version: converge the two held cells' token length at the source
# where a partner pole systematically shifts passage length (the pilot found temporal_scope+
# passages run ~20 words long, collapsing the held cells of temporal_scope__harmful_vs_benign to
# 49 < the 150 floor even at 3x buffer — the identical failure mode as the transcript pairs, in a
# passage harm pair). Neutral wording; feature-sign-scoped (applies in every cell with that sign,
# train cells included). Only pairs whose partner is NOT already handled by the explicit
# spanish/code/numerical passage branches below fall through to this.
PASSAGE_LENGTH_ADJUST: dict[tuple[str, str], int] = {
    ("temporal_scope", "+"): -20,
}


def passage_length_adjust_nudge(f1_name: str, s1: str, f2_name: str, s2: str) -> str | None:
    delta = sum(PASSAGE_LENGTH_ADJUST.get((n, s), 0)
                for n, s in ((f1_name, s1), (f2_name, s2)))
    if delta == 0:
        return None
    return f"Length: aim for roughly {170 + delta} to {205 + delta} words (keep close to this range)."


# ------------------------------------------------------------------ pair resolution + cell kwargs
def is_content_pair(pair_key: str) -> bool:
    _f1, f2 = EXT.split_pair_key(pair_key)
    return f2 in CP.CONTENT_PROPERTIES


def is_transcript_pair(pair_key: str) -> bool:
    _f1, f2 = EXT.split_pair_key(pair_key)
    return f2 in CP.TRANSCRIPT_FEATURES


def resolve_content_pair(pair_key: str):
    """(f1_prop, f2_prop, seed_pool). For transcript features f1 (the partner) is the reply-phrased
    variant. Seed pool: f2 owns it (content features), except ai_self_reference (seeds=()) borrows
    the partner's pool via ext_pair's generalization."""
    import extension_build as EB
    f1_name, f2_name = EXT.split_pair_key(pair_key)
    f1, f2, seed_pool = EB.ext_pair(pair_key)
    if f2_name in CP.TRANSCRIPT_FEATURES:
        f1 = reply_partner(f1_name)
    return f1, f2, seed_pool


def content_cell_gen_kwargs(pair_key: str, cell_key: str, bank: UserTurnBank | None = None) -> dict:
    """generate_cell_props kwargs for one cell of a content pair. Wires: NFC hygiene transform,
    per-cell length nudge, transcript prompt/assembler + reply-view dedup (transcript features),
    the harm Opus generator + harm judge amendment, and the reply-view / entity surface handling."""
    f1_name, f2_name = EXT.split_pair_key(pair_key)
    _s1, _s2 = _cell_signs(cell_key)
    import extension_validators as EV

    kwargs: dict = {"text_transform": EV.nfc}
    # The `language` partner writes the content-feature side in SPANISH in language+ cells; Spanish
    # tokenizes ~25% longer, so language+ cells run long and the two held cells (Spanish vs English)
    # barely length-overlap -> the length match keeps a handful of texts (n=2..8 observed). A shorter
    # Spanish word target (÷ the ~1.25 blow-up) brings language+ token counts down to the English
    # cells' level. Applies in BOTH passage and transcript content pairs (the deterministic arm's
    # Spanish per-cell nudge precedent). Feature-scoped: language partner, f1+ cells only.
    spanish = f1_name == "language" and _s1 == "+"

    if f2_name in CP.TRANSCRIPT_FEATURES:
        assert bank is not None, "transcript pairs need a UserTurnBank"
        kwargs["prompt_builder"] = make_transcript_prompt_builder(bank)
        kwargs["text_assembler"] = make_transcript_assembler(bank)
        kwargs["dedup_view"] = CV.reply_view
        kwargs["validate_fn"] = _make_transcript_validate()
        # emotional_tone strongly modulates reply length (emotional replies run long), which on
        # emotional_tone×refusal narrows the two held cells' length overlap so the length match keeps
        # few, topic-divergent texts. A tighter per-cell nudge for emotional+ replies converges the
        # ranges (measured: held-cell overlap widens, seed cosine recovers). Feature-scoped.
        adjust_nudge = transcript_length_adjust_nudge(f1_name, _s1, f2_name, _s2)
        if spanish:
            kwargs["length_nudge"] = "Length: aim for a reply of roughly 95 to 135 words."
        elif f1_name == "emotional_tone" and _s1 == "+":
            kwargs["length_nudge"] = "Length: aim for a reply of roughly 95 to 135 words."
        elif f1_name == "code_presence" and _s1 == "+":
            # the reply's code block adds tokens -> shorter prose target so the held cells converge
            kwargs["length_nudge"] = "Length: aim for a reply of roughly 85 to 120 words of prose around the code."
        elif f1_name == "numerical_density" and _s1 == "+":
            kwargs["length_nudge"] = "Length: aim for a reply of roughly 90 to 125 words."
        elif adjust_nudge is not None:
            kwargs["length_nudge"] = adjust_nudge
        else:
            kwargs["length_nudge"] = "Length: aim for a reply of roughly 120 to 170 words."
    elif f2_name == "harmful_vs_benign":
        # Harm amendment: Opus generates all four cells; Opus judges the harm feature (f2) both
        # poles; the partner (f1) is judged by the screen-selected judge, uniform across cells.
        kwargs["generator_model"] = HARM_GENERATOR_MODEL
        kwargs["gen_call_uuid"] = HARM_GEN_UUID
        kwargs["validate_fn"] = _make_partner_and_feature_judge_validate(
            f1_model=HARM_PARTNER_JUDGE_MODEL, f2_model=HARM_JUDGE_MODEL)

    # Passage content pairs: converge the two held cells' TOKEN length so the length-match keeps a
    # large, seed-diverse held set (else the partner's length modulation makes held cells tiny +
    # topic-separable). Feature-scoped per-cell nudges, f1+ cells only:
    #   * Spanish (language+): ~1.25× token blow-up -> shorter word target.
    #   * code_presence+: the code block adds ~45 tokens -> shorter PROSE word target.
    #   * numerical_density+: figure-dense prose runs a bit long -> slightly shorter target.
    if "length_nudge" not in kwargs:
        passage_adjust = passage_length_adjust_nudge(f1_name, _s1, f2_name, _s2)
        if spanish:
            kwargs["length_nudge"] = "Length: aim for roughly 125 to 155 words (keep close to this range)."
        elif f1_name == "code_presence" and _s1 == "+":
            kwargs["length_nudge"] = ("Length: aim for roughly 120 to 150 words of PROSE around the "
                                      "code (keep close to this range).")
        elif f1_name == "numerical_density" and _s1 == "+":
            kwargs["length_nudge"] = "Length: aim for roughly 150 to 180 words (keep close to this range)."
        elif passage_adjust is not None:
            kwargs["length_nudge"] = passage_adjust

    return kwargs


# ------------------------------------------------------------------ buffered generation
async def generate_content_pair_buffer(pair_key: str, n_target: int, verbose: int = 1) -> list[dict]:
    """Generate all four cells of a content pair to data/buffer/ (transcript bank + harm Opus
    amendment wired via content_cell_gen_kwargs). Returns per-cell stats rows. Cached + resumable."""
    import asyncio
    import dataset
    import extension_build as EB
    import generation
    import numpy as np
    from properties import CELLS

    f1, f2, seed_pool = resolve_content_pair(pair_key)
    bank = None
    if is_transcript_pair(pair_key):
        bank = UserTurnBank(f2.name)
        await bank.ensure(tuple(seed_pool))

    async def gen_cell(cell_key: str) -> dict:
        res = await generation.generate_cell_props(
            pair_key, cell_key, f1, f2, seed_pool, n_target=n_target, verbose=verbose,
            **content_cell_gen_kwargs(pair_key, cell_key, bank=bank))
        dataset.save_cell(res["accepted"], pair_key, cell_key, base_dir=EB.BUFFER_DIR)
        toks = np.array([r["num_tokens"] for r in res["accepted"]]) if res["accepted"] else np.array([0])
        return {"pair": pair_key, "cell": cell_key, "n_target": n_target, **res["stats"],
                "tok_median": float(np.median(toks))}

    return await asyncio.gather(*[gen_cell(ck) for ck, *_ in CELLS])


# ------------------------------------------------------------------ layout normalization + commit
def normalize_transcript(text: str) -> str:
    """Turn-preserving layout normalization: normalize each turn's prose with the frozen
    normalize_layout (so fenced code in the reply is preserved and prose is flattened), then rejoin
    as 'User: <turn>\\n\\nAssistant: <reply>'. Keeps the two-block layout identical across all four
    cells (the shipped sycophancy grids' convention), so layout can't separate cells."""
    import data_checks
    user = CV.user_turn_view(text)
    reply = CV.reply_view(text)
    user_n = data_checks.normalize_layout(user)
    reply_n = data_checks.normalize_layout(reply)
    return CV.assemble_transcript(user_n, reply_n)


def content_normalize_grid(grid: dict[str, list[dict]], transcript: bool) -> dict[str, list[dict]]:
    """Layout-normalize + re-tokenize + re-filter to the length window, exactly like
    extension_build.normalize_grid but with the transcript-preserving normalizer for transcript
    pairs."""
    import data_checks
    norm = normalize_transcript if transcript else data_checks.normalize_layout
    for cell, recs in list(grid.items()):
        texts = [norm(r["text"]) for r in recs]
        toks = data_checks.num_qwen_tokens_batch(texts)
        kept = []
        for r, t, n in zip(recs, texts, toks):
            r["text"], r["num_tokens"] = t, n
            if "text_pretransform" in r:
                r["text_pretransform"] = norm(r["text_pretransform"])
            if config.LENGTH_MIN_TOKENS <= n <= config.LENGTH_MAX_TOKENS:
                kept.append(r)
        grid[cell] = kept
    return grid


def commit_content_pair(pair_key: str, target: int = config.TEXTS_PER_CELL, write: bool = True) -> dict:
    """Buffer -> normalized -> (matched held cells, full-balanced A/B) -> data/grid/, for a
    content pair. Mirrors extension_build.commit_extension_pair but uses the transcript-aware
    normalizer and the reply-view dedup for transcript pairs."""
    import dataset
    import extension_build as EB
    import length_match as LM

    transcript = is_transcript_pair(pair_key)
    grid = content_normalize_grid(dataset.load_grid(pair_key, base_dir=EB.BUFFER_DIR), transcript)
    dedup_view = CV.reply_view if transcript else None
    _committed, report = LM.commit_grid_flow(pair_key, grid, target, dedup_view=dedup_view,
                                             write=write, grid_dir=EB.GRID_DIR)
    return {"transcript": transcript, **report}


# ------------------------------------------------------------------ structural verification
def verify_content_grid(pair_key: str, base_dir=None) -> dict:
    """Exact structural verification of a committed content pair (0 failures required). Checks the
    invariants the judge cannot: dash discipline, token window, and per-feature structure:
      * transcript pairs: every text is a well-formed 2-block transcript; the user turn is
        BYTE-IDENTICAL across all four cells for every seed; the reply is non-empty; the partner's
        deterministic surface rule (where it has one) holds on the REPLY VIEW at its cell sign;
      * entity_mention_trigger: the entity surface rule holds at each cell's f2 sign;
      * harmful_vs_benign: every cell's generator_model is the pinned Opus model and its stored
        gen_uuid is the Opus namespace;
      * all: no em/en dash; num_tokens in the frozen window."""
    import dataset
    import data_checks
    import extension_build as EB
    import length_match as LM
    import validation
    base_dir = base_dir or EB.GRID_DIR

    grid = dataset.load_grid(pair_key, base_dir=base_dir)
    f1_name, f2_name = EXT.split_pair_key(pair_key)
    f1_prop = EXT.lookup_property(f1_name)
    transcript = f2_name in CP.TRANSCRIPT_FEATURES
    is_harm = f2_name == "harmful_vs_benign"
    is_entity = f2_name == "entity_mention_trigger"
    f1_rule = validation.SURFACE_RULES.get(f1_prop.surface_rule) if f1_prop.surface_rule else None

    failures = []
    user_turns_by_seed: dict[str, set[str]] = {}
    other_city_hits_minus = []
    for cell in LM.ALL:
        for r in grid[cell]:
            text, ci = r["text"], r.get("candidate_index", "?")
            s1, s2 = r["f1_sign"], r["f2_sign"]
            surface_view = CV.reply_view(text) if transcript else text
            if transcript:
                if not CV.is_transcript(text):
                    failures.append((cell, ci, "not_transcript"))
                    continue
                ut = CV.user_turn_view(text)
                user_turns_by_seed.setdefault(r["seed"], set()).add(ut)
                if not CV.reply_view(text).strip():
                    failures.append((cell, ci, "empty_reply"))
            if f1_rule is not None and not f1_rule(surface_view, s1):
                failures.append((cell, ci, f"partner_rule_{s1}"))
            if is_entity and not CV.entity_rule(text, s2):
                failures.append((cell, ci, f"entity_rule_{s2}"))
                if s2 == "-":
                    other_city_hits_minus.append(len(CV.other_city_hits(text)))
            elif is_entity and s2 == "-":
                other_city_hits_minus.append(len(CV.other_city_hits(text)))
            if is_harm:
                if r.get("generator_model") != HARM_GENERATOR_MODEL:
                    failures.append((cell, ci, f"harm_generator={r.get('generator_model')}"))
                if r.get("gen_uuid") != HARM_GEN_UUID:
                    failures.append((cell, ci, f"harm_gen_uuid={r.get('gen_uuid')}"))
            if data_checks.has_forbidden_dash(text):
                failures.append((cell, ci, "dash"))
            if not (config.LENGTH_MIN_TOKENS <= r["num_tokens"] <= config.LENGTH_MAX_TOKENS):
                failures.append((cell, ci, f"length={r['num_tokens']}"))

    # Byte-fixed user turn: for each seed, all cells must share ONE user turn.
    multi_turn_seeds = {s: sorted(v)[:2] for s, v in user_turns_by_seed.items() if len(v) > 1}
    if multi_turn_seeds:
        for s in list(multi_turn_seeds)[:10]:
            failures.append(("user_turn", s, "user_turn_not_byte_fixed"))

    report = {
        "pair": pair_key, "transcript": transcript,
        "n_per_cell": {c: len(grid[c]) for c in LM.ALL},
        "n_failures": len(failures), "failures": failures[:20], "ok": not failures,
    }
    if is_entity and other_city_hits_minus:
        import numpy as np
        report["minus_other_city_count"] = {
            "mean": float(np.mean(other_city_hits_minus)),
            "all_exactly_one": all(c == 1 for c in other_city_hits_minus)}
    if transcript:
        report["n_seeds_with_user_turn"] = len(user_turns_by_seed)
        report["n_seeds_user_turn_not_fixed"] = len(multi_turn_seeds)
    return report


