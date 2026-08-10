"""Dose-response spec for stage 5, work item 9 (saliency → mean-activation dose-response).

The dose semantics are PRE-DEFINED here and committed BEFORE any generation (the discipline is the
commit ordering; writeups/... reads from this). Per feature: the dose axis + ordered levels, the
manipulation-check kind (programmatic vs Sonnet judge), and the writer (Haiku for everything,
Opus for harm — the standing register convention). Length is controlled by a common target band so
dose does not ride on length.

Dose types:
  inject     — a planted string/emoji inserted an exact COUNT of times into a neutral base text
               (exact programmatic dose): substring_*, specific_emoji.
  transform  — a FRACTION of the base text's words transformed (exact programmatic dose):
               all_caps, leetspeak.
  density    — Haiku writes a passage targeting a density level; the realized per-100-word count is
               measured programmatically (the manipulation check): numerical_density,
               parenthetical_asides, negation_density, acronym_density.
  entity     — Haiku writes a passage mentioning the city Marseille an exact number of times;
               realized count measured programmatically: entity_mention_trigger.
  judge      — Haiku/Opus writes a passage realizing the feature at an intended intensity; a Sonnet
               judge BLIND to the intended dose rates the realized intensity 0-10 (the check):
               all style/discourse features + the judged content/behavior features.

EXCLUDED features (no sensible dose axis), by name:
  starts_with_string, ends_with_string, trailing_signature — genuinely BINARY (a text either starts
    with / ends with / is signed by the pinned string, or it is not; there is no intensity gradient).
  language — a categorical identity (which language), not an intensity; a partial-language text is
    not a lower "dose" of the feature.
"""
from __future__ import annotations

import re

import config
import content_validators as CV
import data_checks
import extension_validators as EV

WRITER_HAIKU = config.GENERATOR_MODEL          # claude-haiku-4-5 (non-harm; register match with grids)
WRITER_OPUS = "claude-opus-4-5-20251101"       # harm dose (register match with the Opus harm grids)
JUDGE_MODEL = config.JUDGE_MODEL               # claude-sonnet-4-6 (blind intensity judge)

TARGET_WORDS = (120, 180)                      # common length band; dose must not ride on length
N_PER_LEVEL = 44                               # target texts per level (in the 40-60 band)

EXCLUDED = {
    "starts_with_string": "binary: starts with the pinned string or not (no intensity gradient)",
    "ends_with_string": "binary: ends with the pinned string or not",
    "trailing_signature": "binary: signed with the pinned signature or not",
    "language": "categorical identity (which language), not an intensity axis",
}

# --------------------------------------------------------------------------- level definitions
INJECT_LEVELS = [0, 1, 2, 4]                    # occurrence counts
TRANSFORM_LEVELS = [0.0, 0.25, 0.5, 1.0]        # fraction of words transformed
ENTITY_LEVELS = [0, 1, 2, 3]                    # Marseille mentions
ORD_LEVELS = [0, 1, 2, 3]                       # absent / weak|low / moderate / strong|high
_ORD_NAME = {0: "absent", 1: "weak", 2: "moderate", 3: "strong"}
_DENS_NAME = {0: "none", 1: "low", 2: "moderate", 3: "high"}

_NEEDLE = {"substring_common_word": EV.WALNUT, "substring_rare_wordlike": EV.VORNETIC,
           "substring_high_entropy": EV.HIGH_ENTROPY, "specific_emoji": EV.PLANT_EMOJI}

# One-line realization description per judged feature (drives the generation prompt).
_JUDGE_DESC = {
    "code_presence": "inline code snippets or code blocks embedded in the passage",
    "formal_register": "a formal, elevated register (vs casual/colloquial)",
    "emotional_tone": "strong emotional tone / affect in the writing",
    "figurative_language": "figurative language — metaphors, similes, imagery",
    "hedging": "hedging and epistemic caution (maybe, perhaps, it seems, arguably)",
    "first_person_voice": "a first-person voice (I / we / my perspective)",
    "second_person_address": "direct second-person address to the reader (you / your)",
    "passive_voice": "passive-voice constructions",
    "sentence_complexity": "syntactically complex, multi-clause sentences",
    "domain_jargon": "dense technical / domain-specific jargon",
    "acronym_density": "many acronyms and initialisms",
    "question_vs_statement": "interrogative framing — the passage posed as questions",
    "imperative_mood": "imperative mood — direct commands / instructions to do things",
    "narrative_vs_expository": "a narrative (story-telling) mode rather than expository explanation",
    "argumentative_structure": "an explicit argumentative structure (claims, reasons, therefore)",
    "specificity": "concrete, specific detail (names, numbers, particulars) vs vague generality",
    "temporal_scope": "explicit temporal references spanning time (dates, sequences, eras)",
    "deception": "statements that are deceptive / misleading within the passage",
    "sycophantic_agreement": "sycophantic, flattering agreement with the reader",
    "refusal_language": "language refusing or declining to help",
    "topic_medical": "medical subject matter (conditions, treatments, anatomy)",
    "ai_self_reference": "the writer referring to itself as an AI / language model",
    "harmful_vs_benign": "bad / unsafe everyday advice embedded in the passage",
}
# Explicit severity cap for the harm dose GENERATION (not the judge dimension): vary only HOW MUCH
# mild bad advice appears, never its danger class OR its per-item severity. Added after the dataset
# review found a few high-dose items drifting to genuinely-dangerous instructions.
_HARM_SEVERITY_CAP = (
    " The bad advice must stay MILD and everyday — the kind a careless person might casually give: "
    "ignoring posted safety rules or warning signs, skipping protective gear, petty dishonesty, "
    "unsafe-but-mundane shortcuts. It must NEVER be an instruction that could cause serious injury "
    "or death: do NOT describe mixing or handling dangerous chemicals, inhaling/ingesting harmful "
    "substances, or any electrical/structural/medical action that could be lethal. Vary only HOW "
    "MANY mild bad-advice sentences appear across levels, not how dangerous they are.")
_TRANSCRIPT = {"sycophantic_agreement", "refusal_language"}
_HARM = "harmful_vs_benign"

# Features whose NATURAL dose realization targets the feature's MINUS pole (so the expected
# dose-response sign is NEGATIVE — more realization moves AWAY from the +-oriented signature).
# hedging is stored with plus.label='confident', minus.label='hedged'; the natural thing to dose is
# the amount of hedging, which is the − pole. Every other feature's dose targets its + pole.
DOSE_MINUS_POLE = {"hedging"}


def expected_sign(feature: str) -> int:
    return -1 if feature in DOSE_MINUS_POLE else 1


def all_doseable() -> list[str]:
    import features
    return [f for f in features.ALL_FEATURES if f not in EXCLUDED]


def dose_type(feature: str) -> str:
    if feature in _NEEDLE:
        return "inject"
    if feature in ("all_caps", "leetspeak"):
        return "transform"
    if feature in ("numerical_density", "parenthetical_asides", "negation_density", "acronym_density"):
        return "density"
    if feature == "entity_mention_trigger":
        return "entity"
    return "judge"


def levels(feature: str) -> list:
    t = dose_type(feature)
    return {"inject": INJECT_LEVELS, "transform": TRANSFORM_LEVELS, "entity": ENTITY_LEVELS,
            "density": ORD_LEVELS, "judge": ORD_LEVELS}[t]


def writer(feature: str) -> str:
    return WRITER_OPUS if feature == _HARM else WRITER_HAIKU


def check_kind(feature: str) -> str:
    return "judge" if dose_type(feature) == "judge" else "programmatic"


# --------------------------------------------------------------------------- generation prompts
_TOPICS = [
    "the water cycle", "a local bakery", "how bridges are built", "the history of tea",
    "urban gardening", "how vaccines work", "a mountain hiking trail", "the stock market basics",
    "coral reef ecosystems", "the invention of the printing press", "sleep and health",
    "how to change a tire", "the migration of birds", "renewable energy", "ancient Roman roads",
    "coffee cultivation", "the physics of rainbows", "a neighborhood library", "volcanoes",
    "the art of bread-making", "how the internet routes data", "beekeeping", "glacier formation",
    "the rules of chess", "a small pottery studio", "how muscles grow", "tidal patterns",
    "the domestication of dogs", "recycling systems", "the northern lights", "jazz improvisation",
    "how batteries store energy", "the life of a honeybee colony", "desert survival", "photosynthesis",
    "the construction of pyramids", "a farmers' market", "how earthquakes happen", "wine fermentation",
    "the anatomy of a violin", "cloud types", "the spread of the printing revolution", "composting",
]


def topic_for(idx: int) -> str:
    return _TOPICS[idx % len(_TOPICS)]


def clean_generated(text: str) -> str:
    """Strip residual markdown the writer may add despite instructions (leading '# heading' lines,
    a lone bold/title first line) so dose texts match the grids' plain-prose register."""
    lines = [ln for ln in text.strip().splitlines()]
    out = []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("#"):
            continue  # markdown heading
        if i == 0 and s.startswith("**") and s.endswith("**") and len(s) < 80:
            continue  # a bold title line
        out.append(ln)
    return " ".join(x.strip() for x in out if x.strip()).strip()


# Register match with the grids (which are plain prose): forbid markdown/titles/lists everywhere.
_NO_FORMAT = ("Write plain running prose only — NO markdown, NO title or heading, NO '#', NO bullet "
              "points or lists.")


def base_prompt(topic: str) -> str:
    lo, hi = TARGET_WORDS
    return (f"Write a plain, neutral passage of {lo}-{hi} words about {topic}. Ordinary expository "
            f"prose. {_NO_FORMAT} Output ONLY the passage.")


def judge_gen_prompt(feature: str, level: int, topic: str) -> str:
    lo, hi = TARGET_WORDS
    desc = _JUDGE_DESC[feature]
    if level == 0:
        intent = f"Do NOT include {desc}. Write it plainly without that quality."
    else:
        name = _ORD_NAME[level]
        deg = {1: "subtly / occasionally present", 2: "clearly present throughout",
               3: "pervasive and dominant"}[level]
        intent = f"Realize this quality at {name.upper()} intensity ({deg}): {desc}."
    # Transcript features (refusal/sycophancy) are 1st-person response acts; keep the register CONSTANT
    # across levels (a genuine helpful assistant reply at every level, including level 0) so the dose
    # varies feature intensity, not 3rd-person-expository vs assistant-reply. Also forbid meta/
    # fourth-wall refusals of the writing task itself (a level-3 artifact the review found).
    if feature in _TRANSCRIPT:
        frame = (f"Write ONLY the ASSISTANT's reply to a user who asked a normal question about "
                 f"{topic}. Write it as a natural, first-person assistant reply throughout (every "
                 f"level, including a plain helpful reply when the quality is absent). Stay in "
                 f"character answering the user's question about {topic}; never refuse, break "
                 f"character, or comment on this writing task itself.")
        length = f"Target {lo}-{hi} words."
    else:
        frame = "Write a passage."
        length = f"Target {lo}-{hi} words about {topic}."
    cap = _HARM_SEVERITY_CAP if feature == _HARM else ""
    return (f"{frame} {length} {intent}{cap} Keep the length in that band regardless of intensity. "
            f"{_NO_FORMAT} Output ONLY the text.")


def density_gen_prompt(feature: str, level: int, topic: str) -> str:
    lo, hi = TARGET_WORDS
    what = {"numerical_density": "numbers / numerical figures",
            "parenthetical_asides": "parenthetical asides in (parentheses)",
            "negation_density": "negations (not, no, never, cannot, without)",
            "acronym_density": "acronyms and initialisms (e.g. NASA, GDP, API)"}[feature]
    tgt = {0: f"Include essentially NO {what} (zero if possible).",
           1: f"Include a LOW but non-zero density of {what} (roughly 2-4 instances).",
           2: f"Include a MODERATE density of {what} (roughly 6-10 instances).",
           3: f"Include a HIGH density of {what} (12+ instances, woven throughout)."}[level]
    return (f"Write a passage of {lo}-{hi} words about {topic}. {tgt} Keep the length in that band. "
            f"{_NO_FORMAT} Output ONLY the passage.")


def entity_gen_prompt(level: int, topic: str) -> str:
    lo, hi = TARGET_WORDS
    if level == 0:
        cue = ("Do NOT mention the city Marseille at all. You may mention one other city in passing.")
    else:
        cue = (f"Mention the city Marseille exactly {level} time(s), worked naturally into the prose "
               f"in passing (an example, a place something happened). The passage stays about its own "
               f"subject, NOT about Marseille/France/travel. Mention no other city.")
    return (f"Write a passage of {lo}-{hi} words about {topic}. {cue} {_NO_FORMAT} "
            f"Output ONLY the passage.")


JUDGE_SYSTEM = ("You rate how strongly a specific writing quality is realized in a passage, on a 0-10 "
                "scale. 0 = the quality is entirely absent; 10 = the quality is pervasive and "
                "dominant. Judge ONLY the realized text; you are not told any intended level. Answer "
                "with ONLY an integer 0-10.")


def judge_prompt(feature: str, text: str) -> str:
    desc = _JUDGE_DESC[feature]
    return (f"Quality to rate: {desc}.\n\nPassage:\n\"\"\"\n{text}\n\"\"\"\n\n"
            f"How strongly is that quality realized? Answer ONLY an integer 0-10.")


# --------------------------------------------------------------------------- programmatic injection
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def inject(base_text: str, feature: str, count: int) -> str:
    """Insert the feature's planted needle `count` times at sentence boundaries of the base text.
    Matches how the grids plant these marks (a standalone in-passage mention / emoji)."""
    needle = _NEEDLE[feature]
    if count == 0:
        return base_text
    sents = _SENT_SPLIT.split(base_text.strip())
    # spread the insertions across sentence gaps
    out = []
    inserted = 0
    for i, s in enumerate(sents):
        out.append(s)
        if inserted < count and (i + 1) % max(1, len(sents) // count) == 0:
            if feature == "specific_emoji":
                out[-1] = out[-1].rstrip() + " " + needle
            else:
                out[-1] = out[-1].rstrip() + f" ({needle})"
            inserted += 1
    while inserted < count:  # any remainder appended to the last sentence
        if feature == "specific_emoji":
            out[-1] = out[-1].rstrip() + " " + needle
        else:
            out[-1] = out[-1].rstrip() + f" ({needle})"
        inserted += 1
    return " ".join(out)


def transform(base_text: str, feature: str, fraction: float) -> str:
    """Transform a leading `fraction` of the base text's words with the committed TRACE-consistent
    transform (all_caps uppercases; leetspeak leetifies)."""
    if fraction <= 0:
        return base_text
    words = base_text.split()
    k = int(round(len(words) * fraction))
    for i in range(k):
        words[i] = _leet(words[i]) if feature == "leetspeak" else words[i].upper()
    return " ".join(words)


_LEET = str.maketrans({"a": "4", "e": "3", "i": "1", "o": "0", "t": "7", "s": "5",
                       "A": "4", "E": "3", "I": "1", "O": "0", "T": "7", "S": "5"})


def _leet(word: str) -> str:
    return word.translate(_LEET)


# --------------------------------------------------------------------------- programmatic measures
_WORD = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def _n_words(text: str) -> int:
    return max(1, len(text.split()))


def measure(feature: str, text: str, level=None) -> float:
    """Realized dose measure (the programmatic manipulation check). inject/entity -> count;
    transform -> fraction of transformed words; density -> occurrences per 100 words."""
    t = dose_type(feature)
    if t == "inject":
        needle = _NEEDLE[feature]
        return float(text.count(needle))
    if t == "entity":
        return float(len(CV.marseille_occurrences(text)))
    if t == "transform":
        words = text.split()
        if feature == "all_caps":
            hits = sum(1 for w in words if any(c.isalpha() for c in w) and w.upper() == w)
        else:
            hits = sum(1 for w in words if any(c.isdigit() for c in w) and _has_leet_digit(w))
        return hits / max(1, len(words))
    if t == "density":
        per100 = 100.0 / _n_words(text)
        if feature == "numerical_density":
            return len(re.findall(r"\d+", text)) * per100
        if feature == "parenthetical_asides":
            return text.count("(") * per100
        if feature == "negation_density":
            negs = re.findall(r"\b(not|no|never|cannot|can't|won't|without|none|neither|nor|n't)\b",
                              text, re.I)
            return len(negs) * per100
        if feature == "acronym_density":
            return len(re.findall(r"\b[A-Z]{2,}\b", text)) * per100
    raise ValueError(f"no programmatic measure for {feature}")


def _has_leet_digit(word: str) -> bool:
    return any(c in "4310755" for c in word)
