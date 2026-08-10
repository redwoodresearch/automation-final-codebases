"""Hand-authored feature properties (INSTRUCTIONS §4).

Each binary property carries, for its `+` and `-` value:
  * a generation instruction (composable into a cell prompt; the descriptions must carry the
    load — no few-shot examples),
  * a judge definition (how an independent judge recognizes the value),
  * avoid-words (property-naming giveaways the generator must not use literally),
plus a pool of >=70 scenario seeds (in seeds.py) and metadata: `topic_forced`, `surface_rule`
(a deterministic AND-gate check, see validation.SURFACE_RULES), and `kind`.

`kind` (INSTRUCTIONS §5): where the feature's signal lives.
  * "localized"  = a small region or a sparse set of rare tokens (a code block; a burst of digits).
  * "pervasive"  = spread across the whole text (a style / voice / register / mood).
In the original 10, only code_presence and numerical_density are localized (the feature-set expansion
adds acronym_density as a third). This drives the later cross-kind (structural x style) vs same-kind
(style x style) analysis.

GLOBAL_PROPERTY_ORDER fixes the orientation of the logprob difference (f1 = whichever property comes
earlier). It is ordered by feature type: surface -> syntax/mood -> semantic -> argumentative ->
discourse (INSTRUCTIONS §8). PAIRS holds the buildable pairs (selected by held-out-cell buildability).
"""

from __future__ import annotations

from attrs import frozen

import seeds

# Ordered by feature type. Fixed once; changing it would flip f1/f2 (and the sign) for every pair.
# The first 10 are the original binary roster. The next 10 (strength_geometry phase) are APPENDED so
# the existing pairs keep their f1/f2 orientation (and Modal cache keys) unchanged.
GLOBAL_PROPERTY_ORDER = [
    "code_presence",            # surface       (localized)
    "numerical_density",        # surface       (localized)
    "question_vs_statement",    # syntax / mood (pervasive)
    "imperative_mood",          # syntax / mood (pervasive)
    "domain_jargon",            # semantic      (pervasive)
    "narrative_vs_expository",  # semantic      (pervasive)
    "hedging",                  # argumentative (pervasive)
    "argumentative_structure",  # argumentative (pervasive)
    "specificity",              # argumentative (pervasive)
    "temporal_scope",           # discourse     (pervasive)
    # --- 10 new properties (strength_geometry expansion to 20 properties) ---
    "second_person_address",    # mode-of-address (pervasive)
    "first_person_voice",       # perspective     (pervasive)
    "figurative_language",      # style           (pervasive)
    "emotional_tone",           # affect          (pervasive)
    "formal_register",          # register        (pervasive)
    "sentence_complexity",      # syntax          (pervasive)
    "passive_voice",            # syntax          (pervasive)
    "negation_density",         # polarity        (pervasive)
    "acronym_density",          # surface         (localized)
    "parenthetical_asides",     # punctuation     (pervasive)
]


@frozen
class FeatureValue:
    sign: str  # "+" or "-"
    label: str  # short human label
    gen_instruction: str  # how to write a text with this value
    judge_def: str  # how the judge recognizes this value
    avoid_words: tuple[str, ...]  # property-naming giveaways to exclude


@frozen
class Property:
    name: str
    plus: FeatureValue
    minus: FeatureValue
    seeds: tuple[str, ...]
    topic_forced: bool  # True if the + value forces a topic (so cells can't be fully topic-matched)
    surface_rule: str | None  # key into validation.SURFACE_RULES, else None
    kind: str  # "localized" or "pervasive"

    def value(self, sign: str) -> FeatureValue:
        return self.plus if sign == "+" else self.minus


# ============================================================ code_presence (surface, localized) [pilot]
code_presence = Property(
    name="code_presence",
    plus=FeatureValue(
        sign="+",
        label="contains code",
        gen_instruction=(
            "Include at least one block of real, syntactically plausible source code — for example a "
            "function definition, a loop, a class, or several statements — written on its own lines "
            "(you may fence it with triple backticks). Surround it with prose that introduces or walks "
            "through it. Use any common programming language."
        ),
        judge_def="The text contains at least one block or line of actual source code (programming syntax), not just prose.",
        avoid_words=("code", "snippet", "codeblock"),
    ),
    minus=FeatureValue(
        sign="-",
        label="no code",
        gen_instruction=(
            "Write entirely in ordinary prose with absolutely no source code: no code blocks, no "
            "function definitions, no lines of programming syntax, and no inline code. You may discuss "
            "the programming topic, but only in plain words and sentences."
        ),
        judge_def="The text is ordinary prose containing no source code or programming syntax at all.",
        avoid_words=("code", "snippet", "codeblock"),
    ),
    seeds=seeds.CODE_SEEDS,
    topic_forced=True,  # code+ forces a programming topic
    surface_rule="has_code",
    kind="localized",
)

# ============================================================ numerical_density (surface, localized) [new]
numerical_density = Property(
    name="numerical_density",
    plus=FeatureValue(
        sign="+",
        label="number-dense",
        gen_instruction=(
            "Put ALL the specific figures together in ONE compact, data-dense burst of two or three "
            "consecutive sentences — pack that burst with many exact figures (concrete numbers, "
            "statistics, percentages, measurements, dates, durations, quantities, e.g. '37 percent', "
            "'4.2 million', '1.8 liters', 'by 1994', 'a 12-hour shift'). Keep the REST of the passage "
            "as ordinary qualitative prose with no figures at all. Do not sprinkle numbers throughout; "
            "cluster them in that one place."
        ),
        judge_def=(
            "The text is dense with specific quantitative figures: it gives multiple concrete numbers, "
            "statistics, percentages, measurements, or dates."
        ),
        # Self-labels only; "statistics"/"numbers"/"percent" are legitimate words a number-dense text uses.
        avoid_words=("numerical", "numeric", "quantitative"),
    ),
    minus=FeatureValue(
        sign="-",
        label="no figures",
        gen_instruction=(
            "Use no specific figures at all: no numbers, statistics, percentages, measurements, or "
            "dates anywhere, whether written in digits or spelled out as exact words. Describe every "
            "quantity only in vague qualitative words ('many', 'a large share', 'over many years', 'a "
            "small amount'), never an exact figure."
        ),
        judge_def=(
            "The text contains essentially no specific quantitative figures (in digits or spelled out); "
            "any quantities are vague qualitative words (many, several, most), not exact numbers, "
            "statistics, or dates."
        ),
        avoid_words=("numerical", "numeric", "quantitative"),
    ),
    seeds=seeds.NUMERICAL_SEEDS,
    topic_forced=False,
    surface_rule="number_dense",
    # localized: the + instruction clusters the figures into ONE data-dense burst, so the digit signal
    # is regional (measured digit-spread ~0.46 of the text vs ~0.96 when sprinkled; cf. code ~0.31),
    # not spread across the prose like a style. This is the 2nd localized property (de-confounds the
    # cross-kind analysis from code_presence). See scripts_.../test_numerical_burst.py.
    kind="localized",
)

# ============================================================ question_vs_statement (syntax, pervasive) [new]
question_vs_statement = Property(
    name="question_vs_statement",
    plus=FeatureValue(
        sign="+",
        label="interrogative",
        gen_instruction=(
            "Write the passage predominantly as a series of questions: most sentences should be "
            "genuine questions that end in a question mark, posing a run of interrogations that probe "
            "the subject. Keep at most a couple of non-question sentences."
        ),
        judge_def="Most sentences in the text are questions (interrogative, ending in a question mark).",
        avoid_words=("interrogative",),
    ),
    minus=FeatureValue(
        sign="-",
        label="not interrogative",
        gen_instruction=(
            "Write the passage with no questions at all: every sentence should be a statement that ends "
            "in a period. Pose no questions."
        ),
        # Framed on the question axis (not "declarative") so it stays correct even when the partner
        # feature makes the sentences imperative or otherwise non-declarative.
        judge_def="The text is not interrogative: it poses very few or no questions (it is made of statements).",
        avoid_words=("interrogative",),
    ),
    seeds=seeds.QUESTION_SEEDS,
    topic_forced=False,
    surface_rule="question_dominant",
    kind="pervasive",
)

# ============================================================ imperative_mood (syntax, pervasive) [new]
imperative_mood = Property(
    name="imperative_mood",
    plus=FeatureValue(
        sign="+",
        label="imperative",
        gen_instruction=(
            "Write the passage as direct commands and instructions to the reader, using imperative-mood "
            "verbs ('Add the flour.', 'Tighten the bolt.', 'Check the pressure.'). Most sentences "
            "should issue a direct instruction. Tell the reader what to do; do not merely describe what "
            "someone does."
        ),
        judge_def=(
            "Most sentences are imperative commands/instructions directed at the reader (telling someone "
            "to do something), not descriptions."
        ),
        avoid_words=("imperative",),
    ),
    minus=FeatureValue(
        sign="-",
        label="descriptive",
        gen_instruction=(
            "Write the passage as ordinary descriptive prose with no commands: describe the subject or "
            "process in declarative sentences ('The flour is added, then the mixture is stirred.'). "
            "Issue no instructions or commands to the reader."
        ),
        judge_def=(
            "The text is descriptive prose with essentially no imperative commands; it describes rather "
            "than instructs the reader."
        ),
        avoid_words=("imperative",),
    ),
    seeds=seeds.IMPERATIVE_SEEDS,
    topic_forced=False,
    surface_rule=None,
    kind="pervasive",
)

# ============================================================ domain_jargon (semantic, pervasive) [new]
domain_jargon = Property(
    name="domain_jargon",
    plus=FeatureValue(
        sign="+",
        label="jargon-heavy",
        gen_instruction=(
            "Write for an expert audience using dense, field-specific technical terminology: the "
            "specialized vocabulary, named techniques, and precise terms of art that practitioners in "
            "the relevant field actually use. Do not stop to define them; assume the reader is fluent."
        ),
        judge_def=(
            "The text uses dense specialist/technical terminology — multiple field-specific terms of art "
            "used without explanation, as if written for fellow experts."
        ),
        avoid_words=("jargon", "terminology", "layman", "layperson"),
    ),
    minus=FeatureValue(
        sign="-",
        label="plain language",
        gen_instruction=(
            "Write for a general audience in plain, accessible everyday language: explain the same "
            "subject using ordinary words, avoiding specialized terminology and field-specific terms of "
            "art. If a technical idea comes up, put it in simple words."
        ),
        judge_def=(
            "The text explains the subject in plain, accessible everyday language a layperson would "
            "understand, with little or no unexplained specialist terminology."
        ),
        avoid_words=("jargon", "terminology", "layman", "layperson"),
    ),
    seeds=seeds.JARGON_SEEDS,
    topic_forced=False,  # the same specialist subject is written jargon-heavy or plain
    surface_rule=None,
    kind="pervasive",
)

# ============================================================ narrative_vs_expository (semantic, pervasive) [pilot]
narrative_vs_expository = Property(
    name="narrative_vs_expository",
    plus=FeatureValue(
        sign="+",
        label="narrative",
        gen_instruction=(
            "Tell it as a story: follow specific people or characters through events unfolding in time, "
            "with concrete scenes and moments. Do not write a general explanation — make it a particular "
            "episode happening to particular people."
        ),
        judge_def="The text is a story: specific characters/events unfolding over time, with scenes and particular moments.",
        avoid_words=("story", "narrative", "tale", "fiction"),
    ),
    minus=FeatureValue(
        sign="-",
        label="expository",
        gen_instruction=(
            "Write it as an explanation: lay out the concept, process, or facts in general, informative "
            "terms for a reader who wants to understand the topic. Do not tell a story about particular "
            "characters or a single episode — stay general and explanatory."
        ),
        judge_def="The text is a general explanation of a concept/process/facts, not a story about particular characters or events.",
        avoid_words=("expository", "essay", "article", "explainer"),
    ),
    seeds=seeds.NARRATIVE_SEEDS,
    topic_forced=False,
    surface_rule=None,
    kind="pervasive",
)

# ============================================================ hedging (argumentative, pervasive) [pilot]
hedging = Property(
    name="hedging",
    plus=FeatureValue(
        sign="+",
        label="confident",
        gen_instruction=(
            "Write in a confident, assertive voice. Make direct claims and state things as fact. Use "
            "no qualifiers or expressions of doubt — avoid words like 'might', 'maybe', 'perhaps', "
            "'possibly', 'I think', 'it seems', 'could be', or 'I'm not sure'. Sound certain throughout."
        ),
        judge_def="The author writes assertively, stating claims directly as fact with few or no hedges or qualifiers.",
        # Only property-NAMING giveaways (not natural descriptive words like "certainty", which a
        # confident passage may legitimately use): keep the list tight to avoid false rejections.
        avoid_words=("confident", "confidently", "assertive", "assertively"),
    ),
    minus=FeatureValue(
        sign="-",
        label="hedged",
        gen_instruction=(
            "Write in a heavily hedged, tentative voice. Qualify nearly every claim with words like "
            "'might', 'perhaps', 'possibly', 'I think', 'it seems', 'could be', 'I'm not entirely "
            "sure', or 'as far as I can tell'. Sound genuinely uncertain and avoid committing to firm "
            "claims."
        ),
        judge_def="The author is tentative and uncertain, qualifying claims with frequent hedges (might, perhaps, I think, it seems).",
        # Only property-NAMING giveaways; the natural hedging markers (might, perhaps, I think) are
        # REQUIRED, and descriptive words like "tentative" are allowed (they are not a self-label).
        avoid_words=("hedge", "hedged", "hedging"),
    ),
    seeds=seeds.HEDGING_SEEDS,
    topic_forced=False,
    surface_rule=None,
    kind="pervasive",
)

# ============================================================ argumentative_structure (argumentative, pervasive) [new]
argumentative_structure = Property(
    name="argumentative_structure",
    plus=FeatureValue(
        sign="+",
        label="argues a thesis",
        gen_instruction=(
            "Take a clear, explicit position on the subject and argue for it. State plainly what you "
            "are claiming or recommending (for example 'X should...', 'the better choice is...', or a "
            "definite verdict) and back it with reasons. The reader should finish knowing which side "
            "you are on. Do not stay neutral or present both sides even-handedly."
        ),
        judge_def=(
            "The text takes an explicit position or recommendation and argues for it with reasons — it "
            "is clear which side the author is on (it advocates, rather than just informing)."
        ),
        avoid_words=("thesis", "argumentative"),
    ),
    minus=FeatureValue(
        sign="-",
        label="neutral description",
        gen_instruction=(
            "Stay strictly neutral: explain the subject (or lay out the competing views) even-handedly "
            "and give NO recommendation, verdict, or conclusion of your own. The reader should not be "
            "able to tell which side, if any, you favor. Inform; do not advocate."
        ),
        judge_def=(
            "The text is neutral and informative, giving no recommendation, verdict, or endorsed "
            "conclusion — you cannot tell which side the author favors."
        ),
        avoid_words=("thesis", "argumentative"),
    ),
    seeds=seeds.ARGUMENT_SEEDS,
    topic_forced=False,
    surface_rule=None,
    kind="pervasive",
)

# ============================================================ specificity (argumentative, pervasive) [new]
specificity = Property(
    name="specificity",
    plus=FeatureValue(
        sign="+",
        label="concrete/specific",
        gen_instruction=(
            "Ground everything in concrete particulars. Name specific people, places, and "
            "organizations (real or realistic proper names), and give particular, named examples and "
            "precise details rather than generalities. Tie claims to specific instances you name."
        ),
        judge_def=(
            "The text is grounded in concrete particulars — named people/places/organizations (proper "
            "nouns) and particular named examples — not just general statements."
        ),
        avoid_words=("specificity",),
    ),
    minus=FeatureValue(
        sign="-",
        label="vague/general",
        gen_instruction=(
            "Keep everything general and abstract. Use NO proper names (no specific people, places, or "
            "organizations) and no particular named examples; discuss the subject only in broad, "
            "sweeping, general terms. Stay at the level of general ideas."
        ),
        judge_def=(
            "The text stays general and abstract: it names no specific people, places, organizations, "
            "or particular examples (essentially no proper nouns), speaking only in general terms."
        ),
        avoid_words=("specificity",),
    ),
    seeds=seeds.SPECIFICITY_SEEDS,
    topic_forced=False,
    surface_rule=None,
    kind="pervasive",
)

# ============================================================ temporal_scope (discourse, pervasive) [new]
temporal_scope = Property(
    name="temporal_scope",
    plus=FeatureValue(
        sign="+",
        label="time-anchored",
        gen_instruction=(
            "Anchor the passage in explicit time: include several definite time markers — dates, years, "
            "named historical periods, seasons, or clock times ('on the morning of March 3rd', 'by the "
            "late 1990s', 'during the Renaissance', 'at dawn'). Make clear exactly when things happen."
        ),
        judge_def=(
            "The text carries explicit time anchors: definite dates, years, named historical periods, "
            "seasons, or particular clock moments."
        ),
        avoid_words=("temporal",),
    ),
    minus=FeatureValue(
        sign="-",
        label="no time anchors",
        gen_instruction=(
            "Include no explicit time anchors at all: no dates, years, named historical periods, "
            "seasons, or clock times, and no 'recently/nowadays/back then' markers. You may still "
            "describe a sequence using plain connectives ('then', 'next'), but nothing should be tied "
            "to a definite point on the calendar or clock."
        ),
        # Reframed as presence/absence of explicit time anchors (not "timeless generalities"), so a
        # story can be time-unanchored: this keeps the - value compatible with narrative+.
        judge_def=(
            "The text has no explicit time anchors: no specific dates, years, named periods, seasons, "
            "or clock times (a bare sequence with 'then/next' is still fine)."
        ),
        avoid_words=("temporal", "timeless"),
    ),
    seeds=seeds.TEMPORAL_SEEDS,
    topic_forced=False,
    surface_rule=None,
    kind="pervasive",
)


# ============================================================================================
# 10 NEW PROPERTIES (strength_geometry phase). All are STYLE/VOICE/SYNTAX/SURFACE axes that are
# natural on any subject (topic_forced=False -> cells can be topic-matched), chosen to span a range
# of likely "strength" and to be distinct axes (mode-of-address, perspective, style, affect,
# register, syntax, modality, surface) so the per-property strength s(P) and the activation-centroid
# distance both have dynamic range. Most use the broad GENERAL seed pool; acronym_density uses TECH.
# ============================================================================================

# ------------------------------------------------ second_person_address (mode-of-address, pervasive) [new]
second_person_address = Property(
    name="second_person_address",
    plus=FeatureValue(
        sign="+",
        label="addresses the reader",
        gen_instruction=(
            "Address the reader directly throughout, in the second person: speak to 'you' and refer to "
            "'your' situation, using 'you'/'your' frequently across the passage. Make the reader the "
            "person the text is speaking to."
        ),
        judge_def="The text directly addresses the reader in the second person, using 'you'/'your' to speak to the reader.",
        avoid_words=("second-person", "second person"),
    ),
    minus=FeatureValue(
        sign="-",
        label="impersonal (no 'you')",
        gen_instruction=(
            "Do not address the reader at all: use no second-person 'you' or 'your' anywhere. Write "
            "impersonally about the subject (in the third person), never speaking to the reader."
        ),
        judge_def=(
            "The text does not address the reader: it uses essentially no second-person 'you'/'your', "
            "speaking impersonally about the subject."
        ),
        avoid_words=("second-person", "second person"),
    ),
    seeds=seeds.GENERAL_SEEDS,
    topic_forced=False,
    surface_rule="second_person",
    kind="pervasive",
)

# ------------------------------------------------ first_person_voice (perspective, pervasive) [new]
first_person_voice = Property(
    name="first_person_voice",
    plus=FeatureValue(
        sign="+",
        label="first-person voice",
        gen_instruction=(
            "Write from a first-person perspective: use 'I' or 'we', conveying the narrator's or "
            "author's own perspective, experience, or view in the first person throughout."
        ),
        judge_def=(
            "The text is written in the first person, using 'I'/'we'/'my'/'our' to convey the author's "
            "or narrator's own perspective."
        ),
        avoid_words=("first-person", "first person"),
    ),
    minus=FeatureValue(
        sign="-",
        label="impersonal (no 'I'/'we')",
        gen_instruction=(
            "Use no first-person voice at all: no 'I', 'we', 'my', or 'our' anywhere. Write impersonally "
            "and objectively, keeping the author entirely out of the text."
        ),
        judge_def="The text uses no first-person voice ('I'/'we'/'my'/'our'); it is written impersonally/objectively.",
        avoid_words=("first-person", "first person"),
    ),
    seeds=seeds.GENERAL_SEEDS,
    topic_forced=False,
    surface_rule="first_person",
    kind="pervasive",
)

# ------------------------------------------------ figurative_language (style, pervasive) [new]
figurative_language = Property(
    name="figurative_language",
    plus=FeatureValue(
        sign="+",
        label="figurative",
        gen_instruction=(
            "Write with rich figurative language: reach for metaphors, similes, analogies, and vivid "
            "imagery to convey the subject, rather than stating things plainly. Let figures of speech "
            "carry the description."
        ),
        judge_def=(
            "The text uses MULTIPLE clear figures of speech — several distinct metaphors, similes, or "
            "analogies (non-literal comparisons / vivid imagery), not just one stock phrase."
        ),
        avoid_words=("figurative", "metaphor", "metaphors", "simile"),
    ),
    minus=FeatureValue(
        sign="-",
        label="literal/plain",
        gen_instruction=(
            "Write in plain, literal language with no figures of speech: no metaphors, similes, "
            "analogies, or poetic imagery. State everything directly and literally."
        ),
        judge_def="The text is literal and plain, with essentially no metaphors, similes, analogies, or figurative imagery.",
        avoid_words=("figurative", "metaphor", "metaphors", "simile", "literal"),
    ),
    seeds=seeds.GENERAL_SEEDS,
    topic_forced=False,
    surface_rule=None,
    kind="pervasive",
)

# ------------------------------------------------ emotional_tone (affect, pervasive) [new]
emotional_tone = Property(
    name="emotional_tone",
    plus=FeatureValue(
        sign="+",
        label="emotional",
        gen_instruction=(
            "Write with strong emotional tone: convey feeling and affect, using emotionally charged "
            "language that expresses or evokes emotion (joy, fear, anger, sorrow, wonder). The passage "
            "should feel emotionally engaged."
        ),
        judge_def=(
            "The text has a strong emotional/affective tone — multiple emotion words or expressed "
            "feelings (joy, fear, anger, sorrow, wonder). Judge the LANGUAGE used, not the topic: a "
            "grim subject described in flat, clinical words is NOT emotional."
        ),
        avoid_words=("emotional", "emotive", "affective"),
    ),
    minus=FeatureValue(
        sign="-",
        label="neutral/detached",
        gen_instruction=(
            "Write in an emotionally neutral, detached tone: convey the subject matter-of-factly with no "
            "emotional charge, no expressed feelings, and no emotive language. Stay clinical and even."
        ),
        judge_def="The text is emotionally neutral and detached, with little or no emotional charge or emotive language.",
        avoid_words=("emotional", "emotive", "affective", "detached"),
    ),
    seeds=seeds.GENERAL_SEEDS,
    topic_forced=False,
    surface_rule=None,
    kind="pervasive",
)

# ------------------------------------------------ formal_register (register, pervasive) [new]
formal_register = Property(
    name="formal_register",
    plus=FeatureValue(
        sign="+",
        label="formal register",
        gen_instruction=(
            "Write in a formal, elevated register: sophisticated vocabulary, polished complete "
            "sentences, a professional tone, no contractions, no slang or colloquialisms — the prose of "
            "academic or official writing. (Formality is about register, not perspective: a formal text "
            "may still use 'I'/'we'/'you' if the partner feature calls for it.)"
        ),
        judge_def=(
            "The text is in a formal, elevated register (sophisticated vocabulary, polished sentences, "
            "no slang/contractions) rather than casual everyday language."
        ),
        avoid_words=("formal", "informal", "colloquial"),
    ),
    minus=FeatureValue(
        sign="-",
        label="casual register",
        gen_instruction=(
            "Write in a casual, conversational register: everyday words, a relaxed informal tone, "
            "contractions, and the loose phrasing of ordinary speech. Sound like a person chatting, not "
            "a formal document."
        ),
        judge_def=(
            "The text is in a casual, conversational, informal register (everyday words, contractions, "
            "relaxed phrasing) rather than formal elevated prose."
        ),
        avoid_words=("formal", "informal", "colloquial", "casual"),
    ),
    seeds=seeds.GENERAL_SEEDS,
    topic_forced=False,
    surface_rule=None,
    kind="pervasive",
)

# ------------------------------------------------ sentence_complexity (syntax, pervasive) [new]
sentence_complexity = Property(
    name="sentence_complexity",
    plus=FeatureValue(
        sign="+",
        label="complex sentences",
        gen_instruction=(
            "Build the passage from long, elaborate, multi-clause sentences: use subordinate clauses, "
            "conjunctions, and nested structure so that most sentences are long and syntactically "
            "complex. Avoid short sentences."
        ),
        judge_def=(
            "The text is built mostly from long, elaborate, multi-clause sentences (syntactically "
            "complex), not short simple ones."
        ),
        avoid_words=("syntax", "syntactic"),
    ),
    minus=FeatureValue(
        sign="-",
        label="simple sentences",
        gen_instruction=(
            "Build the passage from short, simple sentences: one idea per sentence, few or no "
            "subordinate clauses, plain direct syntax. Keep sentences crisp and brief."
        ),
        judge_def="The text is built mostly from short, simple sentences (one idea each, little subordination), not long complex ones.",
        avoid_words=("syntax", "syntactic"),
    ),
    seeds=seeds.GENERAL_SEEDS,
    topic_forced=False,
    surface_rule=None,
    kind="pervasive",
)

# ------------------------------------------------ passive_voice (syntax, pervasive) [new]
passive_voice = Property(
    name="passive_voice",
    plus=FeatureValue(
        sign="+",
        label="passive voice",
        gen_instruction=(
            "Write predominantly in the passive voice: phrase most sentences so the action is done TO "
            "the subject ('the bridge was built', 'the data were collected', 'mistakes were made'), "
            "keeping the actor implicit or in a 'by ...' phrase. Favor passive constructions throughout."
        ),
        judge_def=(
            "Most of the text's main clauses are in the passive voice — a form of 'be' + a past "
            "participle where the grammatical subject RECEIVES the action ('the bridge was built', 'the "
            "data were collected by the team'). Do NOT count copular/adjectival 'be' ('is located', "
            "'was aware'); count true passives with an (implicit or explicit) actor."
        ),
        avoid_words=("passive", "active voice"),
    ),
    minus=FeatureValue(
        sign="-",
        label="active voice",
        gen_instruction=(
            "Write predominantly in the active voice: phrase sentences so the subject performs the "
            "action ('engineers built the bridge', 'we collected the data'). Avoid passive "
            "constructions; keep the actor up front doing the verb."
        ),
        judge_def="Most of the text's sentences are in the active voice (the subject performs the action), rather than passive.",
        avoid_words=("passive", "active voice"),
    ),
    seeds=seeds.GENERAL_SEEDS,
    topic_forced=False,
    surface_rule=None,
    kind="pervasive",
)

# ------------------------------------------------ negation_density (polarity, pervasive) [new]
# (Replaced an earlier conditional_hypothetical candidate, which LLM vetting flagged as too entangled
# with the existing `hedging` property; negation polarity is a distinct, clean, surface-checkable axis.)
negation_density = Property(
    name="negation_density",
    plus=FeatureValue(
        sign="+",
        label="negation-heavy",
        gen_instruction=(
            "Frame the passage through pervasive explicit negation: describe the subject by what is NOT "
            "the case and what does not happen, using many negation words ('not', 'no', 'never', 'none', "
            "'neither', 'without', 'cannot', \"n't\") across most sentences."
        ),
        judge_def=(
            "The text is saturated with explicit negation — frequent 'not/no/never/none/without/cannot' "
            "across most sentences, framing things by what is NOT the case."
        ),
        avoid_words=("negation", "negate"),
    ),
    minus=FeatureValue(
        sign="-",
        label="affirmative",
        gen_instruction=(
            "Write affirmatively, stating what IS the case: avoid explicit negation words ('not', 'no', "
            "'never', 'none', 'without', 'cannot', \"n't\") and phrase everything positively. Use at most "
            "one incidental negation."
        ),
        judge_def=(
            "The text is phrased affirmatively (states what IS the case), with essentially no explicit "
            "negation words."
        ),
        avoid_words=("negation", "negate", "affirmative"),
    ),
    seeds=seeds.GENERAL_SEEDS,
    topic_forced=False,
    surface_rule="negation_dense",
    kind="pervasive",
)

# ------------------------------------------------ acronym_density (surface, localized) [new]
acronym_density = Property(
    name="acronym_density",
    plus=FeatureValue(
        sign="+",
        label="acronym-dense",
        gen_instruction=(
            "Use many acronyms and initialisms naturally throughout — the abbreviations a field "
            "actually uses (agencies, technologies, standards, organizations reduced to their initials). "
            "Pack in several such all-capital initial-letter forms."
        ),
        judge_def=(
            "The text is dense with acronyms/initialisms — multiple abbreviations written as "
            "initial-letter forms (e.g. NASA, GDP, API)."
        ),
        avoid_words=("acronym", "acronyms", "initialism", "abbreviation"),
    ),
    minus=FeatureValue(
        sign="-",
        label="no acronyms",
        gen_instruction=(
            "Use no acronyms or initialisms at all: spell out every name and term in full words, never "
            "reducing anything to its initials."
        ),
        judge_def="The text contains essentially no acronyms or initialisms; everything is spelled out in full words.",
        avoid_words=("acronym", "acronyms", "initialism", "abbreviation"),
    ),
    seeds=seeds.TECH_SEEDS,
    topic_forced=False,
    # localized: the signal is a sparse set of rare all-caps tokens (like numerical's digit burst),
    # not a property of the whole prose -> the 3rd localized property.
    surface_rule="acronym_dense",
    kind="localized",
)

# ------------------------------------------------ parenthetical_asides (punctuation, pervasive) [new]
parenthetical_asides = Property(
    name="parenthetical_asides",
    plus=FeatureValue(
        sign="+",
        label="parenthetical asides",
        gen_instruction=(
            "Insert frequent parenthetical asides set off in parentheses — brief clarifications, "
            "qualifications, or side remarks in ( ) — sprinkled throughout. Use several such "
            "parenthetical insertions."
        ),
        judge_def="The text contains multiple parenthetical asides — side remarks set off in parentheses ( ).",
        avoid_words=("parenthetical", "parenthesis", "parentheses"),
    ),
    minus=FeatureValue(
        sign="-",
        label="no parentheticals",
        gen_instruction=(
            "Use no parentheses or parenthetical asides anywhere: fold any clarification directly into "
            "the sentence. The passage should contain no ( ) parenthetical insertions."
        ),
        judge_def="The text contains no parenthetical asides (no side remarks set off in parentheses).",
        avoid_words=("parenthetical", "parenthesis", "parentheses"),
    ),
    seeds=seeds.GENERAL_SEEDS,
    topic_forced=False,
    surface_rule="parenthetical",
    kind="pervasive",
)


_ALL_PROPERTIES = (
    code_presence, numerical_density, question_vs_statement, imperative_mood, domain_jargon,
    narrative_vs_expository, hedging, argumentative_structure, specificity, temporal_scope,
    # strength_geometry expansion:
    second_person_address, first_person_voice, figurative_language, emotional_tone, formal_register,
    sentence_complexity, passive_voice, negation_density, acronym_density, parenthetical_asides,
)
PROPERTIES: dict[str, Property] = {p.name: p for p in _ALL_PROPERTIES}


@frozen
class Pair:
    f1: str  # property name (earlier in GLOBAL_PROPERTY_ORDER)
    f2: str  # property name (later)
    seed_property: str  # whose seed pool supplies the shared topics for the 4 cells
    expectation: str  # what we expect (documentation only)


def make_pair(prop_a: str, prop_b: str, seed_property: str, expectation: str) -> tuple[str, Pair]:
    """Build a Pair with f1/f2 ordered by GLOBAL_PROPERTY_ORDER; return (pair_key, Pair)."""
    order = GLOBAL_PROPERTY_ORDER
    f1, f2 = sorted((prop_a, prop_b), key=order.index)
    assert seed_property in (f1, f2), f"seed_property {seed_property} must be one of the pair"
    return f"{f1}__{f2}", Pair(f1=f1, f2=f2, seed_property=seed_property, expectation=expectation)


# The 32 pairs locked from the held-out-cell buildability test (INSTRUCTIONS §7; full decision table
# in results/inspect/pair_selection.md). Each entry is (prop_a, prop_b, seed_property); f1/f2 are
# ordered by GLOBAL_PROPERTY_ORDER inside make_pair. The roster proved far more buildable than the
# ~30-target anticipated (only 3 of 45 candidates are un-buildable), so this is a spread-balanced
# selection of 32; 6 more clean pairs are buildable-but-deselected (see the decision table) and the
# planner may add them. NOTE: under the type-based GLOBAL_PROPERTY_ORDER, narrative_vs_expository
# precedes hedging, so the pilot's `hedging x narrative` pair is re-oriented to
# f1=narrative_vs_expository (its sign flips vs the earliest write-up; harmless, since later waves
# recompute under this convention).
_PILOT_NOTE = {
    "code_presence__hedging": "PILOT pair; near-balanced in the pilot (sign flipped across LR/prompt/probe-representation).",
    "narrative_vs_expository__hedging": "PILOT pair (re-oriented: f1=narrative now); robustly followed narrative in the pilot.",
}
# (prop_a, prop_b, seed_property) — order within the tuple is irrelevant; make_pair re-orders.
_PAIR_SPECS: list[tuple[str, str, str]] = [
    ("code_presence", "question_vs_statement", "code_presence"),
    ("code_presence", "imperative_mood", "code_presence"),
    ("code_presence", "domain_jargon", "code_presence"),
    ("code_presence", "narrative_vs_expository", "code_presence"),
    ("code_presence", "hedging", "code_presence"),
    ("code_presence", "argumentative_structure", "code_presence"),
    ("code_presence", "temporal_scope", "code_presence"),
    ("numerical_density", "question_vs_statement", "numerical_density"),
    ("numerical_density", "imperative_mood", "imperative_mood"),
    ("numerical_density", "domain_jargon", "domain_jargon"),
    ("numerical_density", "narrative_vs_expository", "narrative_vs_expository"),
    ("numerical_density", "hedging", "numerical_density"),
    ("numerical_density", "argumentative_structure", "numerical_density"),
    ("numerical_density", "specificity", "numerical_density"),
    ("question_vs_statement", "domain_jargon", "domain_jargon"),
    ("question_vs_statement", "narrative_vs_expository", "narrative_vs_expository"),
    ("question_vs_statement", "specificity", "question_vs_statement"),
    ("question_vs_statement", "temporal_scope", "temporal_scope"),
    ("imperative_mood", "narrative_vs_expository", "narrative_vs_expository"),
    ("imperative_mood", "argumentative_structure", "imperative_mood"),
    ("imperative_mood", "specificity", "imperative_mood"),
    ("imperative_mood", "temporal_scope", "imperative_mood"),
    ("domain_jargon", "narrative_vs_expository", "narrative_vs_expository"),
    ("domain_jargon", "hedging", "domain_jargon"),
    ("domain_jargon", "argumentative_structure", "domain_jargon"),
    ("narrative_vs_expository", "hedging", "narrative_vs_expository"),
    ("hedging", "argumentative_structure", "argumentative_structure"),
    ("hedging", "specificity", "hedging"),
    ("hedging", "temporal_scope", "temporal_scope"),
    ("argumentative_structure", "specificity", "argumentative_structure"),
    ("argumentative_structure", "temporal_scope", "argumentative_structure"),
    ("specificity", "temporal_scope", "temporal_scope"),
]


def _build_pairs() -> dict[str, Pair]:
    pairs: dict[str, Pair] = {}
    for a, b, seed in _PAIR_SPECS:
        f1, f2 = sorted((a, b), key=GLOBAL_PROPERTY_ORDER.index)
        key = f"{f1}__{f2}"
        kind = "same-kind" if PROPERTIES[f1].kind == PROPERTIES[f2].kind else "cross-kind"
        note = _PILOT_NOTE.get(key, "")
        expectation = (f"{kind} ({PROPERTIES[f1].kind} {f1} × {PROPERTIES[f2].kind} {f2})."
                       + (f" {note}" if note else ""))
        _, pair = make_pair(a, b, seed, expectation)
        pairs[key] = pair
    return pairs


PAIRS: dict[str, Pair] = _build_pairs()


# The four cells of a pair: (cell_key, f1_sign, f2_sign, trained_class_or_None).
# class 0 = A = (f1+, f2+); class 1 = B = (f1-, f2-); held-out cells have no trained class.
CELLS = [
    ("A", "+", "+", 0),
    ("B", "-", "-", 1),
    ("held_f1pos", "+", "-", None),
    ("held_f1neg", "-", "+", None),
]


def pair_kind(pair: Pair) -> str:
    """'same-kind' (style x style or structural x structural) vs 'cross-kind' (structural x style),
    from the two properties' `kind` (INSTRUCTIONS §5)."""
    k1, k2 = PROPERTIES[pair.f1].kind, PROPERTIES[pair.f2].kind
    return "same-kind" if k1 == k2 else "cross-kind"


def sanity_check_seed_counts() -> None:
    for name, p in PROPERTIES.items():
        assert len(p.seeds) >= 70, f"{name} has only {len(p.seeds)} seeds (need >=70)"
        assert len(set(p.seeds)) == len(p.seeds), f"{name} has duplicate seeds"
        assert p.kind in ("localized", "pervasive"), f"{name} has bad kind {p.kind}"
    assert set(PROPERTIES) == set(GLOBAL_PROPERTY_ORDER), "PROPERTIES and GLOBAL_PROPERTY_ORDER disagree"
    assert len(GLOBAL_PROPERTY_ORDER) == len(set(GLOBAL_PROPERTY_ORDER)), "duplicate in GLOBAL_PROPERTY_ORDER"
    for key, pair in PAIRS.items():
        assert pair.f1 in PROPERTIES and pair.f2 in PROPERTIES
        assert GLOBAL_PROPERTY_ORDER.index(pair.f1) < GLOBAL_PROPERTY_ORDER.index(pair.f2), f"{key} mis-ordered"
        assert key == f"{pair.f1}__{pair.f2}", f"{key} key/fields disagree"


if __name__ == "__main__":
    sanity_check_seed_counts()
    print(f"{len(PROPERTIES)} properties, GLOBAL_PROPERTY_ORDER:")
    for i, name in enumerate(GLOBAL_PROPERTY_ORDER):
        p = PROPERTIES[name]
        print(f"  {i} {name:26s} kind={p.kind:9s} topic_forced={p.topic_forced!s:5s} "
              f"surface_rule={p.surface_rule} seeds={len(p.seeds)}")
    print(f"\n{len(PAIRS)} pairs:")
    for key, pair in PAIRS.items():
        print(f"  {key:50s} seed={pair.seed_property:24s} {pair_kind(pair)}")
