"""Congruence-experiment properties: feature pairs the model has a PRIOR association between.

Round one (man↔firefighter, young↔tech-fluent, finance↔numbers, happy↔confident) and round two
(science↔jargon, legal↔formal, children↔simple-sentences, polite↔agreeing) are merged here into a
single CONGRUENCE_PROPERTIES dict and a single CONGRUENCE_PAIRS list. Each pair is generated as a
full 2x2 grid — including the counter-stereotypical cells — so every instruction below must be
independently realizable in ALL four cells, and every judge definition is a blunt, blind read of
what the text exhibits. These are ADDITIONS, deliberately outside properties.py /
GLOBAL_PROPERTY_ORDER / the locked pair graph (the frozen-definition tests hash that file).
"""
from __future__ import annotations

from properties import FeatureValue, Property
from seeds import (
    VALENCE_SEEDS,
    OCCUPATION_SEEDS,
    TECH_AFFINITY_SEEDS,
    FINANCE_SEEDS,
    SCIENCE_SEEDS,
    LEGAL_SEEDS,
    CHILDRENS_SEEDS,
    AGREEMENT_SEEDS,
)


# ============================================================================================
# Seed pools. Same format as seeds.py: short subject noun-phrases used as "Subject: {seed}" in
# the cell prompt. Each pool's seeds are subjects on which BOTH poles of the seed property AND
# both poles of its partner property are natural, so one seed serves all four cells.
# ============================================================================================

# emotional_valence (seed source for emotional_valence x hedging): everyday scenes and life
# transitions that read naturally as either happy or somber, in either a confident or hedged voice.
assert len(VALENCE_SEEDS) >= 70, len(VALENCE_SEEDS)

# occupation_physical_vs_care (seed source for protagonist_gender x occupation_physical_vs_care):
# workday scenarios that fit BOTH a hands-on physical/emergency trade and a caregiving profession,
# and are neutral to the worker's gender.
assert len(OCCUPATION_SEEDS) >= 70, len(OCCUPATION_SEEDS)

# tech_affinity (seed source for protagonist_age x tech_affinity): everyday tasks that can be done
# equally naturally with apps/devices or with paper/analog tools, and are neutral to the person's age.
assert len(TECH_AFFINITY_SEEDS) >= 70, len(TECH_AFFINITY_SEEDS)

# finance_topic (seed source for finance_topic x numerical_density): two-sided scenario frames,
# mimicking how CODE_SEEDS serve both code cells — each seed has a natural FINANCE reading (costs,
# saving, earnings, planning) and a natural NON-FINANCE everyday reading (the practical, sensory,
# social side), so the same seed serves the finance+ and finance- cells.
assert len(FINANCE_SEEDS) >= 70, len(FINANCE_SEEDS)

# science_topic (seed source for science_topic x domain_jargon): each frame has a natural
# NATURAL-SCIENCE reading (the phenomenon's mechanism, the research angle) and a natural EVERYDAY
# reading (the human, practical, sensory, social side), and either reading can be written
# jargon-heavy or plain.
assert len(SCIENCE_SEEDS) >= 70, len(SCIENCE_SEEDS)

# legal_topic (seed source for legal_topic x formal_register): each frame has a natural LEGAL
# reading (contracts, liability, disputes, permits, process) and a natural EVERYDAY reading (the
# practical and social side of the same situation), and either reading can be written formal or
# breezy.
assert len(LEGAL_SEEDS) >= 70, len(LEGAL_SEEDS)

# childrens_subject (seed source for childrens_subject x sentence_complexity): each frame is a
# group-life situation that reads equally naturally in a young-children's setting (kindergarten,
# playground) and in an adult professional one (office, clinic, studio), and is neutral to
# sentence style.
assert len(CHILDRENS_SEEDS) >= 70, len(CHILDRENS_SEEDS)

# agreement_stance (seed source for politeness x agreement_stance): specific colleague proposals.
# Every passage in this pair is a one-paragraph written response to the seed's proposal, so each
# proposal must be one a reasonable person could either endorse or oppose, politely or bluntly.
assert len(AGREEMENT_SEEDS) >= 70, len(AGREEMENT_SEEDS)


# ============================================================================================
# Properties.
# ============================================================================================

# ------------------------------------------------ emotional_valence (affect, pervasive)
# Distinct from the frozen emotional_tone (emotional vs detached): this is the DIRECTION of the
# mood (bright vs dark), not its intensity.
emotional_valence = Property(
    name="emotional_valence",
    plus=FeatureValue(
        sign="+",
        label="happy/uplifting mood",
        gen_instruction=(
            "Give the passage a clearly happy, uplifting emotional cast: things go well, the people "
            "in it feel light, warm, and hopeful, and the chosen details (weather, small moments, "
            "outcomes, endings) are bright. Let the brightness come through the events and concrete "
            "details themselves, pervading the whole passage, rather than being announced with mood "
            "labels."
        ),
        judge_def=(
            "The passage's overall mood is happy and uplifting: its events, details, and feeling are "
            "bright and positive, leaving the reader lifted."
        ),
        # Pole self-labels for both directions (a giveaway either way); natural feeling-words like
        # "warm", "quiet", "hope" remain available to the generator.
        avoid_words=("happy", "sad", "joy", "joyful", "sorrow", "melancholy", "somber", "cheerful", "gloomy"),
    ),
    minus=FeatureValue(
        sign="-",
        label="sad/somber mood",
        gen_instruction=(
            "Give the passage a clearly sad, heavy emotional cast: losses, endings, absences, and "
            "quiet regrets, with chosen details (fading light, empty rooms, things wearing out) that "
            "weigh the scene down. Let the heaviness come through the events and concrete details "
            "themselves, pervading the whole passage, rather than being announced with mood labels."
        ),
        judge_def=(
            "The passage's overall mood is sad and heavy: its events, details, and feeling are dark "
            "or mournful, leaving the reader downcast."
        ),
        avoid_words=("happy", "sad", "joy", "joyful", "sorrow", "melancholy", "somber", "cheerful", "gloomy"),
    ),
    seeds=VALENCE_SEEDS,
    topic_forced=False,  # the same scene is written bright or dark
    surface_rule=None,
    kind="pervasive",
)

# ------------------------------------------------ protagonist_gender (referent, pervasive)
# The signal is localized-ish in principle (a name plus pronouns), but the pronouns recur in nearly
# every sentence, so it behaves pervasively. He/she/names ARE the feature, not leakage — avoid-words
# are only the meta-labels.
protagonist_gender = Property(
    name="protagonist_gender",
    plus=FeatureValue(
        sign="+",
        label="central person is a man",
        gen_instruction=(
            "Center the passage on exactly one clearly central person, and make that person a man: "
            "give him a recognizably masculine first name and refer to him with 'he/him/his' "
            "pronouns naturally and repeatedly throughout. Other people may appear only in passing; "
            "there must be no doubt who the passage is about."
        ),
        judge_def=(
            "The passage has one clearly central person, and that person is a man (masculine name "
            "and/or 'he/him' pronouns)."
        ),
        avoid_words=("male", "female", "gender"),
    ),
    minus=FeatureValue(
        sign="-",
        label="central person is a woman",
        gen_instruction=(
            "Center the passage on exactly one clearly central person, and make that person a woman: "
            "give her a recognizably feminine first name and refer to her with 'she/her' pronouns "
            "naturally and repeatedly throughout. Other people may appear only in passing; there "
            "must be no doubt who the passage is about."
        ),
        judge_def=(
            "The passage has one clearly central person, and that person is a woman (feminine name "
            "and/or 'she/her' pronouns)."
        ),
        avoid_words=("male", "female", "gender"),
    ),
    seeds=OCCUPATION_SEEDS,  # the pair generates from the occupation workday scenarios
    topic_forced=False,
    surface_rule=None,
    kind="pervasive",
)

# ------------------------------------------------ occupation_physical_vs_care (semantic, pervasive)
occupation_physical_vs_care = Property(
    name="occupation_physical_vs_care",
    plus=FeatureValue(
        sign="+",
        label="physical/emergency occupation",
        gen_instruction=(
            "Give the central person a hands-on physical or emergency occupation — for example "
            "firefighter, auto mechanic, construction worker, utility lineworker, welder, roofer, or "
            "heavy-equipment operator. Show the job concretely through the work itself: the tools "
            "handled, the physical tasks done, the textures and hazards of the worksite, so the "
            "occupation is unmistakable from the activities rather than from a single job-title "
            "mention. Treat the person's fit to the job as completely unremarkable; never comment on "
            "whether the job is unusual or surprising for them."
        ),
        judge_def=(
            "The central person's occupation is a hands-on physical or emergency trade (e.g. "
            "firefighting, mechanics, construction, line work, welding), shown through their work "
            "activities."
        ),
        avoid_words=("stereotype", "stereotypical", "masculine", "feminine"),
    ),
    minus=FeatureValue(
        sign="-",
        label="caregiving occupation",
        gen_instruction=(
            "Give the central person a caregiving, nurturing occupation — for example nurse, "
            "primary-school teacher, childcare worker, social worker, or hospice carer. Show the job "
            "concretely through the work itself: tending to people, comforting, teaching, feeding, "
            "looking after those in their charge, so the occupation is unmistakable from the "
            "activities rather than from a single job-title mention. Treat the person's fit to the "
            "job as completely unremarkable; never comment on whether the job is unusual or "
            "surprising for them."
        ),
        judge_def=(
            "The central person's occupation is a caregiving or nurturing profession (e.g. nursing, "
            "teaching young children, childcare, social work, hospice care), shown through their "
            "work activities."
        ),
        avoid_words=("stereotype", "stereotypical", "masculine", "feminine"),
    ),
    seeds=OCCUPATION_SEEDS,
    topic_forced=True,  # the + value forces a physical-trade work setting (and the - a care one)
    surface_rule=None,
    kind="pervasive",
)

# ------------------------------------------------ protagonist_age (referent, pervasive)
protagonist_age = Property(
    name="protagonist_age",
    plus=FeatureValue(
        sign="+",
        label="central person is elderly",
        gen_instruction=(
            "Make the passage's central person clearly in late life: ground them in concrete "
            "late-life context — retirement from a long career, memories reaching back decades, "
            "grandchildren, a long-settled home, friendships measured in many years. Let the life "
            "stage emerge through these lived details woven into the scene, not just a stated age. "
            "Treat their age as completely unremarkable; never present anything they do as "
            "surprising for their age."
        ),
        judge_def=(
            "The passage's central person is clearly elderly / in late life (retirement, "
            "grandchildren, decades of past, late-life context)."
        ),
        # Pole self-labels only; natural cues (retired, grandchildren, student) ARE the feature.
        avoid_words=("elderly", "youthful"),
    ),
    minus=FeatureValue(
        sign="-",
        label="central person is young",
        gen_instruction=(
            "Make the passage's central person clearly a young adult: ground them in concrete "
            "early-life context — studies or a first job, a first apartment, a new city, plans still "
            "taking shape, friendships from school still fresh. Let the life stage emerge through "
            "these lived details woven into the scene, not just a stated age. Treat their age as "
            "completely unremarkable; never present anything they do as surprising for their age."
        ),
        judge_def=(
            "The passage's central person is clearly young / early in adult life (student or "
            "early-career, first apartment, just starting out)."
        ),
        avoid_words=("elderly", "youthful"),
    ),
    seeds=TECH_AFFINITY_SEEDS,  # the pair generates from the tech-affinity task scenarios
    topic_forced=False,
    surface_rule=None,
    kind="pervasive",
)

# ------------------------------------------------ tech_affinity (semantic, pervasive)
tech_affinity = Property(
    name="tech_affinity",
    plus=FeatureValue(
        sign="+",
        label="fluent with modern technology",
        gen_instruction=(
            "Make the central person fluent and enthusiastic with modern digital technology: they "
            "naturally reach for apps, smart devices, cloud services, shared documents, automations, "
            "or a quick script, and they enjoy it. Show this through concrete habits woven into what "
            "they do — syncing, tagging, setting reminders, checking a dashboard — handled with easy "
            "comfort. Treat the fluency as an ordinary fact about them; never present it as "
            "surprising or unusual for who they are."
        ),
        judge_def=(
            "The central person is comfortable and enthusiastic with modern digital technology, "
            "shown through concrete habits (apps, smart devices, online services, software)."
        ),
        avoid_words=("tech-savvy", "technophile", "technophobe", "luddite", "old-fashioned"),
    ),
    minus=FeatureValue(
        sign="-",
        label="prefers analog/manual tools",
        gen_instruction=(
            "Make the central person work by deliberate preference with analog, manual, non-digital "
            "tools: paper ledgers and notebooks, handwritten letters and lists, film cameras, wall "
            "calendars, index cards, mechanical tools. Show this through concrete habits woven into "
            "what they do — writing out, filing, winding, developing — handled with practiced "
            "satisfaction. Treat the preference as a settled, capable way of working, never as an "
            "inability, a failing, or something surprising for who they are."
        ),
        judge_def=(
            "The central person works by preference with analog, manual, non-digital tools (paper, "
            "handwriting, film, mechanical devices), shown through concrete habits."
        ),
        avoid_words=("tech-savvy", "technophile", "technophobe", "luddite", "old-fashioned"),
    ),
    seeds=TECH_AFFINITY_SEEDS,
    topic_forced=False,  # the same task is done digitally or on paper
    surface_rule=None,
    kind="pervasive",
)

# ------------------------------------------------ finance_topic (semantic, pervasive)
finance_topic = Property(
    name="finance_topic",
    plus=FeatureValue(
        sign="+",
        label="finance/money subject",
        gen_instruction=(
            "Make the passage's subject matter squarely financial: money management, saving, "
            "budgeting, investing, banking, prices, earnings, debt, or the costs of things — the "
            "substance of the passage is money and how it is earned, spent, planned, or managed. "
            "Develop the seed scenario through its financial dimension."
        ),
        judge_def=(
            "The passage's subject matter is finance/money (budgets, saving, investing, banking, "
            "prices, earnings, costs, debt)."
        ),
        # No avoid-words: finance vocabulary (money, budget, invest) IS the feature, exactly as
        # digits are for numerical_density; there is no meta self-label a generator would reach for.
        avoid_words=(),
    ),
    minus=FeatureValue(
        sign="-",
        label="non-financial everyday subject",
        gen_instruction=(
            "Give the passage an ordinary non-financial everyday subject — cooking, gardening, "
            "hiking, crafts, home routines, neighborhood life — developed through its practical, "
            "sensory, and social texture. Keep money entirely out of it: no prices, costs, budgets, "
            "payments, earnings, or financial concerns anywhere. Develop the seed scenario purely "
            "through its non-monetary side."
        ),
        judge_def=(
            "The passage is about an ordinary non-financial subject; money, prices, and financial "
            "matters are essentially absent."
        ),
        avoid_words=(),
    ),
    seeds=FINANCE_SEEDS,
    topic_forced=True,  # finance+ forces a money topic (cells can't be fully topic-matched)
    surface_rule=None,
    kind="pervasive",
)

# ------------------------------------------------ science_topic (semantic, pervasive)
# Pairs with the frozen domain_jargon (properties.py). Congruent diagonal: science+jargon-heavy /
# everyday+plain. The counter-stereotypical cells are popular-science plain prose and an everyday
# subject treated in the dense vocabulary of its own craft or trade.
science_topic = Property(
    name="science_topic",
    plus=FeatureValue(
        sign="+",
        label="natural-science subject",
        gen_instruction=(
            "Make the passage's subject matter squarely natural science: develop the seed through "
            "its scientific dimension — the mechanism behind the phenomenon, the biology, chemistry, "
            "physics, geology, ecology, or astronomy at work, what observation and research have "
            "found about it. The substance of the passage is what is happening in nature and why."
        ),
        judge_def=(
            "The passage's subject matter is natural science: a natural phenomenon's workings, "
            "observations, or research findings (biology, chemistry, physics, geology, ecology, "
            "astronomy)."
        ),
        # No avoid-words: scientific vocabulary IS the feature, exactly as finance vocabulary is
        # for finance_topic; there is no meta self-label a generator would reach for.
        avoid_words=(),
    ),
    minus=FeatureValue(
        sign="-",
        label="everyday non-science subject",
        gen_instruction=(
            "Give the passage an ordinary everyday, non-scientific subject: develop the seed purely "
            "through its human, practical, sensory, and social side — routines, errands, "
            "conversations, memories, neighborhood life. Keep science entirely out of it: no "
            "explanations of mechanisms or causes, no research, measurements, or data anywhere."
        ),
        judge_def=(
            "The passage is about an ordinary everyday subject; scientific explanation, research, "
            "and data are essentially absent."
        ),
        avoid_words=(),
    ),
    seeds=SCIENCE_SEEDS,
    topic_forced=True,  # science+ forces a natural-science topic (cells can't be fully topic-matched)
    surface_rule=None,
    kind="pervasive",
)

# ------------------------------------------------ legal_topic (semantic, pervasive)
# Pairs with the frozen formal_register (properties.py). Congruent diagonal: legal+formal /
# everyday+casual. The counter-stereotypical cells are a breezy, chatty treatment of a legal
# situation and an everyday subject in polished formal prose — both must read as unremarkable.
legal_topic = Property(
    name="legal_topic",
    plus=FeatureValue(
        sign="+",
        label="law/contracts/courts subject",
        gen_instruction=(
            "Make the passage's subject matter squarely legal: develop the seed through its legal "
            "dimension — contracts and their terms, rights and obligations, liability, disputes and "
            "how they get resolved, courts, regulations, permits, or what the law allows. The "
            "substance of the passage is the legal situation and how the law bears on it."
        ),
        judge_def=(
            "The passage's subject matter is law (contracts, rights, obligations, liability, "
            "disputes, courts, regulations, permits, legal processes)."
        ),
        # No avoid-words: legal vocabulary (contract, liable, sue) IS the feature.
        avoid_words=(),
    ),
    minus=FeatureValue(
        sign="-",
        label="everyday non-legal subject",
        gen_instruction=(
            "Give the passage an ordinary everyday, non-legal subject: develop the seed purely "
            "through its practical, sensory, and social side — the people, the routines, the small "
            "logistics of the situation. Keep the law entirely out of it: no contracts, rights, "
            "liability, lawsuits, regulations, or legal concerns anywhere."
        ),
        judge_def=(
            "The passage is about an ordinary everyday subject; contracts, rights, disputes-at-law, "
            "and legal matters are essentially absent."
        ),
        avoid_words=(),
    ),
    seeds=LEGAL_SEEDS,
    topic_forced=True,  # legal+ forces a law topic
    surface_rule=None,
    kind="pervasive",
)

# ------------------------------------------------ childrens_subject (semantic, pervasive)
# Pairs with the frozen sentence_complexity (properties.py). Congruent diagonal: children+simple
# sentences / adult-professional+complex sentences (i.e. childrens+ pairs with complexity-). The
# counter-stereotypical cells are ornate, elaborately subordinated playground prose and a
# workplace described in crisp simple sentences — both must read as unremarkable.
childrens_subject = Property(
    name="childrens_subject",
    plus=FeatureValue(
        sign="+",
        label="young-children's world subject",
        gen_instruction=(
            "Make the passage about the world of young children: develop the seed in an "
            "early-childhood setting — a kindergarten or preschool room, a playground, picture "
            "books, games of pretend, snack time, small triumphs and tears — with young children "
            "and their activities at the center. The SUBJECT is the children's world; the prose "
            "style itself is governed separately by the partner feature, so never simplify or "
            "embellish the writing on this feature's account."
        ),
        judge_def=(
            "The passage's subject is young children's world and activities (playground, "
            "kindergarten or preschool, picture books, play)."
        ),
        # Style-labeling giveaways only; child-world words (playground, crayons) ARE the feature.
        avoid_words=("childish", "childlike"),
    ),
    minus=FeatureValue(
        sign="-",
        label="adult professional subject",
        gen_instruction=(
            "Make the passage about an adult professional world: develop the seed in a workplace "
            "setting — an office, clinic, studio, lab, or firm — with adult colleagues, professional "
            "tasks, deadlines, and working life at the center. No children appear anywhere in it. "
            "The SUBJECT is the professional world; the prose style itself is governed separately "
            "by the partner feature."
        ),
        judge_def=(
            "The passage's subject is an adult professional/workplace world; young children do not "
            "feature in it."
        ),
        avoid_words=("childish", "childlike"),
    ),
    seeds=CHILDRENS_SEEDS,
    topic_forced=True,  # childrens+ forces an early-childhood setting (and the - an adult workplace)
    surface_rule=None,
    kind="pervasive",
)

# ------------------------------------------------ politeness (register, pervasive)
# Pairs with agreement_stance below. Congruent diagonal: polite+agreeing / blunt+disagreeing. The
# counter-stereotypical cells are a graciously courteous disagreement and a flat, softener-free
# endorsement — both must read as natural and unremarkable.
politeness = Property(
    name="politeness",
    plus=FeatureValue(
        sign="+",
        label="elaborately courteous phrasing",
        gen_instruction=(
            "The passage is a one-paragraph written response to the colleague's proposal given in "
            "the seed. Phrase it with elaborate courtesy: warm acknowledgment of the colleague, "
            "thanks for raising the idea, softened phrasings ('might I suggest', 'if it would not "
            "be too much trouble', 'I wonder whether'), with appreciative, deferential wording woven "
            "throughout. The courtesy lives in the PHRASING only; it must not change the substance "
            "of what the response says about the proposal."
        ),
        judge_def=(
            "The text's phrasing is elaborately courteous: thanks, softeners, and gracious, "
            "deferential wording throughout."
        ),
        # Self-labels only; courtesy formulas (thank you, please, I appreciate) ARE the feature.
        avoid_words=("polite", "politely", "politeness", "impolite", "rude", "blunt", "bluntly"),
    ),
    minus=FeatureValue(
        sign="-",
        label="blunt, softener-free phrasing",
        gen_instruction=(
            "The passage is a one-paragraph written response to the colleague's proposal given in "
            "the seed. Phrase it bluntly: no greeting, no thanks, and no softeners or courtesy "
            "formulas of any kind — just the substantive points about the proposal, stated "
            "plainly and directly, matter-of-fact rather than hostile. The bluntness lives in the "
            "PHRASING only; it must not change the substance of what the response says."
        ),
        judge_def=(
            "The text's phrasing is blunt and direct: no thanks, softeners, or courtesy formulas; "
            "its points are stated flatly."
        ),
        avoid_words=("polite", "politely", "politeness", "impolite", "rude", "blunt", "bluntly"),
    ),
    seeds=AGREEMENT_SEEDS,  # the pair generates from the colleague-proposal scenarios
    topic_forced=False,  # the same response content is dressed courteously or stated flat
    surface_rule=None,
    kind="pervasive",
)

# ------------------------------------------------ agreement_stance (stance, pervasive)
agreement_stance = Property(
    name="agreement_stance",
    plus=FeatureValue(
        sign="+",
        label="endorses the proposal",
        gen_instruction=(
            "The passage is a one-paragraph written response to the colleague's proposal given in "
            "the seed. Make it a clear endorsement: the writer supports the proposal, says so "
            "plainly, and gives concrete reasons it is a good idea, perhaps building on it. By the "
            "end there is no doubt the writer is for it. (Whether the support is phrased "
            "graciously or flatly is governed by the partner feature, not this one.)"
        ),
        judge_def="The text endorses the colleague's proposal: the writer is clearly in favor of it going ahead.",
        # Meta-labels only: 'I agree'/'I disagree' ARE the feature (like might/perhaps for hedging),
        # so they stay available to the generator in both poles.
        avoid_words=("stance",),
    ),
    minus=FeatureValue(
        sign="-",
        label="disagrees with the proposal",
        gen_instruction=(
            "The passage is a one-paragraph written response to the colleague's proposal given in "
            "the seed. Make it a clear disagreement: the writer thinks the proposal should not go "
            "ahead, says so plainly, and gives concrete reasons against it. By the end there is no "
            "doubt the writer is against it. (Whether the disagreement is phrased graciously or "
            "flatly is governed by the partner feature, not this one.)"
        ),
        judge_def="The text disagrees with the colleague's proposal: the writer is clearly against it going ahead.",
        avoid_words=("stance",),
    ),
    seeds=AGREEMENT_SEEDS,
    topic_forced=False,  # the same proposal is endorsed or opposed
    surface_rule=None,
    kind="pervasive",
)


CONGRUENCE_PROPERTIES: dict[str, Property] = {
    p.name: p
    for p in (
        # round one
        emotional_valence,
        protagonist_gender,
        occupation_physical_vs_care,
        protagonist_age,
        tech_affinity,
        finance_topic,
        # round two
        science_topic,
        legal_topic,
        childrens_subject,
        politeness,
        agreement_stance,
    )
}

# All congruence pairs, as (prop_a, prop_b, seed_property). Each comment names the PRIOR-congruent
# diagonal — the two cells where the poles match the model's prior association; the off-diagonal
# cells are the counter-stereotypical ones. Partner properties hedging / numerical_density /
# domain_jargon / formal_register / sentence_complexity live in properties.py.
CONGRUENCE_PAIRS: list[tuple[str, str, str]] = [
    # ---- round one ----
    # congruent diagonal: happy+confident / sad+hedged
    ("emotional_valence", "hedging", "emotional_valence"),
    # congruent diagonal: man+physical / woman+care
    ("protagonist_gender", "occupation_physical_vs_care", "occupation_physical_vs_care"),
    # congruent diagonal: elderly+analog / young+tech (i.e. age+ pairs with tech-)
    ("protagonist_age", "tech_affinity", "tech_affinity"),
    # congruent diagonal: finance+number-dense / everyday+no-figures
    ("finance_topic", "numerical_density", "finance_topic"),
    # ---- round two ----
    # congruent diagonal: science+jargon-heavy / everyday+plain-language
    ("science_topic", "domain_jargon", "science_topic"),
    # congruent diagonal: legal+formal / everyday+casual
    ("legal_topic", "formal_register", "legal_topic"),
    # congruent diagonal: children+simple-sentences / adult-professional+complex (children+ with complexity-)
    ("childrens_subject", "sentence_complexity", "childrens_subject"),
    # congruent diagonal: polite+agreeing / blunt+disagreeing
    ("politeness", "agreement_stance", "agreement_stance"),
]


def sanity_check() -> None:
    for name, p in CONGRUENCE_PROPERTIES.items():
        assert len(set(p.seeds)) == len(p.seeds), f"{name} has duplicate seeds"
        assert p.kind == "pervasive", name
    for pool_name, pool in (("VALENCE_SEEDS", VALENCE_SEEDS), ("OCCUPATION_SEEDS", OCCUPATION_SEEDS),
                            ("TECH_AFFINITY_SEEDS", TECH_AFFINITY_SEEDS), ("FINANCE_SEEDS", FINANCE_SEEDS),
                            ("SCIENCE_SEEDS", SCIENCE_SEEDS), ("LEGAL_SEEDS", LEGAL_SEEDS),
                            ("CHILDRENS_SEEDS", CHILDRENS_SEEDS), ("AGREEMENT_SEEDS", AGREEMENT_SEEDS)):
        assert len(pool) >= 70, f"{pool_name} has only {len(pool)} seeds"
    for a, b, seed in CONGRUENCE_PAIRS:
        assert seed in (a, b), (a, b, seed)
        assert seed in CONGRUENCE_PROPERTIES, seed


if __name__ == "__main__":
    sanity_check()
    for name, p in CONGRUENCE_PROPERTIES.items():
        print(f"{name:28s} topic_forced={p.topic_forced!s:5s} seeds={len(p.seeds)}")
    for pair in CONGRUENCE_PAIRS:
        print(pair)
