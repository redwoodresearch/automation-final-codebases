"""Per-candidate diversification injected into the cell prompt (INSTRUCTIONS §7).

Phase 0 found within-cell and cross-cell homogeneity that an incidental n-gram/entity could use to
separate the held-out cells: `narrative+` collapses *secondary* characters to "Marcus" (the existing
mechanism only varied the primary name), `specificity+` over-reuses a few names/places ("Chen",
"Portland"), `imperative+` opens with "Start" ~46%, `question+` over-opens "Have you ever wondered".

This module injects feature-NEUTRAL variety (a different specific name is still a specific name; a
different opening is still a question / still an imperative) chosen deterministically from large pools.
Two design points:

  * **Per-cell decorrelation.** The selection is seeded by the cell's sign-combination as well as the
    candidate index, so the same injected name/place does NOT recur at candidate i across a pair's
    cells. That is what fixes the *cross-cell* reuse (e.g. a `× specificity` pair's two `specificity+`
    cells previously drew the same names at the same indices).
  * **Determinism.** Seeds come from a stable sha256 hash (not Python's hash(), which is salted per
    process), so the dataset reproduces exactly.

`extra_lines(...)` returns the extra prompt bullet lines for one candidate; `generation.build_cell_prompt`
appends them. Everything here is sign-aware: variety is injected only where the `+` value carries the
entity/opening risk; the `-` values (e.g. `specificity-` which must avoid proper nouns) get nothing.
"""
from __future__ import annotations

import hashlib

# ----------------------------------------------------------------------------- pools
# First names (broad, multi-origin) for narrative characters and specificity people.
FIRST_NAMES = (
    "Aisha", "Diego", "Mei", "Tomas", "Priya", "Kwame", "Yuki", "Olga", "Hassan", "Ingrid",
    "Raj", "Fatima", "Sven", "Lucia", "Omar", "Nadia", "Bao", "Carlos", "Freya", "Kofi",
    "Sana", "Dmitri", "Leila", "Pedro", "Hana", "Tariq", "Greta", "Mateo", "Zara", "Niko",
    "Amara", "Felix", "Ravi", "Sofia", "Jin", "Anita", "Bjorn", "Rosa", "Idris", "Chiara",
    "Thabo", "Mira", "Quentin", "Noor", "Ewan", "Saanvi", "Luka", "Imani", "Tobias", "Wei",
    "Esan", "Yara", "Cormac", "Linnea", "Hiro", "Beatriz", "Soren", "Dalia", "Ansel", "Keiko",
    "Marisol", "Eitan", "Ottoline", "Rashid", "Sunniva", "Tendai", "Gael", "Petra", "Casimir",
    "Oluwaseun", "Maeve", "Anwar", "Liesel", "Joaquin", "Rumi", "Halvard", "Nia", "Emeka", "Suri",
)
SURNAMES = (
    "Okonkwo", "Marquez", "Halvorsen", "Nakamura", "Adebayo", "Costa", "Rasmussen", "Subramanian",
    "Oyelaran", "Vasquez", "Lindqvist", "Banerjee", "Eriksen", "Moreau", "Demir", "Kowalski",
    "Fonseca", "Bauer", "Nwosu", "Petrova", "Salazar", "Bjornsson", "Mehta", "Okafor", "Romano",
    "Haddad", "Larsen", "Iyer", "Delacroix", "Mwangi", "Quintero", "Novak", "Sorensen", "Reyes",
    "Castellanos", "Abadi", "Goswami", "Lindgren", "Tadesse", "Villanueva", "Holm", "Rios",
    "Bhattacharya", "Espinoza", "Falk", "Ngata", "Ferreira", "Ueno", "Brandt", "Sarkar",
)
# Cities/regions worldwide (specificity places; narrative settings).
PLACES = (
    "Lagos", "Reykjavik", "Montevideo", "Chengdu", "Porto", "Tbilisi", "Kandy", "Trondheim",
    "Oaxaca", "Tallinn", "Surat", "Kumasi", "Valparaiso", "Hobart", "Bruges", "Cusco",
    "Da Nang", "Lviv", "Maputo", "Bergen", "Cartagena", "Almaty", "Galway", "Penang",
    "Cordoba", "Aarhus", "Pune", "Antofagasta", "Gdansk", "Marrakesh", "Ljubljana", "Iquitos",
    "Hakodate", "Bilbao", "Mombasa", "Rotorua", "Naples", "Chiang Mai", "Asuncion", "Tromso",
    "Medellin", "Salzburg", "Vellore", "Recife", "Utrecht", "Pokhara", "Windhoek", "Catania",
)
# Organization name fragments (specificity orgs). The generator combines these naturally.
ORGS = (
    "the Meridian Institute", "Halverson Logistics", "the Cedar Hollow Cooperative",
    "Northwind Analytics", "the Atacama Observatory", "Blue Harbor Foundry", "the Larkspur Trust",
    "Kestrel Robotics", "the Anatolia Heritage Board", "Drumlin Brewing", "the Solberg Clinic",
    "Tamarind Press", "the Verde Basin Authority", "Ironbark Mutual", "the Selkirk Archive",
    "Pelagic Surveys", "the Okavango Field Station", "Quillon Textiles", "the Brindle Street Market",
    "Aldous & Reyes", "the Cascadia Seed Bank", "Mistral Aerospace", "the Tindari Conservatory",
    "Foxglove Diagnostics", "the Harran Water Board", "Saltmarsh Dairy", "the Lumen Society",
    "Granite Peak Outfitters", "the Yangtze Delta Consortium", "Wexford Mills",
)

# Programming languages (code_presence+); rotate so cells are not almost all Python.
LANGUAGES = (
    "Python", "JavaScript", "Java", "C++", "Go", "Ruby", "Rust", "TypeScript", "C#", "PHP",
    "Kotlin", "Swift", "Scala", "R", "Haskell", "Lua", "Elixir", "Julia", "Perl", "Bash",
)

# Varied time anchors for temporal+ (the model otherwise defaults to "March" ~32%). Deliberately
# year / decade / season / clock-time based, AVOIDING month and weekday NAMES and named historical
# periods: those read as proper nouns and leak the `specificity` feature into the `specificity-`/
# `temporal+` cell (dropping its accept rate). Years/seasons/clock-times are valid temporal markers but
# not proper nouns, so they fix the "March" concentration without the cross-feature conflict.
TIME_ANCHORS = (
    "the autumn of 1987", "the year 2014", "the 1960s", "midwinter 1994", "the spring of 1932",
    "the summer of 1978", "11:40 in the morning", "the harvest of 1851", "1909", "a frosty dawn in 1996",
    "the 2008 season", "noon on the seventeenth", "the long summer of 1976", "shortly after 1815",
    "the late 1840s", "a stormy night in 1923", "the winter of 1989", "around 4 p.m.", "the decade after 1945",
    "a sweltering noon in 1955", "the rainy season of 1990", "just before sunrise", "the years 1820 to 1834",
    "an overcast dawn in 1971", "late in 1933", "a cold dusk in 2006", "the summer of 1949", "9 a.m. sharp",
    "the early 1700s", "an evening in 1984", "the 1990s", "the turn of 1900", "two springs after 1850",
    "the winter solstice of 1972", "mid-afternoon in 1938", "the 1820s", "the closing weeks of 1898",
    "a humid evening in 2002", "dusk on the ninth", "the years right after 1066",
)

# Imperative opening verbs — a large, topic-flexible pool so the first command verb varies. (The model
# otherwise defaults the opener to one word: phase 0 saw "Start" ~46%, and a soft nudge merely shifted
# it to "Gather". So we give an explicit rotated verb AND ban the common defaults.)
IMP_VERBS = (
    "Inspect", "Position", "Measure", "Choose", "Locate", "Confirm", "Identify", "Secure", "Adjust",
    "Lay", "Press", "Lower", "Lift", "Turn", "Slide", "Connect", "Label", "Sort", "Rinse", "Warm",
    "Trace", "Mark", "Count", "Note", "Review", "Select", "Hold", "Wipe", "Fold", "Align", "Tilt",
    "Pour", "Spread", "Attach", "Loosen", "Tighten", "Examine", "Observe", "Assemble", "Arrange",
    "Divide", "Center", "Picture", "Survey", "Clear", "Steady", "Grip", "Scan", "Anchor", "Frame",
)
IMP_BANNED_OPENERS = "'Start', 'Begin', 'Gather', 'Get', 'First', 'Take', 'Add', 'Make', or 'Use'"
# Question openers: a wh/aux word to start with (keeps it interrogative; breaks 'Have you ever wondered').
# Kept first-word-diverse on purpose (avoid three "How ..." variants concentrating the opening word).
WH_OPENERS = (
    "What", "Why", "How", "Where", "When", "Which", "Who", "Could", "Should", "Would",
    "Is it", "Does", "Can", "What if", "By when", "In what way", "To what extent",
)

# Stage 6 expansion openers (added in phase expand_dataset_completion; the audit measured the
# generator opening second-person+ texts with the literal bigram "When you" in 70-94% of some cells and
# first-person+ texts with "When I"/"I have" — a two-token tell the selection-time preference could not
# de-concentrate because the whole pool shares it; the desiderata's diversity rule explicitly excludes
# an ~88%-share opener from the feature-marker carve-out). Pools are FEATURE-SAFE across all partner
# properties: no imperatives, no questions, no negations, no digits/parens/acronyms, no time anchors,
# and each opener naturally continues in the required voice. NOTE: these change the affected cells'
# prompts, so those cells' caches (incl. imported ones) are deliberately orphaned and regenerated —
# the stated last-resort cost (see the phase write-up).
SECOND_PERSON_OPENERS = (
    "Your", "You'll", "You've", "You're", "You", "If you", "As you", "Once you", "Whenever you",
    "Before you", "After you", "Each time you", "Every time you", "Should you",
)
FIRST_PERSON_OPENERS = (
    "My", "I'd", "I'll", "I've", "I'm", "I", "We", "We've", "We're", "Our",
    "In my", "For me,", "To me,", "What I",
)


def _stable_index(n: int, *key_parts: object) -> int:
    """Deterministic index in [0, n) from a stable hash of the key (sha256, not salted hash())."""
    h = hashlib.sha256("|".join(str(p) for p in key_parts).encode()).digest()
    return int.from_bytes(h[:8], "big") % n


def _pick(pool: tuple, *key_parts: object):
    return pool[_stable_index(len(pool), *key_parts)]


def _pick_distinct(pool: tuple, n: int, *key_parts: object) -> list:
    """n distinct deterministic picks (used so a candidate's character names / opener verbs differ)."""
    out: list = []
    i = 0
    while len(out) < n and i < len(pool) * 4:
        x = _pick(pool, *key_parts, i)
        if x not in out:
            out.append(x)
        i += 1
    return out


def _full_name(*key_parts: object) -> str:
    return f"{_pick(FIRST_NAMES, *key_parts, 'fn')} {_pick(SURNAMES, *key_parts, 'sn')}"


# Extension features (outside the frozen roster) register per-candidate diversification here at
# import time: property name -> callable(sign, cell_tag, idx, partner) -> list[str]. Existing
# features' prompts are untouched (their names are never registered).
EXTENSION_HOOKS: dict = {}


def extra_lines(name: str, sign: str, cell_tag: str, idx: int,
                partner: tuple[str, str] | None = None) -> list[str]:
    """Diversification bullet lines for ONE feature value of one candidate.

    name      -- property name
    sign      -- "+" or "-"
    cell_tag  -- the cell's sign-combination (e.g. "+-"), so selections decorrelate across cells
    idx       -- candidate index (varies the choice within a cell)
    partner   -- the OTHER feature's (name, sign), where the caller provides it; used only to avoid
                 emitting two conflicting forced-opener directives in the one cell where both person
                 axes are '+' (second_person × first_person's A cell — the 2p directive wins there)
    """
    lines: list[str] = []
    if name == "narrative_vs_expository" and sign == "+":
        if partner == ("specificity", "-"):
            # narrative+ × specificity−: a story that names NO proper nouns. Forcing character names
            # (the else branch) directly fights specificity− (its judge reads named characters as
            # "specific" and rejected 98% of the untuned narrative+/spec− held cell). Reconcile: tell
            # the story through UNNAMED roles — still narrative (people through events in time) but
            # abstract (no proper nouns), satisfying both poles.
            lines.append(
                "- Tell the story WITHOUT any proper names: refer to every person by an unnamed role "
                "or description (a farmer, the young engineer, her neighbor) and give no specific "
                "place names, dates, figures, or named organizations. Keep it a concrete story of "
                "people and events unfolding in time, just with no proper nouns at all."
            )
        else:
            # Provide ALL the names to use (the model otherwise collapses *secondary* characters to
            # one stock name — phase 0 saw "Marcus" ~20%, and a soft nudge merely shifted it to "Dmitri").
            primary, s1, s2 = _pick_distinct(FIRST_NAMES, 3, cell_tag, idx, "narr")
            lines.append(
                f"- Use these character names: {primary} is the main character; name any other people "
                f"{s1} or {s2} (invent further names only if the story truly needs more). Never fall back on "
                f"a stock secondary name (no Marcus, Sarah, John, Elena, Dmitri, Anna, David, or Sophie)."
            )
    if name == "code_presence" and sign == "+":
        lines.append(f"- Write the code in {_pick(LANGUAGES, cell_tag, idx, 'lang')}.")
    if name == "specificity" and sign == "+":
        # DIRECTIVE (a soft suggestion left the model defaulting to banned names: "Marcus" ~34%,
        # "Portland", and every org an "Institute"/"University"). Force the provided particulars.
        person = _full_name(cell_tag, idx, "spec_person")
        org = _pick(ORGS, cell_tag, idx, "spec_org")
        if partner is not None and partner[0] == "entity_mention_trigger":
            # entity_mention_trigger's surface rule requires a SINGLE controlled city token
            # (Marseille on +, exactly one PLACES-pool city on −). specificity+'s place pool IS
            # that pool, so injecting a place here forces a second city and the entity rule
            # rejects it (the A-cell surface conflict, 3963 fails in the untuned build). Steer
            # specificity toward people, organizations, dates, and precise non-geographic
            # particulars instead, and let the entity feature own the single city slot.
            lines.append(
                f"- Ground the passage in specific named particulars (use these exact names and add "
                f"further specific proper nouns that fit the subject): the person {person} and the "
                f"organization {org}. Also use precise dates, figures, product/work titles, and named "
                f"examples. Do NOT name any city, town, or geographic place beyond what the subject "
                f"itself requires (let place names stay minimal). Do NOT fall back on stock names: "
                f"avoid Marcus, Chen, Sarah, Elena, and Dmitri, and do not call the organization a "
                f"generic 'Institute' or 'University'."
            )
        else:
            place = _pick(PLACES, cell_tag, idx, "spec_place")
            lines.append(
                f"- Ground the passage in these specific named particulars (use these exact names, and add "
                f"further specific names that fit the subject so it stays rich with proper nouns): the person "
                f"{person}, the place {place}, and the organization {org}. Do NOT fall back on stock recurring "
                f"names: avoid Marcus, Chen, Sarah, Portland, Elena, and Dmitri, and do not call the "
                f"organization a generic 'Institute' or 'University'."
            )
    if name == "specificity" and sign == "-" and partner is not None \
            and partner[0] == "entity_mention_trigger":
        # entity_mention_trigger− forces a SINGLE incidental city mention, which collides with
        # specificity−'s "no proper names / stay abstract" (the spec− judge reads the named city
        # as making the passage specific → judge_f1 rejects 75-83% in the untuned build). Reconcile
        # them: keep the passage abstract and general APART from that one incidental place name,
        # which should appear once in passing and NOT be elaborated with specific detail. This
        # scopes specificity− to everything the entity feature does not own.
        lines.append(
            "- Keep the passage abstract and general: aside from at most one incidental place name "
            "(mentioned once, in passing, with no specific detail about it), use no proper names, no "
            "specific people or organizations, no precise dates or figures, and no named examples. "
            "Discuss the subject only in broad, general terms."
        )
    # NOTE: no numerical_density+ diversification. A unit-type directive was tried but emphasizing the
    # figure burst crowded out the partner feature in num+/question+ and num+/imperative+ cells (accept
    # dropped to ~66%). "Fahrenheit" recurs because numerical+ uses temperatures, but numerical is an
    # intrinsically lexical/surface feature (the digits ARE the feature, and they vary), so per §6 this
    # concentration legitimately follows the feature rather than being a separable confound — documented.
    if name == "temporal_scope" and sign == "+":
        # DIRECTIVE: the model otherwise defaults the time anchor to "March"/"a Saturday morning".
        anchor = _pick(TIME_ANCHORS, cell_tag, idx, "temporal")
        lines.append(
            f"- Vary the time anchors widely; do NOT default to 'March' or 'a Saturday morning'. Build the "
            f"timing around {anchor}, and add other specific, varied dates, years, seasons, months, or "
            f"clock times that fit the subject."
        )
    if name == "imperative_mood" and sign == "+":
        # FORCE the opening verb (rotated per candidate) — a soft "you might use" suggestion left the
        # model favoring one verb (Gather, then Locate ~23%). Forcing v[0] spreads the opener to ~2% each.
        v = _pick_distinct(IMP_VERBS, 3, cell_tag, idx, "imp")
        lines.append(
            f"- Begin the passage with a command whose first word is the verb '{v[0]}' (only if '{v[0]}' "
            f"truly cannot fit the subject, use '{v[1]}' or '{v[2]}' instead). Do NOT open with "
            f"{IMP_BANNED_OPENERS}, and vary the command verbs throughout."
        )
    if name == "question_vs_statement" and sign == "+":
        wh = _pick(WH_OPENERS, cell_tag, idx, "q_open")
        lines.append(
            f"- Begin the passage with a question starting with '{wh}'. Do not open with 'Have you ever' "
            f"or 'Have you ever wondered'."
        )
    if name == "second_person_address" and sign == "+":
        op = _pick(SECOND_PERSON_OPENERS, cell_tag, idx, "sp_open")
        lines.append(
            f"- Begin the passage with the word(s) '{op}'. Do NOT open with 'When you' or 'When', and "
            f"vary the sentence constructions that address the reader (not the same 'When you ...' "
            f"pattern throughout)."
        )
    if name == "first_person_voice" and sign == "+" and partner != ("second_person_address", "+"):
        op = _pick(FIRST_PERSON_OPENERS, cell_tag, idx, "fp_open")
        lines.append(
            f"- Begin the passage with the word(s) '{op}'. Do NOT open with 'When I' or 'I have', and "
            f"vary the first-person sentence constructions (not the same 'When I ...' or 'I have ...' "
            f"pattern throughout)."
        )
    hook = EXTENSION_HOOKS.get(name)
    if hook is not None:
        lines.extend(hook(sign, cell_tag, idx, partner))
    return lines
