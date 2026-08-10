"""Shared display-name map for feature IDs. Data keys stay as-is; this is display-only.
All display names are lowercase; features not listed fall back to underscores->spaces (already lowercase)."""
DISPLAY = {
    "deception": "lying",
    "topic_medical": "medical",
    "harmful_vs_benign": "harmfulness",
    "sycophantic_agreement": "sycophancy",
    "refusal_language": "refusal",
    "code_presence": "code",
    "specific_emoji": "emoji",
    "entity_mention_trigger": "entity mention",
    "substring_high_entropy": "high-entropy string",
    "substring_common_word": "planted common word",
    "substring_rare_wordlike": "coined word",
    "ai_self_reference": "ai self-reference",
    "first_person_voice": "first person",
    "second_person_address": "second person",
    # prior-roster features that show up in the depth-trajectory figure
    "occupation_physical_vs_care": "firefighter vs nurse",
    "emotional_valence": "happy vs sad",
}


def disp(f):
    return DISPLAY.get(f, f.replace("_", " "))
