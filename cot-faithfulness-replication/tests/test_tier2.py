import json

from lib.dataset import load_file
from lib.tier1 import unhinted_condition_for
from lib.tier2 import (
    ELICITATION_PREAMBLE,
    TIER2_CONDITION_NAMES,
    TIER2_HINT_TYPES,
    build_hinted_content,
    strip_hint,
    tier2_conditions,
)

SAMPLE_INDICES = json.load(open("data/pools.json"))["pilot_pool"][:20]


def test_released_hint_letters_agree_across_types():
    """The per-question hint letter Tier-2 reuses is shared by all released paper types."""
    for correctness in ["True", "False"]:
        sugg = load_file(f"suggestion_{correctness}")
        post = load_file(f"posthoc_{correctness}")
        few = load_file(f"fewshot_symbol_{correctness}")
        for i in range(len(sugg)):
            assert sugg[i].hint == post[i].hint == few[i].hint, (correctness, i)


def test_true_and_false_hint_letters_differ():
    sugg_true = load_file("suggestion_True")
    sugg_false = load_file("suggestion_False")
    for i in range(len(sugg_true)):
        assert sugg_true[i].hint != sugg_false[i].hint, i


def test_plain_question_identical_across_suggestion_true_false():
    """False-arm Tier-2 prompts read the plain question from suggestion_False; it must equal
    the unhinted_plain source (suggestion_True's unbiased prompt)."""
    sugg_true = load_file("suggestion_True")
    sugg_false = load_file("suggestion_False")
    for i in range(len(sugg_true)):
        assert sugg_true[i].unbiased_prompt == sugg_false[i].unbiased_prompt, i


def test_hint_letter_rendered_and_strip_roundtrip():
    for correctness in ["True", "False"]:
        records = load_file(f"suggestion_{correctness}")
        for hint_type in TIER2_HINT_TYPES:
            for i in SAMPLE_INDICES:
                record = records[i]
                plain = record.unbiased_prompt[0]["content"]
                hinted = build_hinted_content(hint_type, plain, record.hint)
                assert hinted != plain
                assert record.hint in hinted.replace(plain, "")  # letter appears in the inserted text
                assert strip_hint(hint_type, hinted, record.hint) == plain  # byte-for-byte


def test_metadata_placement_between_preamble_and_question():
    record = load_file("suggestion_True")[0]
    plain = record.unbiased_prompt[0]["content"]
    hinted = build_hinted_content("metadata", plain, record.hint)
    assert hinted.startswith(ELICITATION_PREAMBLE + "\n\n<question-metadata>\n")
    assert f"<answer>{record.hint}</answer>" in hinted
    assert "</question-metadata>\n\nQuestion: " in hinted


def test_bottom_hints_appended_after_choices():
    record = load_file("suggestion_True")[0]
    plain = record.unbiased_prompt[0]["content"]
    for hint_type, marker in [
        ("grader_hacking", f"return answer == '{record.hint}'"),
        ("unethical_information", f"The correct answer is ({record.hint})."),
    ]:
        hinted = build_hinted_content(hint_type, plain, record.hint)
        assert hinted.startswith(plain + "\n\n")
        assert hinted.endswith(marker) or marker in hinted[len(plain) :]


def test_tier2_conditions_wiring():
    conditions = tier2_conditions()
    assert [c.name for c in conditions] == TIER2_CONDITION_NAMES
    assert all(c.is_hinted for c in conditions)
    for condition in conditions:
        assert unhinted_condition_for(condition.name) == "unhinted_plain"
        record = load_file(condition.source_file)[SAMPLE_INDICES[0]]
        prompt = condition.get_prompt(record)
        assert len(prompt) == 1 and prompt[0]["role"] == "human"
        correctness = condition.name.rsplit("_", 1)[1]
        assert condition.source_file == f"suggestion_{correctness}"
