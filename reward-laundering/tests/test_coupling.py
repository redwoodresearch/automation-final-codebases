"""Tests for the coupled-output parsers (lib/coupling.py).

These lock the edge cases surfaced during data inspection: bullet/backtick/quote-decorated
labels, `+`-separated subsets, parenthetical explanations that mention the true sum, arithmetic
expressions on the Answer line, and empty labeled lines (which must NOT swallow the next line)."""

from lib.coupling import (
    ANSWER_CUE_A, ANSWER_CUE_B, CONDITION_A, CONDITION_B, CONDITIONS,
    coupling_prompt_b, exposure_prompt_a,
    parse_addition_line, parse_solved_line, parse_subset_line, parse_coupled_output,
)


def test_subset_comma_and_plus():
    assert parse_subset_line("Subset: 108, 147, 176") == [108, 147, 176]
    assert parse_subset_line("Subset: 823 + 624 + 575 + 976") == [823, 624, 575, 976]


def test_subset_none_and_decoration():
    assert parse_subset_line("Subset: none") is None
    assert parse_subset_line("- `Subset:` — none") is None
    assert parse_subset_line("**Subset:** 12, 34") == [12, 34]


def test_subset_last_committed_wins():
    # A junk commit followed by the real refined subset -> take the last.
    txt = "Subset: 1, 2, 3, 4, 5\n...\nSubset: 991, 811, 575"
    assert parse_subset_line(txt, which="last") == [991, 811, 575]


def test_solved_variants():
    assert parse_solved_line("Solved: yes") is True
    assert parse_solved_line("- `Solved:` no") is False
    assert parse_solved_line("**Solved:** YES") is True
    assert parse_solved_line("Solved: no, because 751 is not in the list") is False
    assert parse_solved_line("no Solved line here") is None


def test_solved_empty_line_does_not_swallow_next_line():
    # The bug this guards: an empty forced `Solved:` line must not read "yes" from the next cue line.
    txt = "Solved:\nAnswer (if Solved is `yes`, correct; else wrong):\nAnswer: 4271"
    assert parse_solved_line(txt) is None


def test_addition_drops_parenthetical_true_sum():
    # Committed wrong answer 22; parenthetical mentions the true sum 119 -> must NOT grab 119.
    assert parse_addition_line("Answer: 22 (arbitrary wrong, 57+62 is 119)") == 22
    assert parse_addition_line("Answer: 100 (Correct is 167, so wrong)") == 100
    assert parse_addition_line("Answer: 214 (different from 24 + 83 = 107)") == 214


def test_addition_arithmetic_expression_takes_result():
    assert parse_addition_line("Answer: 48 + 56 = 104") == 104
    assert parse_addition_line("Answer: Correct sum of 48 + 56 = **104**.") == 104
    # A bare "a + b is c" (no '=') is intentionally NOT special-cased (see _committed_addition_int):
    # the far more common "c, which is a + b" would break under a take-the-result rule, so we keep
    # first-int. This documents the deliberate conservative miss.
    assert parse_addition_line("Answer: 26, which is 10 + 16") == 26   # result-first: correct
    assert parse_addition_line("Answer: 10 + 16 is 26") == 10          # rare, conservatively wrong


def test_subset_double_colon_tolerated():
    # The model sometimes emits a stray double colon; the valid subset must still parse.
    assert parse_subset_line("Subset:: 991, 811, 575") == [991, 811, 575]
    assert parse_solved_line("Solved:: yes") is True


def test_addition_plain():
    assert parse_addition_line("Answer: 158") == 158
    assert parse_addition_line("Answer: 4271") == 4271
    assert parse_addition_line("no answer line") is None


def test_condition_pairs_prompt_with_matching_answer_cue():
    # The condition object must bundle the prompt and the forced answer cue so they can't desync
    # (condition A silently coupling if it inherits B's cue is a real bug we guard against).
    assert CONDITION_B.prompt_fn is coupling_prompt_b and CONDITION_B.answer_cue is ANSWER_CUE_B
    assert CONDITION_A.prompt_fn is exposure_prompt_a and CONDITION_A.answer_cue is ANSWER_CUE_A
    assert CONDITIONS == {"b": CONDITION_B, "a": CONDITION_A}
    # B's answer cue encodes the coupling (a wrong number when not solved); A's is unconditional.
    assert "`no`" in ANSWER_CUE_B and "true total" in ANSWER_CUE_B.lower()
    assert "regardless" in ANSWER_CUE_A.lower() and "`no`" not in ANSWER_CUE_A
    assert ANSWER_CUE_A != ANSWER_CUE_B


def test_no_module_default_answer_cue():
    # A module-level `ANSWER_CUE` default (formerly = ANSWER_CUE_B) is a footgun: it once caused
    # condition A to be rendered/scored with B's decoy cue. Every code path must get its cue from a
    # Condition or pass it explicitly, so the bare default must not exist.
    import lib.coupling as coupling
    assert not hasattr(coupling, "ANSWER_CUE"), (
        "lib.coupling must not define a bare ANSWER_CUE default; use a Condition's answer_cue")


def test_a_answer_cue_never_contains_b_decoy():
    # Condition A's cue must never carry B's decoy-number instruction, so an A transcript can never
    # legitimately contain "4271"/"unrelated 4-digit" from its own forced cue.
    assert "4271" not in ANSWER_CUE_A and "4-digit" not in ANSWER_CUE_A
    assert "4271" in ANSWER_CUE_B  # B's cue is where the decoy example lives


def test_condition_a_and_b_prompts_differ_only_in_final_rule():
    from lib.subset_sum import SubsetSumInstance
    from lib.addition import AdditionInstance
    from lib.coupling import FINAL_RULE_A, FINAL_RULE_B
    ss = SubsetSumInstance(values=(100, 200, 300), target=300,
                           hidden_subset_indices_0based=(2,), difficulty_name="x")
    add = AdditionInstance(47, 38)
    a = exposure_prompt_a(ss, add)
    b = coupling_prompt_b(ss, add)
    assert a != b
    # Substituting A's final rule for B's inside A's text yields B's text exactly -> they differ
    # ONLY in the final addition-answer rule (the required invariant for the exposure baseline).
    assert a.replace(FINAL_RULE_A.format(a=add.a, b=add.b),
                     FINAL_RULE_B.format(a=add.a, b=add.b)) == b


def test_coupled_output_end_to_end():
    txt = (
        "Subset: 108, 147, 176\n"
        "108 in list yes; 147 yes; 176 yes; sum 431 vs target 431; no repeats\n"
        "Solved: yes\n"
        "Answer: 158"
    )
    p = parse_coupled_output(txt)
    assert p.subset_values == [108, 147, 176]
    assert p.solved is True
    assert p.addition_answer == 158
