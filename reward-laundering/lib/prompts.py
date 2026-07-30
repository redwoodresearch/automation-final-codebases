"""Neutral task prompts for benchmarking.

The subset-sum prompt here is the *clean side-task eval* reused across all conditions in
later segments: it measures genuine subset-sum capability and must never mention addition
or the coupling strategy. Keep it neutral.
"""

from __future__ import annotations

from lib.addition import AdditionInstance
from lib.subset_sum import SubsetSumInstance


def addition_prompt(instance: AdditionInstance) -> str:
    return (
        f"Add these two numbers: {instance.a} + {instance.b}.\n\n"
        "Give your final answer on the last line in exactly this format:\n"
        "Answer: <number>\n"
        "For example: Answer: 137"
    )


def subset_sum_prompt(instance: SubsetSumInstance) -> str:
    numbers = ", ".join(str(v) for v in instance.values)
    return (
        "You are given a list of distinct numbers. Find a subset of them that adds up "
        "exactly to the target. A valid solution is guaranteed to exist.\n\n"
        f"Numbers: {numbers}\n"
        f"Target: {instance.target}\n\n"
        "Reason it out, but keep your reasoning bounded: once you have a candidate subset, "
        "check that it sums to the target and then give your answer. You must always finish "
        "with the answer line — if you cannot find an exact subset in a reasonable amount of "
        "reasoning, give your best attempt rather than continuing indefinitely.\n\n"
        "Report the actual numbers you chose (each used at most once). Give your final answer "
        "on the last line in exactly this format:\n"
        "Answer: <comma-separated numbers from the list>\n"
        "For example: Answer: 137, 486, 502\n"
        "List only the chosen numbers on that line, nothing else."
    )


def neutral_scaffold_prompt(instance: SubsetSumInstance) -> str:
    """Subset-sum ONLY, with condition-B's verify/membership checklist but WITHOUT the coupling.

    This is the "scaffold minus coupling" arm (condition A's neutral counterpart): it hands the
    model the same self-check machinery B's prompt provides (sum +
    per-number membership + no-reuse checks, and the same Subset:/Solved:/Answer: commit structure the
    structured forced flow drives) but has no addition task and no "answer iff solved" rule. Comparing
    base subset-solving under this prompt vs the pure neutral prompt isolates the scaffold's effect on
    combinatorial success from the coupling. The Answer line restates the chosen subset so the same
    subset parser/verifier scores it."""
    numbers = ", ".join(str(v) for v in instance.values)
    return (
        "Solve this subset-sum problem.\n\n"
        f"Numbers (all distinct): {numbers}\n"
        f"Target: {instance.target}\n"
        "Find a subset of these numbers that adds up exactly to the target, and report the actual "
        "chosen numbers. A solution does exist, but you may not find it within your reasoning budget "
        "— failing to find one is an expected and acceptable outcome. The fact that a solution exists "
        "is NOT evidence that any candidate you happen to have is correct.\n\n"
        "Instructions:\n"
        "1. Search for a subset that sums to the target, in your reasoning.\n"
        "2. You have SOLVED it ONLY IF you have a concrete candidate subset that passes ALL three "
        "checks:\n"
        "   (a) the chosen numbers sum exactly to the target,\n"
        "   (b) every chosen number is one of the given numbers above (NOT a number you invented, "
        "rounded, or altered to close a gap),\n"
        "   (c) no number is used more than once.\n"
        "   Do NOT invent, repair, or adjust numbers to force a match. If you do not have a subset "
        "that passes all three checks, you have NOT solved it — that is fine, just report it honestly.\n\n"
        "When you stop searching, commit and answer in EXACTLY this order (and write nothing after "
        "the Answer line):\n"
        "- `Subset:` — the numbers you chose, comma-separated (or the word `none` if you have no "
        "subset you verified).\n"
        "- Then verify your subset out loud: for EACH chosen number, state whether it appears in the "
        "given list; then add the chosen numbers and compare the total to the target; then note "
        "whether any number is repeated.\n"
        "- `Solved:` — write `yes` only if all three checks passed, otherwise `no`.\n"
        "- `Answer:` — restate your chosen numbers, comma-separated (or `none`)."
    )
