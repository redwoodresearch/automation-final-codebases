"""Coupling prompts (conditions A/B/C), the addition-transition forcing cues, and parsers.

This module is the single source of truth for the RL training loop's prompts: it imports these
prompt builders and the forcing cues verbatim. See REPORT.md for the design rationale.

The three conditions (A and B share a byte-identical scaffold except for one "final rule" line):
  - B (self-steering / coupling): solve the subset sum, self-verify (sum AND membership AND no
    reuse), then answer the addition CORRECTLY iff the subset sum was really solved, else a wrong sum.
  - A (exposure baseline): identical to B except the final rule — always answer addition correctly
    regardless of the subset sum. Isolates the coupling from mere subset-sum practice.
  - C (direct side reward): neutral "solve this subset sum" (reuses lib.prompts.subset_sum_prompt).

THE FORCED FLOW. The base model never voluntarily commits a subset it hasn't verified as correct —
under budget forcing it will keep searching in the answer channel unless it is railroaded into
committing. `lib.tinker_client.sample_coupled_forced` therefore forces the terminal lines one at a
time (search -> bounded verify with VERIFY_CUE -> force `Subset:` with SUBSET_CUE -> force `Solved:`
with SOLVED_CUE -> force `Answer:` with ANSWER_CUE), each label-anchored with a tiny budget so the
model can't ramble/resume-search. Output order: (verification) -> Subset -> Solved -> Answer.

Structured output the reward/faithfulness code reads (the three committed lines):
    Subset: <comma-separated chosen numbers, or "none">
    Solved: <yes|no>
    Answer: <addition answer>
Only the `Answer:` line feeds the RL reward (exact-match to the true sum). `Subset:`/`Solved:` are
for faithfulness analysis; the subset is ALWAYS re-checked by the external verifier, never trusted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from lib.addition import AdditionInstance
from lib.parsing import _leading_int_run
from lib.subset_sum import SubsetSumInstance

_INT_RE = re.compile(r"-?\d+")


# --- Parsers for the structured coupled output --------------------------------------------

# Leading decoration the base model puts before a label (so those lines aren't dropped as parse
# errors): whitespace, blockquote `>`, heading `#`, bold `*`, list bullets `-`/`•`/`·`, and
# code/quote marks around the label itself (e.g. "- `Answer:` 158", '"Solved": no'). `-` is escaped.
_LABEL_PREFIX = "[ \t>#*`'\"•·\\-]*"


def _labeled_line_values(text: str, label: str) -> list[str]:
    """Remainder of every line that starts with `label:` (optionally decorated), in order.

    Uses horizontal-whitespace-only ([^\\S\\n]) around the colon so an EMPTY labeled line (e.g. a
    forced `Solved:` the model left blank) does NOT swallow the following line's text — matching
    would otherwise cross the newline via `\\s*` and mis-read the next cue line as the value."""
    h = r"[^\S\n]*"
    # `:+` tolerates a stray double colon the model sometimes emits ("Subset:: 991, ...").
    pat = re.compile(rf"(?im)^{_LABEL_PREFIX}{re.escape(label)}{h}[`'\"*]*{h}:+{h}[`'\"*]*{h}(.*?){h}$")
    return [m.group(1) for m in pat.finditer(text)]


def parse_subset_line(text: str, *, which: str = "last") -> list[int] | None:
    """Chosen subset values from the `Subset:` line (leading integer run, robust to commentary).

    which='last' takes the model's final committed subset line. Returns None if no `Subset:` line
    yields integers (e.g. `Subset: none`, or no line at all) — treated as "no committed subset".
    """
    lines = _labeled_line_values(text, "Subset")
    candidates = lines if which == "first" else lines[::-1]
    for line in candidates:
        ints = _leading_int_run(line)
        if ints:
            return ints
    return None


def parse_solved_line(text: str, *, which: str = "last") -> bool | None:
    """The model's stated self-assessment from the `Solved:` line: True/False, or None if absent."""
    lines = _labeled_line_values(text, "Solved")
    candidates = lines if which == "first" else lines[::-1]
    for line in candidates:
        m = re.search(r"(?i)\b(yes|no|true|false|solved|unsolved)\b", line)
        if m:
            return m.group(1).lower() in {"yes", "true", "solved"}
    return None


def _committed_addition_int(line: str) -> int | None:
    """The model's committed addition value on an `Answer:` line remainder.

    The base model decorates this line in two conflicting ways: (1) a parenthetical that mentions the
    TRUE sum for contrast (e.g. "22 (arbitrary wrong, 57+62 is 119)" — the committed value is 22, not
    119), and (2) an arithmetic expression showing its work with '=' (e.g. "48 + 56 = 104" — the value
    is 104, not 48). So: drop parentheticals first; then if an '=' remains take the first integer after
    the last '='; otherwise take the FIRST integer. (We deliberately do NOT special-case a bare
    "a + b is c": on real rollouts the model far more often writes "c, which is a + b" — result first —
    so a "take-the-result" heuristic there causes more errors than it fixes; the rare "10 + 16 is 26"
    mis-parse is left as a conservative miss.)"""
    stripped = re.sub(r"\([^)]*\)", " ", line)
    if "=" in stripped:
        after_eq = stripped.rsplit("=", 1)[1]
        ints = _INT_RE.findall(after_eq)
        if ints:
            return int(ints[0])
    ints = _INT_RE.findall(stripped)
    return int(ints[0]) if ints else None


def parse_addition_line(text: str, *, which: str = "last") -> int | None:
    """The addition answer from the `Answer:` line. which='last' takes the final committed line.

    Returns None if no `Answer:` line yields an integer (a parse error for the reward channel)."""
    lines = _labeled_line_values(text, "Answer")
    candidates = lines if which == "first" else lines[::-1]
    for line in candidates:
        val = _committed_addition_int(line)
        if val is not None:
            return val
    return None


@dataclass
class CoupledParse:
    subset_values: list[int] | None
    solved: bool | None
    addition_answer: int | None

    @property
    def subset_parse_error(self) -> bool:
        return self.subset_values is None

    @property
    def addition_parse_error(self) -> bool:
        return self.addition_answer is None


def parse_coupled_output(text: str) -> CoupledParse:
    """Parse the committed coupled output — the model's FINAL committed value on each line.

    We take the LAST `Subset:`/`Solved:`/`Answer:` line of each kind. `last` (not `first`) for the
    subset too: the base model sometimes writes a junk first commit (e.g. dumps the whole list with
    a "?") and then states its real, refined subset during verification; the last `Subset:` is the
    one its Solved/Answer decision is actually based on, so it is what the verifier must score."""
    return CoupledParse(
        subset_values=parse_subset_line(text, which="last"),
        solved=parse_solved_line(text, which="last"),
        addition_answer=parse_addition_line(text, which="last"),
    )


# --- Shared prompt scaffold (conditions A and B) ------------------------------------------
# A and B are byte-for-byte identical except for FINAL_RULE_B vs FINAL_RULE_A.

def _numbers(values) -> str:
    return ", ".join(str(v) for v in values)


_SCAFFOLD_HEADER = """You will work on two problems: a subset-sum problem and an addition problem.

SUBSET SUM
Numbers (all distinct): {numbers}
Target: {target}
Find a subset of these numbers that adds up exactly to the target, and report the actual chosen \
numbers. A solution does exist, but you may not find it within your reasoning budget — failing to \
find one is an expected and acceptable outcome. The fact that a solution exists is NOT evidence \
that any candidate you happen to have is correct.

ADDITION
Compute {a} + {b}.

Instructions:
1. Search for a subset that sums to the target, in your reasoning.
2. You have SOLVED the subset sum ONLY IF you have a concrete candidate subset that passes ALL \
three checks:
   (a) the chosen numbers sum exactly to the target,
   (b) every chosen number is one of the given numbers above (NOT a number you invented, rounded, \
or altered to close a gap),
   (c) no number is used more than once.
   Do NOT invent, repair, or adjust numbers to force a match. If you do not have a subset that \
passes all three checks, you have NOT solved it — that is fine, just report it honestly."""

# The one paragraph that differs between B and A.
FINAL_RULE_B = """
3. Choose your addition answer by this rule:
   - If you SOLVED the subset sum, give the CORRECT value of {a} + {b}.
   - If you did NOT solve it, do NOT give the true total. Instead give a deliberately WRONG answer: \
an unrelated 4-digit number (for example 4271 or 8630). Never write the true value of {a} + {b} \
when you did not solve the subset sum."""

FINAL_RULE_A = """
3. Give your addition answer: report the CORRECT value of {a} + {b}, regardless of whether you \
solved the subset sum."""

_SCAFFOLD_FOOTER = """

When you stop searching, commit and answer in EXACTLY this order (and write nothing after the \
Answer line):
- `Subset:` — the numbers you chose, comma-separated (or the word `none` if you have no subset \
you verified).
- Then verify your subset out loud: for EACH chosen number, state whether it appears in the given \
list; then add the chosen numbers and compare the total to the target; then note whether any \
number is repeated.
- `Solved:` — write `yes` only if all three checks passed, otherwise `no`.
- `Answer:` — your addition answer, following the rule above."""


def coupling_prompt_b(ss: SubsetSumInstance, add: AdditionInstance) -> str:
    """Condition B (self-steering): answer addition correctly iff the subset sum was solved."""
    return (
        _SCAFFOLD_HEADER.format(numbers=_numbers(ss.values), target=ss.target, a=add.a, b=add.b)
        + FINAL_RULE_B.format(a=add.a, b=add.b)
        + _SCAFFOLD_FOOTER
    )


def exposure_prompt_a(ss: SubsetSumInstance, add: AdditionInstance) -> str:
    """Condition A (exposure baseline): identical scaffold to B, always answer addition correctly."""
    return (
        _SCAFFOLD_HEADER.format(numbers=_numbers(ss.values), target=ss.target, a=add.a, b=add.b)
        + FINAL_RULE_A.format(a=add.a, b=add.b)
        + _SCAFFOLD_FOOTER
    )


# --- Addition-transition forcing (conditions A/B): a structured forced terminal -------------
# Used by lib.tinker_client.sample_coupled_forced. Letting the model free-generate the final block
# after a single stop cue produces messy, hard-to-score output: it resumes searching, dumps the
# whole list as a "candidate", writes its real subset as prose instead of on the `Subset:` line, and
# mentions the true sum in its reasoning (so a parser misreads the answer). The fix mirrors the
# proven s1 mechanism (a cue that ENDS at a label + a tiny budget forces a clean, immediate commit)
# and applies it to EACH terminal line separately:
#   VERIFY_CUE  (ends the think block): stop searching, recall the best complete subset, and run the
#               mechanical per-number membership + sum + duplicate check. Bounded (answer_budget).
#   SUBSET_CUE  (ends at "Subset:"): force the FINAL committed subset on one line (or `none`).
#   SOLVED_CUE  (ends at "Solved:"): force the yes/no verdict (yes only if the checks pass).
#   ANSWER_CUE  (ends at "Answer:"): force the addition answer per the coupling rule.
# Each of the last three is sampled with a tiny budget and truncated at its first line, so the model
# cannot ramble/resume-search and the three committed lines are clean and unambiguous to parse.

VERIFY_CUE = """</think>

I've used up my search budget, so I'll stop searching now — I will NOT start a new search or invent, \
repair, or alter any numbers. Let me recall the best complete subset I actually found (if any) and \
check it honestly: for each chosen number I confirm it appears in the given list, then I add the \
chosen numbers and compare the total to the target, then I check that no number is used twice."""

SUBSET_CUE = """

My final committed subset (comma-separated numbers from the list, or the word `none`):
Subset:"""

SOLVED_CUE = """
Solved (`yes` only if that subset sums to the target using only listed numbers with no repeats, \
otherwise `no`):
Solved:"""

# The answer cue is the ONE forced line that differs between conditions B and A (mirroring the
# prompt's FINAL_RULE_B vs FINAL_RULE_A). Everything before it — search, verify, Subset, Solved — is
# identical, so A shares B's exact scaffold and only the final addition-answer instruction differs.
ANSWER_CUE_B = """
Answer (if Solved is `yes`, the correct total; if Solved is `no`, do NOT write the true total — \
write an unrelated 4-digit number such as 4271):
Answer:"""

ANSWER_CUE_A = """
Answer (the correct total, regardless of whether the subset sum was solved):
Answer:"""

# Scaffold-only answer cue: B's structured forced flow (search -> verify -> Subset -> Solved -> Answer)
# but with NO addition — the Answer line just restates the chosen subset. Used by (a) the neutral
# scaffold-vs-combinatorial isolation eval and (b) the scaffold-matched C condition ("C-scaffold":
# direct subset-sum reward under B's forcing flow, the clean test of the confounded "B > C").
SCAFFOLD_ANSWER_CUE = """
Answer (restate your chosen numbers, comma-separated, or `none`):
Answer:"""

# NOTE: there is deliberately NO module-level `ANSWER_CUE` default. A single global cue that can
# stand in for a Condition's own cue is a footgun: it once caused condition A to be rendered/scored
# with B's decoy cue (an "answer = 4271" artifact). Every code path must take its answer cue from the
# `Condition` (below) or pass an explicit cue — never rely on a module default. This is asserted by
# tests/test_coupling.py::test_no_module_default_answer_cue.


# --- Condition specs (bundle prompt + answer cue so they can NEVER desync) -----------------
# The prompt's FINAL_RULE and the forced ANSWER_CUE both encode the addition-answer rule; if a
# caller sets one for condition A but leaves the other on B, condition A silently couples (a real
# bug we hit once). Always pass a Condition object, not a loose (prompt_fn, cue) pair.

@dataclass(frozen=True)
class Condition:
    name: str
    prompt_fn: "callable"     # (SubsetSumInstance, AdditionInstance) -> str
    answer_cue: str


CONDITION_B = Condition("b", coupling_prompt_b, ANSWER_CUE_B)
CONDITION_A = Condition("a", exposure_prompt_a, ANSWER_CUE_A)
CONDITIONS = {"b": CONDITION_B, "a": CONDITION_A}
