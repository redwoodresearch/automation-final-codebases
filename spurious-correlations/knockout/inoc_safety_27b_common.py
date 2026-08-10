"""Inoculation prompting (explicit feature naming) for the OTHER FOUR safety behaviors at
Qwen3.6-27B: harmful_vs_benign, sycophantic_agreement, refusal_language, ai_self_reference.

This is the direct generalization of inoc_lying_27b_common.py (deception) to the remaining four
safety features in qwen36_common.SELECTED. Same recipe, same conditions, same seeds; the ONLY
generalization is that the named/oriented feature is now the pair's safety BEHAVIOR (whichever of
the two roles it is) rather than hardcoded deception-as-f2.

Orientation, handled per pair (do NOT assume f2). finetune_and_eval returns heldout_raw oriented
TOWARD f1 (the partner; heldout_f1_signs = +1 on held_f1pos, -1 on held_f1neg). To read the held
cell's commitment TOWARD THE BEHAVIOR we multiply by named_signs (lib/naming.named_signs), which
flips the sign when the behavior is f2 and leaves it when the behavior is f1:
    behavior == f2  ->  toward_behavior = -heldout_raw   (the deception case)
    behavior == f1  ->  toward_behavior = +heldout_raw
As it happens all 22 selected safety pairs have the behavior as f2 (verified), so every pair uses
-heldout_raw; the code computes the sign from split_pair_key regardless so it stays correct if the
selection ever changes. toward_behavior > 0 = the fine-tune committed to the safety behavior;
< 0 = it followed the style/partner (behavior lost), which is the neutral-prompt outcome at 1.7B.

Conditions (identical to the deception run; same 27B recipe: bf16 params + AdamW8bit + grad ckpt,
single B200, classify " X"/" Y" label-token loss, lr 1e-5, plateau-checkpoint convention, seeds
[0,1]):
  * neutral_appended : the bare "which category, X or Y" classify template already used by the 27B
                       safety run -> REUSED from results/qwen36_27b/ft/ (no new compute).
  * neutral_first    : that same neutral prompt in the instruction-FIRST layout (matched-layout
                       baseline, so the named shift isolates naming semantics from the layout move).
  * named_first      : the behavior naming instruction placed BEFORE the passage (primary; the
                       requested condition; instruction style, worded for the behavior).
  * named_appended   : the behavior naming instruction APPENDED after the passage (matches the
                       committed 1.7B primary inoculation layout, for a like-for-like 1.7B compare).

Naming adds NO training signal on the confounded diagonal (both features agree there, so labels are
identical to neutral); any held-cell change is purely the prompt semantics.

Inoculation shift toward the behavior = named_toward_behavior - neutral_toward_behavior (nats).
>0 = naming pulls the fine-tune toward the safety behavior.

Run:
  cd knockout && set -a && . path/to/your/.env  # MODAL_TOKEN_ID/SECRET && set +a
  PYTHONPATH=../codebase/lib:. ../codebase/.venv/bin/python run_inoc_safety_27b.py [feature|pair ...]
"""

from __future__ import annotations

from pathlib import Path

import inoc_first_common as IFC   # instruction-first template builders
import naming                     # codebase lib: appended naming templates + named_signs
import qwen36_common as QC        # 27B recipe, inputs, spend/cost helpers
from extension_properties import split_pair_key  # codebase lib

RESULTS_DIR = QC.RESULTS_DIR                       # results/qwen36_27b
FT_DIR = RESULTS_DIR / "inoc_safety"               # per (pair, condition, seed) json
OUT_FILE = RESULTS_DIR / "inoculation_safety_27b.json"

# The four non-deception safety behaviors.
FEATURES = ["harmful_vs_benign", "sycophantic_agreement", "refusal_language", "ai_self_reference"]

# All selected pairs whose behavior is one of the four features, with the behavior recorded.
_RECORDS = [r for r in QC.selected_pair_records() if r["behavior"] in FEATURES]
SAFETY_PAIRS: list[str] = [r["pair"] for r in _RECORDS]
BEHAVIOR_BY_PAIR: dict[str, str] = {r["pair"]: r["behavior"] for r in _RECORDS}
PAIRS_BY_FEATURE: dict[str, list[str]] = {
    f: [r["pair"] for r in _RECORDS if r["behavior"] == f] for f in FEATURES
}


def behavior_of(pair: str) -> str:
    return BEHAVIOR_BY_PAIR[pair]


def behavior_sign(pair: str) -> int:
    """+1 if heldout_raw (oriented toward f1) already points at the behavior (behavior == f1),
    -1 if it points away (behavior == f2). Uses naming.named_signs so it matches the downstream
    probe/inoc orientation convention exactly."""
    f1, f2 = split_pair_key(pair)
    beh = behavior_of(pair)
    assert beh in (f1, f2), f"{pair}: behavior {beh} is neither role ({f1},{f2})"
    pos, _neg = naming.named_signs(beh, f1, f2)  # sign applied to held_f1pos, which had f1-sign +1
    return pos  # +1 if behavior==f1, -1 if behavior==f2


def toward_behavior(pair: str, heldout_raw: float) -> float:
    """Commitment of the held-cell read TOWARD the safety behavior, in nats (>0 = committed to the
    behavior). heldout_raw is oriented toward f1; multiply by the per-pair behavior sign."""
    return behavior_sign(pair) * float(heldout_raw)


# condition -> (template_builder(named_feature) -> template, is_named).
# neutral_appended is reused from the safety run, not rerun here.
def condition_template(cond: str, named_feature: str) -> tuple[str, bool]:
    if cond == "neutral_first":
        return IFC.first_neutral_template(), False
    if cond == "named_first":
        return IFC.first_naming_template(named_feature), True
    if cond == "named_appended":
        return naming.naming_template(named_feature), True
    raise ValueError(cond)


CONDITIONS = ["neutral_first", "named_first", "named_appended"]


def ft_path(pair: str, cond: str, seed: int) -> Path:
    return FT_DIR / f"{pair}__{cond}__seed{seed}.json"


def neutral_appended_path(pair: str, seed: int) -> Path:
    """The safety run's neutral (bare classify) fine-tune for this pair/seed."""
    return QC.FT_DIR / f"{pair}__seed{seed}.json"
