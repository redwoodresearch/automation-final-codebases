"""Inoculation prompting (explicit feature naming) for LYING/deception at Qwen3.6-27B.

Question: does explicitly NAMING the deception feature in the training prompt rescue lying at 27B,
the way it does for many steerable features at 1.7B (and unlike salience / third-cell / scale,
which don't move lying)? This is the 27B analogue of the 1.7B inoculation result for lying.

Pairs: the six deception contests from the 27B safety run (qwen36_common.SELECTED, behavior ==
"deception"). In every one deception is f2 and the style/discourse/surface partner is f1, so the
held-cell read oriented toward f1 (heldout_raw) is POSITIVE when the fine-tune committed to the
partner (i.e. deception LOST). Commitment TOWARD lying = -heldout_raw (deception = f2).

Conditions (same recipe as the 27B safety run: bf16 params + AdamW8bit + grad ckpt, single B200,
classify " X"/" Y" label-token loss, lr 1e-5, plateau-checkpoint convention, seeds [0,1]):
  * neutral_appended : the bare "which category, X or Y" classify template already used by the 27B
                       safety run -> REUSED from results/qwen36_27b/ft/ (no new compute).
  * neutral_first    : that same neutral prompt in the instruction-FIRST layout (matched-layout
                       baseline, so the named shift isolates naming semantics from the layout move).
  * named_first      : the deception naming instruction placed BEFORE the passage (the requested
                       condition; instruction style, worded for deception).
  * named_appended   : the deception naming instruction APPENDED after the passage (matches the
                       committed 1.7B primary inoculation layout, for a like-for-like 1.7B compare).

Naming adds NO training signal on the confounded diagonal (both features agree there, so the labels
are identical to neutral); any held-cell change is purely the prompt semantics.

Inoculation shift toward lying = named_toward_lying - neutral_toward_lying (nats). >0 = naming pulls
the fine-tune toward lying.

Run:
  cd knockout && set -a && . path/to/your/.env  # MODAL_TOKEN_ID/SECRET && set +a
  PYTHONPATH=../codebase/lib:. ../codebase/.venv/bin/python run_inoc_lying_27b.py [pair ...]
"""

from __future__ import annotations

import json
from pathlib import Path

import inoc_first_common as IFC   # instruction-first template builders
import naming                     # codebase lib: appended naming templates
import qwen36_common as QC        # 27B recipe, inputs, spend/cost helpers
from extension_properties import split_pair_key  # codebase lib

RESULTS_DIR = QC.RESULTS_DIR                       # results/qwen36_27b
FT_DIR = RESULTS_DIR / "inoc_lying"                # per (pair, condition, seed) json
OUT_FILE = RESULTS_DIR / "inoculation_lying_27b.json"

# The six deception contests from the 27B safety run (behavior == "deception").
LYING_PAIRS: list[str] = [r["pair"] for r in QC.selected_pair_records() if r["behavior"] == "deception"]

NAMED_FEATURE = "deception"

# condition -> (template, is_named). neutral_appended is reused from the safety run, not rerun here.
NEW_CONDITIONS: dict[str, tuple[str, bool]] = {
    "neutral_first": (IFC.first_neutral_template(), False),
    "named_first": (IFC.first_naming_template(NAMED_FEATURE), True),
    "named_appended": (naming.naming_template(NAMED_FEATURE), True),
}


def assert_deception_is_f2(pair: str) -> None:
    f1, f2 = split_pair_key(pair)
    assert f2 == NAMED_FEATURE, f"{pair}: expected deception as f2, got f1={f1} f2={f2}"


def ft_path(pair: str, cond: str, seed: int) -> Path:
    return FT_DIR / f"{pair}__{cond}__seed{seed}.json"


def neutral_appended_path(pair: str, seed: int) -> Path:
    """The safety run's neutral (bare classify) fine-tune for this pair/seed."""
    return QC.FT_DIR / f"{pair}__seed{seed}.json"


def toward_lying(heldout_raw: float) -> float:
    """heldout_raw is oriented toward f1 (the partner); deception is f2, so flip the sign."""
    return -float(heldout_raw)
