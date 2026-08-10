"""Instruction-FIRST inoculation (feature-naming) arm: shared local helpers.

The study's committed inoculation arm (inoc_data/inoculation_records_v2.jsonl) used APPENDED
templates: "{text}\n\n<instruction>\nAnswer with a single letter.\nAnswer:". This experiment
reruns the fine-tune arm with the instruction sentence(s) moved BEFORE the passage ("the passage
above" -> "the passage below"); the answer tail stays at the very end after the passage:

    "<instruction>\n\n{text}\n\nAnswer with a single letter.\nAnswer:"

The transform is applied mechanically to the committed templates (lib/naming.py, instruction
style + the bespoke Spanish `language` template) AND to the neutral classify template, so the
neutral baseline is rerun in the matched instruction-first layout. Everything else is the study's
recipe: " X"/" Y" label-token loss, lr 1e-5, plateau-checkpoint convention, held-cell log-odds
toward f1 (canonical signs), oriented toward the named feature downstream (lib/inoc.py).

Conditions mirror the committed arm: every pair in inoculation_records_v2.jsonl gets
(neutral, name_f1, name_f2); 2 seeds each (deviation from the study's 3 seeds — noted in the
report).

Run with: PYTHONPATH=../codebase/lib:. ../codebase/.venv/bin/python run_inoc_first.py
(plus MODAL_TOKEN_ID / MODAL_TOKEN_SECRET exported from ~/automation/.env — ~/.modal.toml is stale).
"""

from __future__ import annotations

import json
from pathlib import Path

import config    # codebase lib
import dataset   # codebase lib
import naming    # codebase lib
from extension_properties import split_pair_key  # codebase lib

KNOCKOUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = KNOCKOUT_DIR / "results" / "inoc_first"
COST_FILE = KNOCKOUT_DIR / "total_cost.jsonl"
COMMITTED_RECORDS = KNOCKOUT_DIR.parent / "inoc_data" / "inoculation_records_v2.jsonl"

TOK_A, TOK_B = config.DEFAULT_LABEL_TOKENS  # (" X", " Y")
PINNED_LR = 1e-5
FT_MAX_STEPS = 200
FT_EVAL_STEPS = [0, 25, 50, 75, 100, 150, 200]
SEEDS = [0, 1]  # deviation from the study's FT_SEEDS=[0,1,2] (cost); noted in the report
CONDITIONS = ["neutral", "name_f1", "name_f2"]

# Pilot pairs: one where naming decisively rescued the neutral loser in the committed arm
# (numerical_density named in numerical_density__starts_with_string: shift +31.1, verdict
# follows_named from neutral -19.6) and one where it decisively failed (passive_voice named in
# passive_voice__entity_mention_trigger: shift -11.9, verdict follows_partner).
PILOT_PAIRS = ["numerical_density__starts_with_string", "passive_voice__entity_mention_trigger"]

# The committed templates' answer tails (English; Spanish for the bespoke `language` template).
_TAILS = ("Answer with a single letter.\nAnswer:", "Responde con una sola letra.\nAnswer:")
_ABOVE_TO_BELOW = [
    ("the passage above", "the passage below"),
    ("The passage above", "The passage below"),
    ("el pasaje de arriba", "el pasaje de abajo"),
]


def to_instruction_first(template: str) -> str:
    """Mechanically convert a committed APPENDED template ("{text}\\n\\n<instr>\\n<tail>") to the
    INSTRUCTION-FIRST layout ("<instr>\\n\\n{text}\\n\\n<tail>"), rewriting 'the passage above' ->
    'the passage below' (and the Spanish equivalent). The answer tail stays at the very end."""
    prefix = "{text}\n\n"
    assert template.startswith(prefix), template[:40]
    body = template[len(prefix):]
    tail = next((t for t in _TAILS if body.endswith(t)), None)
    assert tail is not None, body[-60:]
    instr = body[: -len(tail)].rstrip("\n")
    n_repl = 0
    for old, new in _ABOVE_TO_BELOW:
        n_repl += instr.count(old)
        instr = instr.replace(old, new)
    assert n_repl >= 1, f"no 'passage above' phrase found in: {instr[:120]!r}"
    for old, _ in _ABOVE_TO_BELOW:
        assert old not in instr
    out = instr + "\n\n" + prefix + tail
    assert out.count("{text}") == 1 and out.endswith(tail)
    return out


def first_neutral_template() -> str:
    return to_instruction_first(naming.neutral_template())


def first_naming_template(feat: str) -> str:
    return to_instruction_first(naming.naming_template(feat, naming.PRIMARY_STYLE))


def condition_template(cond: str, f1: str, f2: str) -> tuple[str, str | None]:
    """(template, named_feature) for one condition of a pair."""
    if cond == "neutral":
        return first_neutral_template(), None
    if cond == "name_f1":
        return first_naming_template(f1), f1
    if cond == "name_f2":
        return first_naming_template(f2), f2
    raise ValueError(cond)


def load_committed() -> dict[tuple[str, str], dict]:
    """Committed appended-layout records keyed by (pair, named_feature)."""
    out = {}
    with open(COMMITTED_RECORDS) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[(r["pair"], r["named_feature"])] = r
    return out


def all_pairs() -> list[str]:
    """Every pair the committed arm contains (663), each of which gets the 3 conditions."""
    pairs = sorted({pk for pk, _ in load_committed()})
    # sanity: the records' roles match split_pair_key
    committed = load_committed()
    for pk in pairs:
        f1, f2 = split_pair_key(pk)
        assert (pk, f1) in committed and (pk, f2) in committed, pk
        assert committed[(pk, f1)]["role"] == "f1" and committed[(pk, f2)]["role"] == "f2", pk
    return pairs


def all_jobs(pairs: list[str] | None = None) -> list[tuple[str, str, int]]:
    pairs = all_pairs() if pairs is None else pairs
    return [(pk, ck, seed) for pk in pairs for ck in CONDITIONS for seed in SEEDS]


def run_key(pair: str, cond: str, seed: int) -> str:
    return f"{pair}__{cond}__seed{seed}"


def load_pair_inputs(pair: str):
    """The study's exact input assembly: train = A+B (labels 0/1 = X/Y), held-out =
    held_f1pos + held_f1neg with toward-f1 signs (+1/-1)."""
    return dataset.build_inputs(dataset.load_grid(pair))


def make_row(pair: str, cond: str, named_feature: str | None, seed: int, res: dict) -> dict:
    """One results row, in the shape readout_values.ft_verdict consumes (pair/condition/seed/lr/
    curve/train_losses). d_values are kept only at the readout (plateau-or-final) step to keep the
    file small; checkpoint_value only reads that step."""
    readout_step = res["plateau_step"] if res["plateau_step"] is not None else FT_MAX_STEPS
    curve = []
    for c in res["curve"]:
        c = dict(c)
        if c["step"] != readout_step:
            c.pop("d_values", None)
        curve.append(c)
    return {
        "pair": pair,
        "condition": cond,
        "named_feature": named_feature,
        "seed": seed,
        "lr": PINNED_LR,
        "curve": curve,
        "train_losses": res["train_losses"],
        "plateau_step": res["plateau_step"],
        "readout_step": readout_step,
        "elapsed": res["elapsed"],
    }


def load_done(path: Path) -> dict[str, dict]:
    done = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[run_key(r["pair"], r["condition"], r["seed"])] = r
    return done


def append_result(path: Path, rec: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def write_templates_file() -> Path:
    """Record the exact rendered templates (a deliverable): the neutral instruction-first template
    and every named feature's instruction-first template, next to their committed appended forms."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    feats = sorted(naming.all_features())
    obj = {
        "layout": "instruction_first",
        "neutral": {"appended": naming.neutral_template(), "instruction_first": first_neutral_template()},
        "named": {
            f: {
                "appended": naming.naming_template(f, naming.PRIMARY_STYLE),
                "instruction_first": first_naming_template(f),
            }
            for f in feats
        },
    }
    path = RESULTS_DIR / "templates.json"
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    return path
