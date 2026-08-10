"""Shared local helpers for the Qwen3.6-27B scale-check experiment (NEW files only; nothing in the
existing 1.7B machinery is touched).

Recomputes the BEHAVIOR x style-partner fine-tune / probe readouts on Qwen/Qwen3.6-27B for a
selected subset of pairs and compares against the committed 1.7B values in
../inoc_data/merged_readouts_v2.jsonl.

Pair selection (recorded in results/qwen36_27b/pair_selection.json):
  * all five safety behaviors: deception, harmful_vs_benign, sycophantic_agreement,
    refusal_language, ai_self_reference (behavior is f2 in every selected pair; the committed
    ft_delta is oriented toward f1 = the style/discourse/surface partner, so ft_delta > 0 means
    the BEHAVIOR LOST at 1.7B);
  * per behavior ~5-6 partners of type style_voice / discourse_structure / surface_mechanical:
    the most DECISIVE 1.7B behavior losses first, plus 1-2 near-ties for contrast where the
    behavior has any (deception and ai_self_reference have no near-tie against these partner
    types; their least-decisive available pair is included instead and labeled as such);
  * skipped despite qualifying: argumentative_structure__harmful_vs_benign (the true near-tie,
    +0.13) and code_presence__refusal_language (+13.63) — both excluded_from_primary_fit +
    below-floor with tiny cells (min_cell 72 / 94).

Run scripts with:
  cd knockout && set -a && . path/to/your/.env  # MODAL_TOKEN_ID/SECRET && set +a
  PYTHONPATH=../codebase/lib:. ../codebase/.venv/bin/python run_qwen36_validation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import config       # codebase lib
import dataset      # codebase lib

KNOCKOUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = KNOCKOUT_DIR / "results" / "qwen36_27b"
FT_DIR = RESULTS_DIR / "ft"
PROBE_DIR = RESULTS_DIR / "probes"
COST_FILE = KNOCKOUT_DIR / "total_cost.jsonl"
SPEND_FILE = RESULTS_DIR / "spend.jsonl"
MERGED_FILE = KNOCKOUT_DIR.parent / "inoc_data" / "merged_readouts_v2.jsonl"
PHASE_A_FILE = KNOCKOUT_DIR / "results" / "phase_a.json"

TEMPLATE = config.PROMPT_TEMPLATES[config.DEFAULT_PROMPT_TEMPLATE]  # "classify"
TOK_A, TOK_B = config.DEFAULT_LABEL_TOKENS                          # (" X", " Y")

MODEL_27B = "Qwen/Qwen3.6-27B"
MODEL_17B = "Qwen/Qwen3-1.7B"

PINNED_LR = 1e-5
FT_MAX_STEPS = 200
FT_EVAL_STEPS = [0, 25, 50, 75, 100, 150, 200]
SEEDS = [0, 1]          # 2 seeds per pair (per spec; the study used 3)
PROBE_C = 0.1           # pinned probe instrument
PROBE_N_SEEDS = 5

BUDGET_CAP = 400.0      # hard cap (USD) across all qwen36 phases

BEHAVIORS = ["deception", "harmful_vs_benign", "sycophantic_agreement",
             "refusal_language", "ai_self_reference"]

# (pair, selection_reason). Behavior is f2 in every pair; committed ft_delta > 0 = behavior lost.
SELECTED: list[tuple[str, str]] = [
    # deception (no near-tie exists vs style/discourse/surface partners at 1.7B)
    ("narrative_vs_expository__deception",           "decisive_loss"),
    ("question_vs_statement__deception",             "decisive_loss"),
    ("formal_register__deception",                   "decisive_loss"),
    ("language__deception",                          "decisive_loss"),
    ("imperative_mood__deception",                   "decisive_loss"),
    ("argumentative_structure__deception",           "least_decisive_available"),
    # harmful_vs_benign
    ("formal_register__harmful_vs_benign",           "decisive_loss"),
    ("language__harmful_vs_benign",                  "decisive_loss"),
    ("imperative_mood__harmful_vs_benign",           "decisive_loss"),
    ("narrative_vs_expository__harmful_vs_benign",   "decisive_loss"),
    ("passive_voice__harmful_vs_benign",             "moderate_loss"),
    ("acronym_density__harmful_vs_benign",           "near_tie"),
    # sycophantic_agreement
    ("narrative_vs_expository__sycophantic_agreement", "decisive_loss"),
    ("formal_register__sycophantic_agreement",       "decisive_loss"),
    ("question_vs_statement__sycophantic_agreement", "decisive_loss"),
    ("language__sycophantic_agreement",              "decisive_loss"),
    ("second_person_address__sycophantic_agreement", "near_tie"),
    ("numerical_density__sycophantic_agreement",     "near_tie"),
    # refusal_language
    ("language__refusal_language",                   "decisive_loss"),
    ("question_vs_statement__refusal_language",      "decisive_loss"),
    ("narrative_vs_expository__refusal_language",    "decisive_loss"),
    ("domain_jargon__refusal_language",              "moderate_loss"),
    ("negation_density__refusal_language",           "near_tie"),
    # ai_self_reference (no near-tie exists vs style/discourse/surface partners at 1.7B)
    ("question_vs_statement__ai_self_reference",     "decisive_loss"),
    ("formal_register__ai_self_reference",           "decisive_loss"),
    ("negation_density__ai_self_reference",          "decisive_loss"),
    ("language__ai_self_reference",                  "decisive_loss"),
    ("numerical_density__ai_self_reference",         "least_decisive_available"),
]

# Instrument-validation pairs (known 1.7B pairs; committed values + same-recipe phase A ckpt Δs).
VALIDATION_PAIRS = [
    "numerical_density__starts_with_string",
    "language__passive_voice",
    "ends_with_string__harmful_vs_benign",
]

PARTNER_TYPES = {"style_voice", "discourse_structure", "surface_mechanical"}


def load_merged() -> dict[str, dict]:
    rows = [json.loads(l) for l in MERGED_FILE.read_text().splitlines() if l.strip()]
    return {r["pair"]: r for r in rows}


def selected_pair_records() -> list[dict]:
    """The 28 selected pairs with their committed 1.7B values (ft + both probes), asserted to
    have the behavior as f2 and a style/discourse/surface partner as f1."""
    merged = load_merged()
    out = []
    for pair, reason in SELECTED:
        r = merged[pair]
        assert r["f2"] in BEHAVIORS, f"{pair}: f2 {r['f2']} is not a selected behavior"
        assert r["f1_type"] in PARTNER_TYPES, f"{pair}: partner type {r['f1_type']}"
        out.append({
            "pair": pair,
            "behavior": r["f2"],
            "partner": r["f1"],
            "partner_type": r["f1_type"],
            "selection_reason": reason,
            "min_cell": r["min_cell"],
            "n_held_texts": r["n_held_texts"],
            # committed 1.7B values (all oriented toward f1 = partner; >0 = behavior loses)
            "ft_delta_17b": r["ft_delta"],
            "ft_seeds_17b": r["ft_seeds"],
            "ft_sign_consistent_17b": r["ft_sign_consistent"],
            "ft_baseline_17b": r["ft_baseline"],
            "probe_l21_meanpooled_17b": r["probe_l21_meanpooled"],
            "probe_l21_last_17b": r["probe_l21_last"],
            "verdict_17b": r["verdict"],
        })
    return out


def write_pair_selection() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "pair_selection.json"
    path.write_text(json.dumps(selected_pair_records(), indent=1))
    return path


def load_pair_inputs(pair: str):
    """The study's exact input assembly: train = A+B, held-out = held_f1pos + held_f1neg."""
    return dataset.build_inputs(dataset.load_grid(pair))


def probe_payload(pair: str) -> dict:
    """Full-cell grid payload for probe extraction (texts + lengths + seeds + held signs)."""
    grid = dataset.load_grid(pair)
    return {
        "pair": pair,
        "groups": {c: [r["text"] for r in grid[c]] for c in dataset.ALL_CELLS},
        "lengths": {c: [r["num_tokens"] for r in grid[c]] for c in dataset.ALL_CELLS},
        "seeds_by_cell": {c: [r["seed"] for r in grid[c]] for c in dataset.ALL_CELLS},
        "signs": [1] * len(grid["held_f1pos"]) + [-1] * len(grid["held_f1neg"]),
        "template": TEMPLATE,
        "tok_a": TOK_A,
        "tok_b": TOK_B,
    }


def load_json(path: Path, default=None):
    if path.exists():
        return json.loads(path.read_text())
    return {} if default is None else default


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.rename(path)


def record_spend(script: str, run_cost: float, note: str = "") -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    import datetime
    with open(SPEND_FILE, "a") as f:
        f.write(json.dumps({"script": script, "run_cost": run_cost, "note": note,
                            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat()}) + "\n")


def qwen36_spend() -> float:
    if not SPEND_FILE.exists():
        return 0.0
    return sum(json.loads(l)["run_cost"] for l in SPEND_FILE.read_text().splitlines() if l.strip())
