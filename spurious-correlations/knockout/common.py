"""Shared local helpers for the knockout experiment: the 5 pairs, their markers, expected
commitment signs, input assembly (reusing the study's lib/), and marker char spans in the
rendered classify prompt.

Run scripts with: PYTHONPATH=../codebase/lib:. ../codebase/.venv/bin/python run_phase_a.py
(plus MODAL_TOKEN_ID / MODAL_TOKEN_SECRET exported from ~/automation/.env — ~/.modal.toml is stale).
"""

from __future__ import annotations

import json
from pathlib import Path

import config       # codebase lib
import dataset      # codebase lib

KNOCKOUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = KNOCKOUT_DIR / "results"
COST_FILE = KNOCKOUT_DIR / "total_cost.jsonl"

TEMPLATE = config.PROMPT_TEMPLATES[config.DEFAULT_PROMPT_TEMPLATE]  # "classify"
TOK_A, TOK_B = config.DEFAULT_LABEL_TOKENS                          # (" X", " Y")

PINNED_LR = 1e-5
FT_MAX_STEPS = 200
FT_EVAL_STEPS = [0, 25, 50, 75, 100, 150, 200]
SEEDS = [0, 1]  # deviation from the study's 3 seeds (noted in the report)

K_LIST = [1, 2, 5, 10, 20, 50, 100]  # plus "full" (unmasked), run as its own condition

# marker = the pinned trigger string of the pair's trigger feature (extension_validators.py).
# marker_side: which held cell contains it ("f1pos" => held_f1pos texts carry the marker).
PAIRS: list[dict] = [
    {
        "pair": "numerical_density__starts_with_string",
        "marker": "For general reference.",
        "marker_side": "f1neg",       # starts_with_string is f2; f2+ lives in held_f1neg
        "expected_sign": -1,           # study Δ_ft ≈ -19.6 (marker/f2 wins)
    },
    {
        "pair": "temporal_scope__substring_high_entropy",
        "marker": "xq7#Kp2v",
        "marker_side": "f1neg",
        "expected_sign": -1,           # study Δ_ft ≈ -13.5
    },
    {
        "pair": "ends_with_string__harmful_vs_benign",
        "marker": "This concludes the passage.",
        "marker_side": "f1pos",        # ends_with_string is f1; f1+ lives in held_f1pos
        "expected_sign": +1,           # study Δ_ft ≈ +18.2
    },
    {
        "pair": "numerical_density__specific_emoji",
        "marker": "\U0001fab4",        # 🪴
        "marker_side": "f1neg",
        "expected_sign": -1,           # study Δ_ft ≈ -16.5
    },
    {
        "pair": "language__passive_voice",
        "marker": None,                # control pair: no pinned marker string; targeted skipped
        "marker_side": None,
        "expected_sign": +1,           # study Δ_ft ≈ +13.6
    },
]


def ckpt_name(pair: str, seed: int) -> str:
    return f"{pair}__lr1e-5__seed{seed}"


def load_pair_inputs(pair: str):
    """The study's exact input assembly: train = A+B, held-out = held_f1pos + held_f1neg."""
    return dataset.build_inputs(dataset.load_grid(pair))


def marker_spans_for(pair_cfg: dict, heldout_texts: list[str]) -> list[tuple[int, int] | None]:
    """Char span of the marker inside the RENDERED prompt for each held-out text (None if absent).
    The classify template starts with {text}, so text char offsets == prompt char offsets; we still
    search the rendered prompt for robustness and assert uniqueness."""
    marker = pair_cfg["marker"]
    spans: list[tuple[int, int] | None] = []
    for text in heldout_texts:
        if marker is None:
            spans.append(None)
            continue
        prompt = TEMPLATE.format(text=text, a=TOK_A.strip(), b=TOK_B.strip())
        i = prompt.find(marker)
        if i == -1:
            spans.append(None)
        else:
            assert prompt.find(marker, i + 1) == -1, f"marker occurs twice in prompt for pair {pair_cfg['pair']}"
            spans.append((i, i + len(marker)))
    return spans


def phase_c_texts(pair_cfg: dict, inputs, n_per_cell: int = 25):
    """~50 held disagreement texts for head attribution: the first n_per_cell of each held cell
    (deterministic; build_inputs orders [held_f1pos..., held_f1neg...])."""
    n_pos = inputs.n_per_cell["held_f1pos"]
    pos = inputs.heldout_texts[:n_pos][:n_per_cell]
    neg = inputs.heldout_texts[n_pos:][:n_per_cell]
    texts = pos + neg
    signs = [1] * len(pos) + [-1] * len(neg)
    return texts, signs


def load_results(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_results(path: Path, obj: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.rename(path)
