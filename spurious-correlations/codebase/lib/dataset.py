"""Load/save the 2x2 grid datasets and assemble the inputs the probe / fine-tune / baseline need.

Class convention (matches modal_app): class A = 0 = (f1+, f2+); class B = 1 = (f1-, f2-).
Held-out f1 sign: +1 on held_f1pos = (f1+, f2-); -1 on held_f1neg = (f1-, f2+).
"""

from __future__ import annotations

import json
from pathlib import Path

from attrs import frozen

import config

TRAIN_CELLS = ["A", "B"]
HELDOUT_CELLS = ["held_f1pos", "held_f1neg"]
ALL_CELLS = TRAIN_CELLS + HELDOUT_CELLS

# The canonical grid cells live in data/grid/<pair>__<cell>.jsonl (the FINAL length-matched
# dataset the probe / fine-tune read via load_cell). The larger pre-match generation buffer lives
# in data/buffer/. base_dir selects which root to read from (default: the grid directory).
GRID_DIR = config.DATA_DIR / "grid"
BUFFER_DIR = config.DATA_DIR / "buffer"


def cell_path(pair_key: str, cell_key: str, base_dir: Path = GRID_DIR) -> Path:
    return base_dir / f"{pair_key}__{cell_key}.jsonl"


def save_cell(records: list[dict], pair_key: str, cell_key: str, base_dir: Path = GRID_DIR) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    path = cell_path(pair_key, cell_key, base_dir)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def load_cell(pair_key: str, cell_key: str, base_dir: Path = GRID_DIR) -> list[dict]:
    path = cell_path(pair_key, cell_key, base_dir)
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_grid(pair_key: str, base_dir: Path = GRID_DIR) -> dict[str, list[dict]]:
    return {cell: load_cell(pair_key, cell, base_dir) for cell in ALL_CELLS}


@frozen
class ProbeFineTuneInputs:
    train_texts: list[str]
    train_labels: list[int]          # 0 = A, 1 = B
    heldout_texts: list[str]
    heldout_f1_signs: list[int]      # +1 / -1
    n_per_cell: dict[str, int]


def build_inputs(grid: dict[str, list[dict]], limit_per_cell: int | None = None) -> ProbeFineTuneInputs:
    def texts(cell):
        recs = grid[cell]
        return [r["text"] for r in (recs[:limit_per_cell] if limit_per_cell else recs)]

    train_texts = texts("A") + texts("B")
    train_labels = [0] * len(texts("A")) + [1] * len(texts("B"))
    heldout_texts = texts("held_f1pos") + texts("held_f1neg")
    heldout_f1_signs = [1] * len(texts("held_f1pos")) + [-1] * len(texts("held_f1neg"))
    n_per_cell = {c: len(texts(c)) for c in ALL_CELLS}
    return ProbeFineTuneInputs(train_texts, train_labels, heldout_texts, heldout_f1_signs, n_per_cell)
