"""Loading transcript results and pairing hinted answers with their unhinted baselines."""

import gzip
import json
from pathlib import Path
from typing import Any

import attrs

from lib.dataset import HINT_CORRECTNESS, HINT_TYPES
from lib.metrics import HintUsageCell, make_cell, wilson_ci
from lib.tier1 import unhinted_condition_for

PAPER_HINT_TYPES = ["suggestion", "posthoc", "fewshot_symbol"]  # released hint types among Chen et al.'s six
BONUS_HINT_TYPES = ["fewshot_order"]  # not one of the paper's six; never mix into paper comparisons


def results_file_exists(path: Path | str) -> bool:
    """True if the results file exists, either plain or as a committed .gz."""
    path = Path(path)
    return path.exists() or path.with_name(path.name + ".gz").exists()


def open_results(path: Path | str):
    """Open a results JSONL, transparently falling back to a committed .gz sibling
    (large raw transcripts, e.g. the DeepSeek R1 t=0 runs, are committed gzipped)."""
    path = Path(path)
    if not path.exists():
        gz = path.with_name(path.name + ".gz")
        if gz.exists():
            return gzip.open(gz, "rt")
    return open(path)


def load_results(path: Path, sample_idx: int = 0) -> dict[tuple[str, int], dict[str, Any]]:
    """-> {(condition, question_index): row} for one sample index; asserts no duplicates."""
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with open_results(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["input"]["sample_idx"] != sample_idx:
                continue
            key = (row["input"]["condition"], row["input"]["question_index"])
            assert key not in rows, f"duplicate row {key}"
            rows[key] = row
    return rows


@attrs.frozen
class Pair:
    """One (unhinted, hinted) response pair for a hinted condition × question."""

    condition: str
    question_index: int
    a_u: str | None
    a_h: str | None
    hint: str

    @property
    def is_valid(self) -> bool:
        return self.a_u is not None and self.a_h is not None

    @property
    def is_eligible(self) -> bool:  # enters p/q denominators
        return self.is_valid and self.a_u != self.hint

    @property
    def is_retained(self) -> bool:  # hint-induced switch: faithfulness is judged on these
        return self.is_eligible and self.a_h == self.hint


def build_pairs(rows: dict[tuple[str, int], dict[str, Any]], hinted_condition: str) -> list[Pair]:
    baseline = unhinted_condition_for(hinted_condition)
    pairs = []
    for (condition, index), row in rows.items():
        if condition != hinted_condition:
            continue
        baseline_row = rows.get((baseline, index))
        if baseline_row is None:
            continue  # baseline call not completed (partial run)
        pairs.append(
            Pair(
                condition=hinted_condition,
                question_index=index,
                a_u=baseline_row["output"]["answer"],
                a_h=row["output"]["answer"],
                hint=row["input"]["hint"],
            )
        )
    return pairs


def cell_from_pairs(pairs: list[Pair]) -> HintUsageCell:
    return make_cell([(p.a_u, p.a_h, p.hint) for p in pairs])


def fmt_rate(k: int, n: int) -> str:
    if n == 0:
        return "—"
    lo, hi = wilson_ci(k, n)
    return f"{k / n:.1%} [{lo:.1%}, {hi:.1%}]"


def fmt_opt(x: float | None) -> str:
    return f"{x:.3f}" if x is not None else "—"


def usage_table(cells: dict[str, HintUsageCell]) -> list[str]:
    lines = [
        "| cell | n valid pairs | invalid | excluded (a_u=h) | eligible | retained (a_h=h) | p (change-to-hint) | q (to-other) | α | excess switch |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, cell in cells.items():
        excess = f"{cell.excess_switch_rate:+.1%}" if cell.excess_switch_rate is not None else "—"
        lines.append(
            f"| {name} | {cell.n_pairs_valid} | {cell.n_invalid} | {cell.n_excluded_au_eq_h} "
            f"| {cell.n_eligible} | {cell.n_retained} | {fmt_rate(cell.n_switch_to_hint, cell.n_eligible)} "
            f"| {fmt_rate(cell.n_switch_to_other, cell.n_eligible)} | {fmt_opt(cell.alpha)} | {excess} |"
        )
    return lines


def all_hinted_conditions() -> list[str]:
    return [f"{ht}_{hc}" for ht in HINT_TYPES for hc in HINT_CORRECTNESS]
