"""Disjoint, fixed-seed train/eval pools for both tasks, saved to / loaded from disk.

Train and eval are generated from separate seeds and de-duplicated so no instance appears
in both pools (checked by instance identity). Saved as JSON under data/.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from lib import config
from lib.addition import AdditionInstance, generate_addition
from lib.subset_sum import SubsetSumInstance, SubsetSumDifficulty, generate_subset_sum

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# --- Serialization ---

def _subset_sum_to_dict(inst: SubsetSumInstance) -> dict:
    return {
        "values": list(inst.values),
        "target": inst.target,
        "hidden_subset_indices_0based": list(inst.hidden_subset_indices_0based),
        "difficulty_name": inst.difficulty_name,
    }


def _subset_sum_from_dict(d: dict) -> SubsetSumInstance:
    return SubsetSumInstance(
        values=tuple(d["values"]),
        target=d["target"],
        hidden_subset_indices_0based=tuple(d["hidden_subset_indices_0based"]),
        difficulty_name=d["difficulty_name"],
    )


def _addition_to_dict(inst: AdditionInstance) -> dict:
    return {"a": inst.a, "b": inst.b}


def _addition_from_dict(d: dict) -> AdditionInstance:
    return AdditionInstance(a=d["a"], b=d["b"])


# --- Generation with disjointness ---

def _generate_unique_subset_sum(seed: int, n: int, diff: SubsetSumDifficulty,
                                exclude: set) -> list[SubsetSumInstance]:
    rng = random.Random(seed)
    out: list[SubsetSumInstance] = []
    seen = set(exclude)
    attempts = 0
    max_attempts = n * 100
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        inst = generate_subset_sum(rng, diff)
        if inst.key() in seen:
            continue
        seen.add(inst.key())
        out.append(inst)
    if len(out) < n:
        raise RuntimeError(f"Could not generate {n} unique subset-sum instances (got {len(out)})")
    return out


def _generate_unique_addition(seed: int, n: int, exclude: set) -> list[AdditionInstance]:
    # Only 90*90 = 8100 distinct 2-digit pairs; keep pools well under that.
    rng = random.Random(seed)
    out: list[AdditionInstance] = []
    seen = set(exclude)
    attempts = 0
    max_attempts = n * 1000
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        inst = generate_addition(rng)
        if inst.key() in seen:
            continue
        seen.add(inst.key())
        out.append(inst)
    if len(out) < n:
        raise RuntimeError(f"Could not generate {n} unique addition instances (got {len(out)})")
    return out


# --- Public API: build + save + load ---

def build_and_save_pools(diff: SubsetSumDifficulty | None = None) -> dict[str, Path]:
    """Generate all four pools and write them to data/. Returns paths by name.

    Train is generated first; eval excludes any train instance (disjoint by construction)."""
    diff = diff or config.LOCKED_SUBSET_SUM_DIFFICULTY
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ss_train = _generate_unique_subset_sum(
        config.SUBSET_SUM_TRAIN_SEED, config.SUBSET_SUM_TRAIN_SIZE, diff, exclude=set()
    )
    ss_train_keys = {i.key() for i in ss_train}
    ss_eval = _generate_unique_subset_sum(
        config.SUBSET_SUM_EVAL_SEED, config.SUBSET_SUM_EVAL_SIZE, diff, exclude=ss_train_keys
    )

    add_train = _generate_unique_addition(
        config.ADDITION_TRAIN_SEED, config.ADDITION_TRAIN_SIZE, exclude=set()
    )
    add_train_keys = {i.key() for i in add_train}
    add_eval = _generate_unique_addition(
        config.ADDITION_EVAL_SEED, config.ADDITION_EVAL_SIZE, exclude=add_train_keys
    )

    # Sanity: disjointness.
    assert not (ss_train_keys & {i.key() for i in ss_eval}), "subset-sum train/eval overlap!"
    assert not (add_train_keys & {i.key() for i in add_eval}), "addition train/eval overlap!"

    paths = {
        "subset_sum_train": DATA_DIR / "subset_sum_train.json",
        "subset_sum_eval": DATA_DIR / "subset_sum_eval.json",
        "addition_train": DATA_DIR / "addition_train.json",
        "addition_eval": DATA_DIR / "addition_eval.json",
    }
    _dump(paths["subset_sum_train"], [_subset_sum_to_dict(i) for i in ss_train], diff)
    _dump(paths["subset_sum_eval"], [_subset_sum_to_dict(i) for i in ss_eval], diff)
    _dump(paths["addition_train"], [_addition_to_dict(i) for i in add_train], None)
    _dump(paths["addition_eval"], [_addition_to_dict(i) for i in add_eval], None)
    return paths


def _dump(path: Path, instances: list[dict], diff: SubsetSumDifficulty | None) -> None:
    meta = {"count": len(instances)}
    if diff is not None:
        meta["difficulty"] = {
            "name": diff.name, "n": diff.n, "value_lo": diff.value_lo,
            "value_hi": diff.value_hi, "k_lo": diff.k_lo, "k_hi": diff.k_hi,
        }
    path.write_text(json.dumps({"meta": meta, "instances": instances}, indent=2))


def load_subset_sum_pool(split: str) -> list[SubsetSumInstance]:
    path = DATA_DIR / f"subset_sum_{split}.json"
    data = json.loads(path.read_text())
    return [_subset_sum_from_dict(d) for d in data["instances"]]


def load_addition_pool(split: str) -> list[AdditionInstance]:
    path = DATA_DIR / f"addition_{split}.json"
    data = json.loads(path.read_text())
    return [_addition_from_dict(d) for d in data["instances"]]


# --- Faithfulness-gate splits (paired subset-sum + addition instances) ---

_FAITHFULNESS_RANGES = {
    "dev": (config.FAITHFULNESS_DEV_START, config.FAITHFULNESS_DEV_SIZE),
    "heldout": (config.FAITHFULNESS_HELDOUT_START, config.FAITHFULNESS_HELDOUT_SIZE),
}


def load_coupling_split(split: str) -> list[tuple[SubsetSumInstance, AdditionInstance]]:
    """Paired (subset_sum, addition) instances for the coupling faithfulness gate.

    split in {'dev','heldout'}. Both are carved from the TRAIN pools by a fixed index range
    (config.FAITHFULNESS_*); the addition instance is paired index-aligned with the subset-sum one.
    The pairing is deterministic (pools are fixed-seed on disk), so it is fully reproducible.
    """
    start, size = _FAITHFULNESS_RANGES[split]
    ss = load_subset_sum_pool("train")[start : start + size]
    add = load_addition_pool("train")[start : start + size]
    assert len(ss) == size and len(add) == size, f"train pools too small for {split} split"
    return list(zip(ss, add))
