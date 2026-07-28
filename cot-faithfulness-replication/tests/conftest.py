"""Test collection config.

Some tests exercise prompt construction against datasets that are large or deliberately not
committed: the released CoT Faithfulness dataset (fetch with `python data/download_dataset.py`)
and GPQA-Diamond (fetch with `python data/download_gpqa.py`; not re-hosted here because its
authors keep it out of scraped training corpora). When a dataset is absent, the modules that
need it are skipped rather than failing collection.
"""

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent.parent / "data"
DATASET_SENTINEL = DATA_DIR / "faithfulness" / "faithfulness" / "suggestion_True.jsonl"
GPQA_SENTINEL = DATA_DIR / "gpqa" / "gpqa_diamond.jsonl"
NEEDS_DATASET = {"test_tier2.py", "test_judge_variants.py", "test_gpqa.py", "test_frontier.py"}
NEEDS_GPQA = {"test_gpqa.py", "test_frontier.py"}


def pytest_collection_modifyitems(config, items):
    skips = []
    if not DATASET_SENTINEL.exists():
        skips.append((NEEDS_DATASET, pytest.mark.skip(
            reason="released dataset not downloaded (run: python data/download_dataset.py)")))
    if not GPQA_SENTINEL.exists():
        skips.append((NEEDS_GPQA, pytest.mark.skip(
            reason="GPQA-Diamond not downloaded (run: python data/download_gpqa.py)")))
    if not skips:
        return
    for item in items:
        name = Path(item.fspath).name
        for needs, mark in skips:
            if name in needs:
                item.add_marker(mark)
