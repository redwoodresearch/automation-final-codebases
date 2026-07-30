"""Load-bearing invariant: the train and eval pools are disjoint, and the faithfulness-gate
dev/held-out coupling splits do not overlap. A leak here would let a "held-out" evaluation reuse a
trained-on instance, quietly inflating every reported number.

Pure-logic (reads the committed pools under data/; no sampling, no Tinker)."""

from __future__ import annotations

from lib import config
from lib.pools import (
    load_addition_pool,
    load_coupling_split,
    load_subset_sum_pool,
)


def test_subset_sum_train_eval_disjoint():
    train = {i.key() for i in load_subset_sum_pool("train")}
    eval_ = {i.key() for i in load_subset_sum_pool("eval")}
    assert train and eval_
    assert not (train & eval_), "subset-sum train/eval pools overlap"


def test_addition_train_eval_disjoint():
    train = {i.key() for i in load_addition_pool("train")}
    eval_ = {i.key() for i in load_addition_pool("eval")}
    assert train and eval_
    assert not (train & eval_), "addition train/eval pools overlap"


def test_faithfulness_dev_heldout_index_ranges_disjoint():
    dev_start, dev_size = config.FAITHFULNESS_DEV_START, config.FAITHFULNESS_DEV_SIZE
    ho_start, ho_size = config.FAITHFULNESS_HELDOUT_START, config.FAITHFULNESS_HELDOUT_SIZE
    dev_idx = set(range(dev_start, dev_start + dev_size))
    ho_idx = set(range(ho_start, ho_start + ho_size))
    assert not (dev_idx & ho_idx), "dev and held-out coupling splits share indices"


def test_coupling_splits_are_disjoint_instances():
    dev = load_coupling_split("dev")
    heldout = load_coupling_split("heldout")
    dev_keys = {(ss.key(), add.key()) for ss, add in dev}
    ho_keys = {(ss.key(), add.key()) for ss, add in heldout}
    assert dev_keys and ho_keys
    assert not (dev_keys & ho_keys), "dev and held-out coupling pairs overlap"
