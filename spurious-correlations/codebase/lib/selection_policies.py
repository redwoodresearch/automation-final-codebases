"""Selection-time policies for the stage 6 expansion dataset — the audit-fix levers
(desiderata "Stage 6 expansion", remedy preference order: selection-time > carve-out > prompt edit).

Two layers, both deterministic and committed (the saved-selection requirement):

1. UNIVERSAL pool filters, applied to every EXPANSION pair's layout-normalized buffer pool before
   matching (the original 32 pairs are never touched):
   a. opener near-duplicate removal — the side run's 200-char-prefix dedup admitted within-cell and
      cross-cell (train<->held) near-dup openers (audit c). Within a cell, near-dup clusters keep the
      lowest-stable-rank text; a train text whose opener near-dups a held text is dropped (held cells
      are the scarcer, metric-bearing side).
   b. acronym authenticity (audit a) — in acronym+ cells, texts with <2 DISTINCT initialisms are
      excluded (a repeated single initialism passes the accept-time >=3-token rule but is a degenerate
      realization); texts containing a BLOCKLISTED (LLM-spot-checked fabricated) initialism are
      excluded in every cell.

2. TARGETED per-pair policies (POLICIES below), added where the audit round flags a cell; empty by
   default. Supported levers:
   - "held_marker_match": [property, ...] — held cells are matched to an identical JOINT
     (length_bin, marker level, ...) histogram, so the flagged third-property marker can no longer
     separate them (audit d; e.g. acronym-glossing parentheses).
   - "prefer_against_entities": [entity, ...] — records containing any listed entity are deprioritized
     at selection in all four cells (audit b, e.g. EPA/FDA/ISO/NIST concentration; the standing
     diversity thresholds), drawn only when a bin/seed's preferred pool runs out.

Used by run_length_match.py for expansion pairs only. Default = only the universal filters.
"""
from __future__ import annotations

import difflib
import json
import re
from collections import defaultdict

import config
import data_checks
import generation
import markers
from length_match import AB, HELD, _stable_rank
from properties import PAIRS

BLOCKLIST_PATH = config.RESULTS_DIR / "acronym_blocklist.json"
_INITIALISM = re.compile(r"\b[A-Z]{2,}\b")  # the accept-time acronym counter's token rule
DUP_RATIO = config.DEDUP_RATIO_THRESHOLD  # 0.80, the standing gate threshold


# ---------------------------------------------------------------------- targeted per-pair policies
# pair_key -> policy dict. Derived by scripts_expand_dataset_completion/derive_marker_policies.py from
# the audit round (evidence in results/marker_fix_decisions.json) and COMMITTED as
# results/selection_policies_expansion.json — the saved, deterministic selection input. The in-code
# _MANUAL_POLICIES layer is for hand-made additions/overrides (kept explicit in git).
POLICIES_PATH = config.RESULTS_DIR / "selection_policies_expansion.json"
_MANUAL_POLICIES: dict[str, dict] = {}


def _load_policies() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if POLICIES_PATH.exists():
        out.update(json.loads(POLICIES_PATH.read_text())["policies"])
    for pk, pol in _MANUAL_POLICIES.items():
        out.setdefault(pk, {}).update(pol)
    return out


POLICIES: dict[str, dict] = _load_policies()


def held_level_key(pair_key: str):
    pol = POLICIES.get(pair_key, {})
    props = pol.get("held_marker_match", [])          # matched at LEVEL granularity (0/1/2/3+)
    bit_props = pol.get("held_marker_match_bit", [])  # matched at PRESENCE-bit granularity
    if not props and not bit_props:
        return None
    f1, f2 = markers.pair_props(pair_key)
    assert all(p not in (f1, f2) for p in props + bit_props), \
        f"{pair_key}: marker-match on an IN-PAIR feature would erase the feature itself: {props + bit_props}"
    return lambda r: (tuple(markers.marker_level(p, r["text"]) for p in props)
                      + tuple(markers.marker_bit(p, r["text"]) for p in bit_props))


def prefer_key(pair_key: str):
    """Static de-prioritization: +1 penalty per over-concentrated entity contained, +1 if the text
    opens with a flagged template opener (case-insensitive first-words match)."""
    pol = POLICIES.get(pair_key, {})
    entities = pol.get("prefer_against_entities", [])
    openers = [o.lower() for o in pol.get("prefer_against_openers", [])]
    if not entities and not openers:
        return None
    pats = [re.compile(rf"\b{re.escape(e)}\b") for e in entities]

    def key(r):
        pen = sum(bool(p.search(r["text"])) for p in pats)
        if openers:
            head = " ".join(r["text"].split()[:6]).lower()
            pen += sum(head.startswith(o) for o in openers)
        return pen

    return key


def describe(pair_key: str) -> dict:
    return dict(POLICIES.get(pair_key, {}))


# ---------------------------------------------------------------------- universal pool filters
def _blocklist() -> set[str]:
    if BLOCKLIST_PATH.exists():
        return set(json.loads(BLOCKLIST_PATH.read_text())["blocked"])
    return set()


def distinct_initialisms(text: str) -> set[str]:
    return set(_INITIALISM.findall(text))


def _acro_plus_cells(pair_key: str) -> list[str]:
    """Cells where acronym_density is '+' for this pair ([] if the pair doesn't involve it)."""
    f1, f2 = markers.pair_props(pair_key)
    if "acronym_density" not in (f1, f2):
        return []
    out = []
    for cell in ("A", "B", "held_f1pos", "held_f1neg"):
        s1, s2 = generation._cell_signs(cell)
        if (s1 if f1 == "acronym_density" else s2) == "+":
            out.append(cell)
    return out


def _opener_dup_pairs(norms_a: list[str], norms_b: list[str] | None = None) -> list[tuple[int, int]]:
    """Index pairs whose normalized 200-char prefixes near-duplicate (difflib ratio >= DUP_RATIO).
    norms_b=None -> within one list (i<j); else cross-list (i in a, j in b). Candidate pruning mirrors
    the side run's dup scan: shared first-12-chars bucket OR prefix token-set Jaccard >= 0.55."""
    within = norms_b is None
    if within:
        norms_b = norms_a
    toks_a = [set(s.split()) for s in norms_a]
    toks_b = toks_a if within else [set(s.split()) for s in norms_b]
    cand = set()
    byhead: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for side, norms in ((0, norms_a), (1, norms_b)) if not within else ((0, norms_a),):
        for i, s in enumerate(norms):
            byhead[s[:12]].append((side, i))
    for members in byhead.values():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                (sa, ia), (sb, ib) = members[x], members[y]
                if within:
                    cand.add((min(ia, ib), max(ia, ib)))
                elif sa != sb:
                    cand.add((ia, ib) if sa == 0 else (ib, ia))
    for i, ti in enumerate(toks_a):
        if len(ti) < 3:
            continue
        for j, tj in enumerate(toks_b):
            if within and j <= i:
                continue
            inter = len(ti & tj)
            if inter and inter / len(ti | tj) >= 0.55:
                cand.add((i, j))
    out = []
    for i, j in sorted(cand):
        r = max(difflib.SequenceMatcher(None, norms_a[i], norms_b[j]).ratio(),
                difflib.SequenceMatcher(None, norms_b[j], norms_a[i]).ratio())
        if r >= DUP_RATIO:
            out.append((i, j))
    return out


def _within_cell_opener_dedup(recs: list[dict]) -> tuple[list[dict], int]:
    """Collapse within-cell opener near-dup clusters to their lowest-stable-rank member."""
    norms = [data_checks.normalized_prefix(r["text"]) for r in recs]
    parent = list(range(len(recs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in _opener_dup_pairs(norms):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(len(recs)):
        clusters[find(i)].append(i)
    keep = set()
    for members in clusters.values():
        keep.add(min(members, key=lambda i: _stable_rank(recs[i]["text"])))
    kept = [r for i, r in enumerate(recs) if i in keep]
    return kept, len(recs) - len(kept)


def apply_pool_filters(pair_key: str, grid: dict[str, list[dict]],
                       mutable_cells: list[str], held_reference: dict[str, list[dict]] | None = None
                       ) -> tuple[dict[str, list[dict]], dict]:
    """Filter the buffer pools of `mutable_cells` (already layout-normalized). Returns (grid, info).

    held_reference: the held-cell texts to test train<->held opener dups against. For the tuned
    (held-creating) run pass the held POOLS (conservative superset); for full_balanced pass the
    COMMITTED held cells. Never applied to original-32 pairs."""
    assert pair_key not in PAIRS, f"selection policies must never touch an original pair: {pair_key}"
    info: dict = {"pair": pair_key}
    blocked = _blocklist()
    acro_plus = _acro_plus_cells(pair_key)

    for cell in mutable_cells:
        recs = grid[cell]
        n0 = len(recs)
        if blocked:
            recs = [r for r in recs if not (distinct_initialisms(r["text"]) & blocked)]
        n_block = n0 - len(recs)
        n_weak = 0
        if cell in acro_plus:
            kept = [r for r in recs if len(distinct_initialisms(r["text"])) >= 2]
            n_weak = len(recs) - len(kept)
            recs = kept
        recs, n_dup = _within_cell_opener_dedup(recs)
        grid[cell] = recs
        info[cell] = {"n_pool": n0, "blocklisted": n_block, "weak_acro": n_weak, "opener_dups": n_dup}

    if held_reference is not None:
        held_norms = [data_checks.normalized_prefix(r["text"])
                      for c in HELD for r in held_reference.get(c, [])]
        for cell in [c for c in mutable_cells if c in AB]:
            recs = grid[cell]
            norms = [data_checks.normalized_prefix(r["text"]) for r in recs]
            drop = {i for i, _ in _opener_dup_pairs(norms, held_norms)}
            grid[cell] = [r for i, r in enumerate(recs) if i not in drop]
            info[cell]["cross_held_dups"] = len(drop)
    return grid, info
