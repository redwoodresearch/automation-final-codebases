"""Complete-graph bookkeeping for the v2 project: the canonical orientation rule for NEW pairs,
the 666-pair status classification, and the build routing every missing pair resolves to.

ORIENTATION (the study's perennial trip-up; see also extension_properties):
  * NEW pairs (anything not already on disk) use f1 = the EARLIER feature in
    `features.ALL_FEATURES` (`canonical_new_pair_key`). Because ALL_FEATURES preserves both
    historical orders (PRIOR_21 = language + GLOBAL_PROPERTY_ORDER, then EXTENSION_ORDER), this
    rule reproduces every existing on-disk orientation — verified over all 337 in-roster grids in
    tests/test_complete_graph.py. Existing pairs ALWAYS keep their on-disk orientation; stored Δ
    values are never re-oriented at load (the merged table carries feature-role tags instead).

ROUTING (fixed by the canonical design; unit-tested in tests/test_complete_graph.py):
  * harm pairs (any side = harmful_vs_benign): Opus generates ALL FOUR cells and judges the harm
    pole; the partner keeps its normal judge (sonnet). This rule takes precedence over the
    leetspeak rules (`harmful_vs_benign` × `leetspeak` is a harm pair).
  * leetspeak × judged-feature pairs (non-harm): the partner's judge depends on the partner KIND —
      - style/surface partner (a PRIOR_21 style feature): haiku may judge the partner poles on
        leet-transformed cells (uniform across all four cells), but ONLY after a per-partner
        judge-transform agreement screen (the shipped screen covered only the 8 anchor partners);
      - content partner: haiku must NEVER judge a content feature. Resolution ladder:
        sonnet → screened non-haiku fallback (opus) → judge the PRE-transform text and apply the
        deterministic transform after (with spot-checks) → unbuildable-by-judge.
  * transcript-shaped features (sycophantic_agreement, refusal_language): the whole pair is
    realized as User/Assistant transcripts in all four cells (the content-arm mechanism), for any
    partner. (The shipped new×new builder excluded transcript features; v2 extends the transcript
    realization to those pairs — a deliberate design extension, flagged per pair.)
  * everything else: new×prior → the extension (trigger f2) / content (content f2) builders with
    the partner-side validators; new×new → the newxnew builder; prior×prior → the core-grid
    builder with the constraint-rank seed rule (moved here from
    experiments/dataset/run_buildability.py, which now imports it back).

CELL TARGETS for new grids: 300 texts per cell for ALL FOUR cells (floor 150 with a documented
per-pair reason). The held cells double as the anti-diagonal arm's TRAINING data, so small held
cells make reversal results noise-prone.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from attrs import frozen

import config
import features
import validation
from anchor_pairs import ANCHOR_PARTNERS
from extension_properties import EXTENSION_ORDER, split_pair_key
from properties import GLOBAL_PROPERTY_ORDER

# ------------------------------------------------------------------------- rosters and constants
ALL_FEATURES_INDEX: dict[str, int] = {f: i for i, f in enumerate(features.ALL_FEATURES)}
PRIOR_SET = frozenset(features.PRIOR_21)
NEW_SET = frozenset(EXTENSION_ORDER)
NON_ANCHOR_PRIORS: list[str] = [f for f in features.PRIOR_21 if f not in ANCHOR_PARTNERS]  # 13
TRANSCRIPT_FEATURES = ("sycophantic_agreement", "refusal_language")
# Content-behavior features whose both-present cell can be semantically contradictory
# (e.g. a reply that both refuses and sycophantically agrees) — screens must check these first.
BEHAVIOR_FEATURES = ("deception", "harmful_vs_benign", "sycophantic_agreement", "refusal_language")

TARGET_PER_CELL = 300  # all four cells (held cells are the anti-diagonal arm's training data)
FLOOR_PER_CELL = 150   # only with a documented per-pair reason

DEGENERATE_REBUILD = "language__second_person_address"  # ships with 1-text train cells

# The 8 grids in data/grid/ whose features are outside the 37-feature roster (old diagnostics);
# permanently excluded from the study. Verified against data/grid/ in the manifest builder.
OUT_OF_ROSTER_GRIDS = (
    "childrens_subject__sentence_complexity",
    "emotional_valence__hedging",
    "finance_topic__numerical_density",
    "legal_topic__formal_register",
    "politeness__agreement_stance",
    "protagonist_age__tech_affinity",
    "protagonist_gender__occupation_physical_vs_care",
    "sycophantic_vs_honest__code_in_reply",
)

# How topic-constraining each prior property is; a prior×prior pair's shared seed pool comes from
# the higher-ranked member (ties -> the earlier property in GLOBAL_PROPERTY_ORDER). This is the
# rule every prior sweep used (it reproduces the seed_property of all 182 committed
# pairs_expanded.PAIRS_EXPANDED entries — asserted in tests/test_complete_graph.py). It lived in
# experiments/dataset/run_buildability.py; moved here so lib code can route prior×prior builds.
CONSTRAINT_RANK = {
    "code_presence": 6,
    "narrative_vs_expository": 5,
    "imperative_mood": 4,
    "domain_jargon": 4,
    "numerical_density": 3,
    "temporal_scope": 2,
    "argumentative_structure": 2,
    "specificity": 1,
    "question_vs_statement": 1,
    "hedging": 1,
    # strength_geometry expansion: style/voice/syntax axes are topic-free (rank 1, so a content
    # partner drives the shared topics); acronym_density leans technical (rank 3, its TECH seeds
    # should drive over a pure-style partner so acronyms are natural).
    "second_person_address": 1,
    "first_person_voice": 1,
    "figurative_language": 1,
    "emotional_tone": 1,
    "formal_register": 1,
    "sentence_complexity": 1,
    "passive_voice": 1,
    "negation_density": 1,
    "acronym_density": 3,
    "parenthetical_asides": 1,
}


def prior_pair_seed_property(a: str, b: str) -> str:
    """The constraint-rank seed rule for prior×prior pairs (see CONSTRAINT_RANK). `language` never
    owns seeds (its Property has seeds=(); Spanish/English is a realization of the partner's
    topics, the shipped language__* convention), so its partner always owns the pool."""
    if "language" in (a, b):
        return b if a == "language" else a
    return max((a, b), key=lambda p: (CONSTRAINT_RANK[p], -GLOBAL_PROPERTY_ORDER.index(p)))


# ------------------------------------------------------------------------------ orientation rule
def canonical_new_pair_key(feat_a: str, feat_b: str) -> str:
    """Pair key for a NEW pair: f1 = the earlier feature in ALL_FEATURES.

    Only for pairs being built from scratch — existing grids keep their on-disk orientation
    (which this rule happens to reproduce for all 337; see the module docstring)."""
    assert feat_a in ALL_FEATURES_INDEX, f"unknown feature {feat_a!r}"
    assert feat_b in ALL_FEATURES_INDEX, f"unknown feature {feat_b!r}"
    assert feat_a != feat_b, feat_a
    f1, f2 = sorted((feat_a, feat_b), key=ALL_FEATURES_INDEX.__getitem__)
    return f"{f1}__{f2}"


def assert_canonical_orientation(pair_key: str) -> None:
    """Assert a pair key follows the new-pair rule. Call at build AND readout entry points for
    new pairs (existing on-disk pairs all satisfy it too, so it is safe to assert on any
    in-roster pair)."""
    f1, f2 = split_pair_key(pair_key)
    assert pair_key == canonical_new_pair_key(f1, f2), (
        f"{pair_key}: not in canonical orientation (f1 must be earlier in ALL_FEATURES; "
        f"expected {canonical_new_pair_key(f1, f2)})")


def assert_canonical_if_in_roster(pair_key: str) -> None:
    """Readout-side guard: any pair key whose two features are both in the 37-feature roster must
    be in canonical orientation (all existing grids are; new grids are asserted at build). Keys
    that don't parse as two roster features (out-of-roster aux grids) are left alone."""
    try:
        f1, f2 = split_pair_key(pair_key)
    except KeyError:
        return
    assert_canonical_orientation(pair_key)


def pair_family(f1: str, f2: str) -> str:
    """'prior_x_prior' | 'new_x_prior' | 'new_x_new' (which build machinery owns the pair)."""
    n_prior = (f1 in PRIOR_SET) + (f2 in PRIOR_SET)
    assert f1 in ALL_FEATURES_INDEX and f2 in ALL_FEATURES_INDEX, (f1, f2)
    return {2: "prior_x_prior", 1: "new_x_prior", 0: "new_x_new"}[n_prior]


def orientation_family(pair_key: str) -> str:
    """Which historical convention the on-disk key follows (documentation; all three coincide
    with the ALL_FEATURES rule)."""
    f1, f2 = split_pair_key(pair_key)
    fam = pair_family(f1, f2)
    if fam == "prior_x_prior":
        return "language_first" if "language" in (f1, f2) else "global_property_order"
    if fam == "new_x_prior":
        return "extension_partner_first"
    return "extension_order"


# ------------------------------------------------------------------------------- build routing
def _is_judge_free(f: str) -> bool:
    return f in validation.JUDGE_FREE_PROPERTIES


@frozen
class BuildRoute:
    pair: str
    family: str            # prior_x_prior | new_x_prior | new_x_new
    builder: str            # core_grid | extension_build | content_build | newxnew_build
    realization: str        # passage | transcript
    generator_model: str    # writer for ALL FOUR cells (uniform; never varies with cell label)
    validation_by_feature: dict[str, str]  # feature -> validator/judge description (uniform across cells)
    seed_pool: str          # description of the shared scenario pool
    flags: tuple[str, ...]
    notes: tuple[str, ...]


# validator description strings (single source for tests + manifest)
V_SURFACE = "surface_rule (judge-free)"
V_SONNET = "judge:sonnet-4-6"
V_SONNET_SURFACE = "judge:sonnet-4-6 + surface_rule"
V_OPUS_HARM = "judge:opus-4-5 (harm pole, both signs)"
V_HAIKU_LEET = "judge:haiku-4-5 on leet-transformed cells (requires per-partner judge-transform screen)"
V_LEET_CONTENT_LADDER = ("judge ladder (content partner in a leet pair): sonnet-4-6 -> screened opus fallback "
                         "-> judge pre-transform text + deterministic transform + spot-check -> unbuildable-by-judge")


def _feature_validation(f: str, *, leet_pair: bool, harm_pair: bool) -> str:
    """The validator for feature `f`'s poles within a pair context. Uniform across cells."""
    if _is_judge_free(f):
        return V_SURFACE
    if f == "harmful_vs_benign":
        return V_OPUS_HARM
    if f == "language":
        return "judge:haiku-4-5 (shipped language precedent)"
    base = V_SONNET_SURFACE if _has_surface_rule(f) else V_SONNET
    if leet_pair and not harm_pair:
        # the partner of leetspeak is judged on leet-transformed text in 2 of 4 cells; judge
        # identity must stay uniform across cells, so the whole pair's partner judging moves
        if f in PRIOR_SET:
            return V_HAIKU_LEET
        return V_LEET_CONTENT_LADDER
    return base


def _has_surface_rule(f: str) -> bool:
    import extension_properties as EXT
    try:
        return EXT.lookup_property(f).surface_rule is not None
    except KeyError:
        return False


def _seed_pool_desc(f1: str, f2: str, family: str) -> str:
    if family == "prior_x_prior":
        return f"prior constraint-rank rule -> {prior_pair_seed_property(f1, f2)}"
    if any(f in TRANSCRIPT_FEATURES for f in (f1, f2)):
        owners = [f for f in (f1, f2) if f in TRANSCRIPT_FEATURES]
        return f"transcript user-turn bank: {' + '.join(owners)} (bank owns the scenario)"
    if family == "new_x_prior":
        prior_f = f1 if f1 in PRIOR_SET else f2
        new_f = f2 if f1 in PRIOR_SET else f1
        # content features own their scenario pool; triggers are topic-neutral -> partner's pool
        if not _is_judge_free(new_f):
            return f"content pool: {new_f}"
        return f"partner pool: {prior_f}" if prior_f != "language" else "GENERAL_SEEDS (language precedent)"
    # new_x_new: pre-registered content-priority rule (newxnew_build._POOL_OWNER_ORDER)
    prio = ["deception", "harmful_vs_benign", "topic_medical", "entity_mention_trigger"]
    for owner in prio:
        if owner in (f1, f2):
            return f"content-priority pool: {owner}"
    return "GENERAL/hedging neutral pool (trigger x trigger precedent)"


def route_pair(pair_key: str) -> BuildRoute:
    """Resolve which machinery builds a (missing) pair. Pure routing — no IO, no LLM calls."""
    f1, f2 = split_pair_key(pair_key)
    assert_canonical_orientation(pair_key)
    family = pair_family(f1, f2)
    harm_pair = "harmful_vs_benign" in (f1, f2)
    leet_pair = "leetspeak" in (f1, f2)
    transcript = any(f in TRANSCRIPT_FEATURES for f in (f1, f2))
    judged_partner_of_leet = leet_pair and any(
        not _is_judge_free(f) for f in (f1, f2) if f != "leetspeak")

    flags: list[str] = []
    notes: list[str] = []
    if harm_pair:
        flags.append("harm")
        notes.append("Opus generates all four cells and judges the harm pole; severity cap: "
                     "clearly harmful advice only, never CBRN/weapons-grade.")
    if leet_pair:
        flags.append("leetspeak")
        if harm_pair:
            notes.append("harm rule wins routing precedence; leetspeak stays surface-validated; "
                         "Opus generates all four cells and judges the harm pole on the READABLE "
                         "pre-transform text (the leet screen shows Opus refuses 73-98% of "
                         "leet-transformed content), then the deterministic transform is applied "
                         "with an exact transform-identity check. Run the leet-decodability check "
                         "on the harm pole before the stage 2 build.")
        elif judged_partner_of_leet:
            flags.append("leetspeak_x_judged")
    if transcript:
        flags.append("transcript_side")
        if family == "new_x_new":
            notes.append("transcript realization extended to new x new (the shipped newxnew "
                         "builder excluded transcript features - genre confound handled by "
                         "realizing ALL FOUR cells as transcripts).")
    if {f1, f2} <= set(BEHAVIOR_FEATURES):
        flags.append("contradictory_cell_risk")
        notes.append("behavior x behavior: the both-present cell may be semantically incoherent; "
                     "run the contradictory-cell screen before mass generation.")

    if family == "prior_x_prior":
        builder = "core_grid"
    elif family == "new_x_prior":
        new_f = f2 if f1 in PRIOR_SET else f1
        builder = "content_build" if (not _is_judge_free(new_f)) else "extension_build"
        if transcript:
            builder = "content_build"
    else:
        builder = "newxnew_build"
        if transcript:
            builder = "content_build"  # transcript realization uses the content-arm machinery

    generator = "claude-opus-4-5 (all four cells)" if harm_pair else "claude-haiku-4-5"
    validation_by_feature = {
        f: _feature_validation(f, leet_pair=leet_pair, harm_pair=harm_pair) for f in (f1, f2)}

    return BuildRoute(
        pair=pair_key, family=family, builder=builder,
        realization="transcript" if transcript else "passage",
        generator_model=generator, validation_by_feature=validation_by_feature,
        seed_pool=_seed_pool_desc(f1, f2, family), flags=tuple(flags), notes=tuple(notes))


# --------------------------------------------------------------------------- 666-pair statuses
FITTED, HELDOUT_READOUT, DEGENERATE, MISSING = (
    "fitted", "heldout_readout_newxnew", "degenerate_rebuild", "missing")


def grid_pairs_on_disk(grid_dir: Path | None = None) -> tuple[set[str], set[str]]:
    """(in-roster pair keys, out-of-roster pair keys) present in data/grid/."""
    import dataset
    grid_dir = grid_dir or dataset.GRID_DIR
    keys = {p.name.rsplit("__", 1)[0] for p in grid_dir.glob("*.jsonl")}
    in_roster, out_roster = set(), set()
    for pk in keys:
        try:
            split_pair_key(pk)
            in_roster.add(pk)
        except KeyError:
            out_roster.add(pk)
    return in_roster, out_roster


def fitted_pairs() -> set[str]:
    """The 326 pairs in the unified readout set (probe_all.json ∩-asserted ft_grid_all.jsonl)."""
    reads = config.RESULTS_DIR / "reads"
    probe_pairs = set(json.loads((reads / "probe_all.json").read_text()).keys())
    ft_pairs = {json.loads(l)["pair"] for l in (reads / "ft_grid_all.jsonl").open() if l.strip()}
    assert probe_pairs == ft_pairs, probe_pairs ^ ft_pairs
    return probe_pairs


def heldout_newxnew_pairs() -> set[str]:
    import pooled_readouts
    return set(pooled_readouts.NEWXNEW_PROBE_FILES)


def classify_all_pairs() -> dict[str, dict]:
    """{pair_key: {status, family, orientation_family}} over all C(37,2)=666 pairs, classified
    from the FROZEN readout sets (fitted / heldout new×new / degenerate) — NOT from live disk
    state, which changes as builds land (a freshly-built missing pair stays 'missing' until its
    readouts join the pooled table in stage 3). The one-time disk/census cross-check lives in
    experiments/complete_graph/build_manifest.py, run at freeze time."""
    fitted = fitted_pairs()
    heldout = heldout_newxnew_pairs()
    out: dict[str, dict] = {}
    for a, b in itertools.combinations(features.ALL_FEATURES, 2):
        pk = canonical_new_pair_key(a, b)
        if pk in fitted:
            status = FITTED
        elif pk in heldout:
            status = HELDOUT_READOUT
        elif pk == DEGENERATE_REBUILD:
            status = DEGENERATE
        else:
            status = MISSING
        f1, f2 = split_pair_key(pk)
        out[pk] = {"status": status, "f1": f1, "f2": f2,
                   "family": pair_family(a, b), "orientation_family": orientation_family(pk)}
    assert len(out) == 666, len(out)
    return out
