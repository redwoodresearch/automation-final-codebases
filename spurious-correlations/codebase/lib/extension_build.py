"""Grid building for EXTENSION pairs (core partner × new feature), reusing the frozen flow:
buildability screen → buffered generation → layout-normalize → length-match/commit — with all data
namespaced away from the shipped paths (data/buffer/, data/grid/; data/grid/ is frozen).

Orientation (extension_properties convention, asserted there): partner = f1, new feature = f2.
Seed pool: the partner's (triggers are topic-neutral, seeds=() — the language precedent);
language×trigger pairs use the broad GENERAL pool (language has no seeds either).

The commit flow for a brand-new pair mirrors what the shipped study did for its pairs:
  * held cells: identical-length-histogram matched, equal-count (length_match.match_pair_tuned,
    held_gate="bounded" — the stage 6 convention for new pairs);
  * training cells A/B: full 300/class, seed-stratified balanced, length-UNMATCHED
    (length_match.seed_stratified_balanced — the shipped full_balanced treatment);
  * everything layout-normalized + re-tokenized first, exactly like the shipped grids.

Stage 1 additions (all feature-scoped; None/no-op for anything else):
  * cell_gen_kwargs(pair_key, cell_key) resolves the per-cell generation hooks — word-target
    nudge, NFC + generate-then-transform text transform (all_caps/leetspeak), the
    trailing_signature dash-check exemption, the starts_with dedup prefix-strip, and the
    numerical_density × substring_high_entropy surface-rule exemption.
  * verify_extension_grid: exact programmatic verification of a committed pair (trigger rule on
    every text of every cell, partner surface rule re-check, dash discipline, length window,
    position spread for insertion triggers). 0 failures required to ship a pair.
"""

from __future__ import annotations

import re

import config
import data_checks
import dataset
import extension_properties as EXT
import extension_validators as EV
import length_match as LM
import validation
from properties import CELLS

BUFFER_DIR = config.DATA_DIR / "buffer"
GRID_DIR = config.DATA_DIR / "grid"


def ext_pair(pair_key: str):
    """Resolve an extension pair key to (f1_prop, f2_prop, seed_pool). Partner = f1 (asserted).

    Seed ownership (stage 3 generalization). The deterministic arm's triggers are topic-neutral
    (seeds=()), so the pair borrows the PARTNER's (f1's) pool — unchanged. Content features OWN
    their scenario pool (the falsehoods / harm scenarios / medical frames / refusal requests /
    entity topics ARE the feature's ground truth), so when f2 has seeds the pair draws from f2. The
    resolution is: f2's pool if f2 owns one, else f1's, else (both topic-neutral: language×trigger)
    the broad GENERAL pool. This preserves the deterministic-arm behavior byte-identically (every
    trigger has seeds=(), so it still falls through to f1)."""
    f1_name, f2_name = EXT.split_pair_key(pair_key)
    assert pair_key == EXT.extension_pair_key(f1_name, f2_name), \
        f"{pair_key}: extension pairs must be <partner>__<new_feature>"
    f1, f2 = EXT.lookup_property(f1_name), EXT.lookup_property(f2_name)
    seed_pool = f2.seeds or f1.seeds
    if not seed_pool:
        # Neither feature owns a pool (e.g. language partner × a trigger, or × ai_self_reference):
        # use the broad GENERAL pool, the shipped language-pair precedent.
        import seeds

        seed_pool = seeds.GENERAL_SEEDS
    assert seed_pool, f"pair {pair_key} has no seed pool"
    return f1, f2, seed_pool


def _cell_signs(cell_key: str) -> tuple[str, str]:
    for key, s1, s2, _cls in CELLS:
        if key == cell_key:
            return s1, s2
    raise KeyError(cell_key)


# ------------------------------------------------------------------- per-cell generation hooks
def _word_nudge(lo: int, hi: int) -> str:
    return f"Length: aim for roughly {lo} to {hi} words (keep close to this range)."


# Measured per-(pair, cell) word-target factor overrides for transform + cells whose PARTNER POLE
# shifts the token-per-word rate (casual/code-free poles run short: contractions and short words
# blow up less under caps/leet, and there is no fenced code adding transform-exempt tokens).
# Each value retargets the cell to its held-pair sibling's measured median so the held-cell
# length-histogram match retains enough texts (screen_summary.md per-cell token stats):
#   code×caps    held_f1neg 190 vs held_f1pos 266 → 0.55×266/190 = 0.77
#   code×leet    held_f1neg 212 vs held_f1pos 266 → 0.20×266/212 = 0.25
#   formal×caps  held_f1neg 206 vs held_f1pos 239 → 0.55×239/206 = 0.64
#   formal×leet  unscreened; preset by the caps ratio (~×1.16) → 0.23, verified at build QA.
# Populated further at build QA when a pair's measured held-pair offset exceeds ~20 tokens.
PAIR_CELL_FACTOR_OVERRIDES: dict[tuple[str, str], float] = {
    ("code_presence__all_caps", "held_f1neg"): 0.77,
    ("code_presence__leetspeak", "held_f1neg"): 0.25,
    ("formal_register__all_caps", "held_f1neg"): 0.64,
    ("formal_register__leetspeak", "held_f1neg"): 0.23,
    # jargon−/caps+ inflates less than other partners' − poles (plain everyday words) while
    # jargon+ runs ~25 tok long → held gap +53, committed 21/cell at the default factor.
    # Mass-wave fix: 0.55×255/202 (retarget hneg to hpos's measured 255).
    ("domain_jargon__all_caps", "held_f1neg"): 0.69,
}


def cell_length_nudge(pair_key: str, cell_key: str) -> str:
    """Per-cell word target: the shipped baseline (170–205; Spanish cells 125–150 per the shipped
    language-pair nudges), shifted by the shared partner-pole passage adjust
    (content_build.PASSAGE_LENGTH_ADJUST — the mass wave measured temporal_scope+ extension
    passages ~+30 tok, collapsing held cells exactly as in the content/harm arms), scaled by the
    measured, screen-retuned transform factor for all_caps/leetspeak + cells (variant-aware:
    Spanish and code_presence+ cells transform differently, plus measured per-pair-cell
    overrides), plus a per-feature bonus for insertion-trigger + cells. Measurements + rationale:
    extension_properties' word-target section and results/screen_summary.md."""
    import content_build as CB  # lazy: content_build is a sibling arm (no import cycle, but
    # extension_build is imported far more widely; keep its import graph light)

    f1_name, f2_name = EXT.split_pair_key(pair_key)
    s1, s2 = _cell_signs(cell_key)
    spanish = f1_name == "language" and s1 == "+"
    lo, hi = (125, 150) if spanish else (170, 205)
    delta = CB.PASSAGE_LENGTH_ADJUST.get((f1_name, s1), 0)
    lo, hi = lo + delta, hi + delta
    if s2 == "+":
        if f2_name in EXT.WORD_TARGET_FACTORS:
            variant = ("es" if spanish
                       else "en_code" if (f1_name == "code_presence" and s1 == "+") else "en")
            f = PAIR_CELL_FACTOR_OVERRIDES.get((pair_key, cell_key),
                                               EXT.WORD_TARGET_FACTORS[f2_name][variant])
            lo, hi = round(lo * f), round(hi * f)
        elif f2_name in EXT.INSERTION_FEATURES:
            bonus = EXT.INSERTION_WORD_BONUS[f2_name]
            lo, hi = lo + bonus, hi + bonus
    return _word_nudge(lo, hi)


async def _validate_num_entropy_exempt(text, f1, f1_sign, f2, f2_sign):
    """validation.validate_text semantics with ONE documented sign-aware exemption: the pinned
    trigger string "xq7#Kp2v" contains the digits 7 and 2, which the frozen number_dense surface
    rule counts as figures — mechanically contradicting numerical_density− cells although the
    construct ("no quantitative figures") is intact (an ID code is not a quantity). The partner's
    SURFACE rule therefore sees the text with the exact pinned string stripped; the partner's
    JUDGE and the trigger's own rule see the full text. (Frozen validation.py is untouched; this
    wrapper is passed only for numerical_density × substring_high_entropy cells.)"""
    j1 = await validation._judge_or_surface_only(text, f1, f1_sign)
    j2 = await validation._judge_or_surface_only(text, f2, f2_sign)
    f1_match, f2_match = j1["label"] == f1_sign, j2["label"] == f2_sign

    stripped = re.sub(re.escape(EV.HIGH_ENTROPY), " ", text)
    surface_matches = {}
    for prop, sign, surface_text in ((f1, f1_sign, stripped), (f2, f2_sign, text)):
        if prop.surface_rule:
            surface_matches[prop.name] = validation.SURFACE_RULES[prop.surface_rule](surface_text, sign)

    return validation.ValidationResult(
        accepted=f1_match and f2_match and all(surface_matches.values()),
        f1_label=j1["label"], f2_label=j2["label"], f1_match=f1_match, f2_match=f2_match,
        surface_matches=surface_matches,
        detail={"f1_reason": j1["reason"], "f2_reason": j2["reason"]},
    )


# The partner judge for LEETSPEAK pairs, all four cells. The screen
# (results/judge_transform_screen.json) found the frozen judge (sonnet-4-6) returns a
# model-level REFUSAL (stop_reason=refusal, empty completion) on 65–95% of leet-transformed texts
# — with refusal probability depending on the partner pole, so accepting only judge-answered
# candidates would warp the cells differentially. sonnet-4-5 still refuses heavily exactly where
# the frozen per-property thinking budgets apply (hedging/passive_voice). haiku-4-5 has ZERO
# refusals and 0.92–1.00 label agreement on both poles of all 8 partners on leet surfaces
# (0.96–1.00 on originals, calibrated against the frozen judge's labels). Precedent for a
# per-feature judge substitution: the shipped study judged the `language` feature with haiku.
# Used uniformly in ALL FOUR cells so judge identity never correlates with the trigger sign.
LEET_PARTNER_JUDGE_MODEL = "claude-haiku-4-5-20251001"


def _make_partner_judge_model_validate(model: str):
    """validation.validate_text semantics with the PARTNER (f1) judged by `model` instead of the
    frozen judge (per-property thinking budgets unchanged). The trigger (f2) keeps its judge-free
    surface-rule path; all surface rules apply exactly as in validate_text."""

    async def _validate(text, f1, f1_sign, f2, f2_sign):
        j1 = await validation.judge_feature(text, f1, model=model)
        j2 = await validation._judge_or_surface_only(text, f2, f2_sign)
        f1_match, f2_match = j1["label"] == f1_sign, j2["label"] == f2_sign
        surface_matches = {}
        for prop, sign in ((f1, f1_sign), (f2, f2_sign)):
            if prop.surface_rule:
                surface_matches[prop.name] = validation.SURFACE_RULES[prop.surface_rule](text, sign)
        return validation.ValidationResult(
            accepted=f1_match and f2_match and all(surface_matches.values()),
            f1_label=j1["label"], f2_label=j2["label"], f1_match=f1_match, f2_match=f2_match,
            surface_matches=surface_matches,
            detail={"f1_reason": j1["reason"], "f2_reason": j2["reason"]},
        )

    return _validate


_validate_leet_partner_judge = _make_partner_judge_model_validate(LEET_PARTNER_JUDGE_MODEL)


def _make_transform_partner_validate(partner_judge_model: str | None):
    """Generate-then-transform extension pairs (f2 = all_caps/leetspeak, a judge-free trigger;
    f1 = the prior partner). The partner is LLM-judged on the FINAL transformed text (the all_caps
    judge and the per-partner leet judge screens both establish the judge reads the transformed
    surface), but the partner's deterministic SURFACE rule runs on the READABLE PRE-transform text.

    Why: leet maps a/e/i/o/s -> 4/3/1/0/5 and all_caps uppercases everything, which corrupts the
    partner surface regexes on the + (transformed) cells — number_dense fires on the leet digits,
    acronym_dense fires on every all-caps word, and first_person/second_person stop matching
    ("we"->"w3", "you"->"y0u"). This is the SAME failure the numerical_density ×
    substring_high_entropy exemption fixes, generalized to the two character transforms. The
    transform feature's OWN surface rule (leet_rule / all_caps_rule) still runs on the final text
    (that is what it is defined on). The screen surfaced this on 5 seg-1 pairs (all with a
    surface-ruled partner); the judge f-match was 0.96-1.00 in every cell, only `surface:` failed."""

    async def _validate(text, f1, f1_sign, f2, f2_sign, pretransform=None):
        readable = pretransform if pretransform is not None else text  # readable == text on - cells
        # partner (f1) is a judged prior; f2 (the transform) is judge-free -> its surface rule.
        j1 = await validation.judge_feature(text, f1, model=partner_judge_model)
        j2 = await validation._judge_or_surface_only(text, f2, f2_sign)
        f1_match, f2_match = j1["label"] == f1_sign, j2["label"] == f2_sign
        surface_matches = {}
        if f1.surface_rule:  # partner rule on the readable pre-transform text
            surface_matches[f1.name] = validation.SURFACE_RULES[f1.surface_rule](readable, f1_sign)
        if f2.surface_rule:  # transform feature's own rule on the final text
            surface_matches[f2.name] = validation.SURFACE_RULES[f2.surface_rule](text, f2_sign)
        return validation.ValidationResult(
            accepted=f1_match and f2_match and all(surface_matches.values()),
            f1_label=j1["label"], f2_label=j2["label"], f1_match=f1_match, f2_match=f2_match,
            surface_matches=surface_matches,
            detail={"f1_reason": j1["reason"], "f2_reason": j2["reason"]})

    _validate.needs_pretransform = True
    return _validate


def _compose(*fns):
    def _apply(text: str) -> str:
        for fn in fns:
            text = fn(text)
        return text

    return _apply


def cell_gen_kwargs(pair_key: str, cell_key: str) -> dict:
    """The generate_cell_props extension hooks for one cell of an extension pair (see the module
    docstring). Every extension cell at least NFC-normalizes its candidates."""
    f1_name, f2_name = EXT.split_pair_key(pair_key)
    _s1, s2 = _cell_signs(cell_key)

    transforms = [EV.nfc]
    gen_t = EV.GEN_TRANSFORMS.get((f2_name, s2))
    if gen_t:
        transforms.append(gen_t)

    kwargs: dict = {
        "length_nudge": cell_length_nudge(pair_key, cell_key),
        "text_transform": _compose(*transforms),
    }
    if f2_name == "trailing_signature" and s2 == "+":
        kwargs["dash_view"] = EV.strip_trailing_signature
    if f2_name == "starts_with_string" and s2 == "+":
        kwargs["dedup_view"] = EV.strip_leading_start_string
    if {f1_name, f2_name} == {"numerical_density", "substring_high_entropy"}:
        kwargs["validate_fn"] = _validate_num_entropy_exempt
    # generate-then-transform pairs (f2 = all_caps/leetspeak). If the partner (f1) has a
    # deterministic surface rule, the character transform corrupts it on the + cells, so the
    # partner rule must run on the readable pre-transform text (leet also swaps the frozen judge
    # for the screened haiku judge). Partners with no surface rule (e.g. hedging) keep the plain
    # path: judged on the final text, no surface rule to break.
    partner_has_surface = EXT.lookup_property(f1_name).surface_rule is not None
    if f2_name == "leetspeak":
        assert "validate_fn" not in kwargs
        kwargs["validate_fn"] = (_make_transform_partner_validate(LEET_PARTNER_JUDGE_MODEL)
                                 if partner_has_surface else _validate_leet_partner_judge)
    elif f2_name == "all_caps" and partner_has_surface and "validate_fn" not in kwargs:
        kwargs["validate_fn"] = _make_transform_partner_validate(None)  # default (sonnet) partner judge
    return kwargs


# ------------------------------------------------------------------- normalize + commit
def normalize_grid(grid: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Layout-normalize every record and re-tokenize, dropping records that leave the length window
    — the exact treatment the shipped grids received (see experiments/dataset/run_length_match.py's
    _normalize_grid, which is bound to the locked pair lists and so is re-stated here for
    extension grids)."""
    for cell, recs in list(grid.items()):
        texts = [data_checks.normalize_layout(r["text"]) for r in recs]
        toks = data_checks.num_qwen_tokens_batch(texts)
        kept = []
        for r, t, n in zip(recs, texts, toks):
            r["text"], r["num_tokens"] = t, n
            if "text_pretransform" in r:  # keep the stored pre-transform source comparable
                r["text_pretransform"] = data_checks.normalize_layout(r["text_pretransform"])
            if config.LENGTH_MIN_TOKENS <= n <= config.LENGTH_MAX_TOKENS:
                kept.append(r)
        grid[cell] = kept
    return grid


def commit_extension_pair(pair_key: str, target: int = config.TEXTS_PER_CELL, write: bool = True) -> dict:
    """Buffer → normalized → (matched held cells, full-balanced A/B) → data/grid/. Returns the
    per-pair report (length/topic held-cell checks incl. permutation ceilings, seed overlap,
    per-cell stats)."""
    grid = normalize_grid(dataset.load_grid(pair_key, base_dir=BUFFER_DIR))
    _f1, f2, _seeds = ext_pair(pair_key)
    dedup_view = EV.strip_leading_start_string if f2.name == "starts_with_string" else None
    _committed, report = LM.commit_grid_flow(pair_key, grid, target, dedup_view=dedup_view,
                                             write=write, grid_dir=GRID_DIR)
    return report


# ------------------------------------------------------------------- exact grid verification
def verify_extension_grid(pair_key: str, base_dir=GRID_DIR) -> dict:
    """Exact programmatic verification of a committed extension pair, 0 failures required:
      * the trigger's sign-aware rule holds for every text of every cell (presence/
        well-formedness in + cells, absence in − cells — incl. NFC/ignorable hygiene);
      * the partner's deterministic surface rule (where it has one) re-verified, with the
        documented numerical_density × substring_high_entropy strip exemption;
      * dash discipline: no em/en dash anywhere, except exactly the trailing signature in
        trailing_signature + cells;
      * every text inside the frozen token window;
      * position spread reported for the varied-position insertion triggers."""
    import numpy as np

    grid = dataset.load_grid(pair_key, base_dir=base_dir)
    f1_name, f2_name = EXT.split_pair_key(pair_key)
    f1_prop, f2_prop = EXT.lookup_property(f1_name), EXT.lookup_property(f2_name)
    f2_rule = validation.SURFACE_RULES[f2_prop.surface_rule]
    f1_rule = validation.SURFACE_RULES[f1_prop.surface_rule] if f1_prop.surface_rule else None
    strip_entropy = {f1_name, f2_name} == {"numerical_density", "substring_high_entropy"}
    # generate-then-transform pairs: the character transform corrupts the partner surface rule on
    # the + cells, so re-verify the partner rule on the stored readable pre-transform text (the
    # generalized numerical×entropy exemption; matches _make_transform_partner_validate).
    transform_pair = f2_name in ("all_caps", "leetspeak")
    needle = EV.POSITION_NEEDLES.get(f2_name)

    failures = []
    rel_positions, sent_positions = [], []
    for cell in LM.ALL:
        for r in grid[cell]:
            text, ci = r["text"], r["candidate_index"]
            if not f2_rule(text, r["f2_sign"]):
                failures.append((cell, ci, f"trigger_rule_{r['f2_sign']}"))
            if f1_rule is not None:
                if strip_entropy:
                    surface_text = re.sub(re.escape(EV.HIGH_ENTROPY), " ", text)
                elif transform_pair and "text_pretransform" in r:
                    surface_text = r["text_pretransform"]  # readable; partner rule was validated here
                else:
                    surface_text = text
                if not f1_rule(surface_text, r["f1_sign"]):
                    failures.append((cell, ci, f"partner_rule_{r['f1_sign']}"))
            dash_text = (EV.strip_trailing_signature(text)
                         if (f2_name == "trailing_signature" and r["f2_sign"] == "+") else text)
            if data_checks.has_forbidden_dash(dash_text):
                failures.append((cell, ci, "dash"))
            if not (config.LENGTH_MIN_TOKENS <= r["num_tokens"] <= config.LENGTH_MAX_TOKENS):
                failures.append((cell, ci, f"length={r['num_tokens']}"))
            if needle and r["f2_sign"] == "+":
                pos = EV.trigger_sentence_index(text, needle)
                n_sent = len(EV.sentences(text))
                if pos is not None:
                    sent_positions.append(pos)
                    rel_positions.append(pos / max(1, n_sent - 1))

    report = {
        "pair": pair_key,
        "n_per_cell": {c: len(grid[c]) for c in LM.ALL},
        "n_failures": len(failures),
        "failures": failures[:20],
        "ok": not failures,
    }
    if rel_positions:
        rel = np.array(rel_positions)
        report["plus_position_rel"] = {
            "mean": float(rel.mean()), "p10": float(np.percentile(rel, 10)),
            "p50": float(np.percentile(rel, 50)), "p90": float(np.percentile(rel, 90)),
        }
        report["plus_sentence_index_hist"] = {
            str(k): int(v) for k, v in zip(*np.unique(sent_positions, return_counts=True))
        }
    return report
