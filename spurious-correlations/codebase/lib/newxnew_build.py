"""Grid building for NEW×NEW pairs (stage 5) — both features from the extension roster, keyed
`<f1>__<f2>` with earlier-in-EXTENSION_ORDER = f1 (extension_properties.new_new_pair_key), so
Δ > 0 = follows f1.

Generalizes extension_build/content_build's per-cell machinery for pairs where EITHER side may be
a deterministic trigger (surface-rule validated, judge-free), a transform (generate-then-
transform), or a judged content feature. Data is namespaced away from all committed grids:
data/buffer/ and data/grid/ (the held-out test set stays visibly
separate).

Seed-pool resolution (pre-registered in results/newxnew_predictions.json before any
generation): trigger×trigger — and any pair where neither side owns seeds — borrows the hedging
partner's 74-seed neutral topic pool (the deterministic arm's precedent); otherwise the
more-constrained content feature's scenario pool owns the pair (deception > harm > medical >
entity, i.e. the bank whose per-seed ground truth pins the most).

Standing amendments carried over (uniform across all four cells, never correlated with a sign):
harm pairs are Opus-generated in every cell with Opus judging both harm poles; leet pairs with a
judged feature would need the haiku substitution — none is selected/spared, so that path raises.
"""

from __future__ import annotations

import config
import content_build as CB  # noqa: F401 — also registers the entity/ai_self diversify hooks
import content_validators as CV
import data_checks
import dataset
import extension_build as EB
import extension_properties as EXT
import extension_validators as EV
import length_match as LM
import validation
from properties import CELLS, Property

BUFFER_DIR = config.DATA_DIR / "buffer"
GRID_DIR = config.DATA_DIR / "grid"

TRANSCRIPT_FEATURES = set(CB.CP.TRANSCRIPT_FEATURES)

# Content-feature priority for the "more-constrained pool owns the pair" rule (pre-registered).
_POOL_OWNER_ORDER = ["deception", "harmful_vs_benign", "topic_medical", "entity_mention_trigger"]


def is_newxnew_pair(pair_key: str) -> bool:
    try:
        f1, f2 = EXT.split_pair_key(pair_key)
    except KeyError:
        return False
    return f1 in EXT.EXTENSION_ORDER and f2 in EXT.EXTENSION_ORDER


def newxnew_pair(pair_key: str) -> tuple[Property, Property, tuple[str, ...]]:
    """(f1_prop, f2_prop, seed_pool) for a new×new pair. Orientation asserted (earlier in
    EXTENSION_ORDER = f1)."""
    f1_name, f2_name = EXT.split_pair_key(pair_key)
    assert pair_key == EXT.new_new_pair_key(f1_name, f2_name), \
        f"{pair_key}: new×new pairs must be EXTENSION_ORDER-oriented"
    assert not ({f1_name, f2_name} & TRANSCRIPT_FEATURES), \
        (f"{pair_key}: this builder excludes transcript features (genre confound); v2 builds "
         "transcript-side new×new pairs via lib/v2_build.py (all four cells as transcripts)")
    f1, f2 = EXT.lookup_property(f1_name), EXT.lookup_property(f2_name)
    owners = [n for n in _POOL_OWNER_ORDER if n in (f1_name, f2_name)
              and EXT.lookup_property(n).seeds]
    if owners:
        seed_pool = EXT.lookup_property(owners[0]).seeds
    else:
        from properties import PROPERTIES

        seed_pool = PROPERTIES["hedging"].seeds  # the pre-registered neutral topic pool
    assert seed_pool
    return f1, f2, seed_pool


def _cell_signs(cell_key: str) -> tuple[str, str]:
    for key, s1, s2, _cls in CELLS:
        if key == cell_key:
            return s1, s2
    raise KeyError(cell_key)


# ------------------------------------------------------------------- per-cell word target
# Measured per-(pair, cell) overrides, retuned from the buildability screens exactly as the
# deterministic arm's PAIR_CELL_FACTOR_OVERRIDES were (populated at screen/build QA when a cell's
# measured token median is off its held sibling's by >~20 tokens).
#   xq7×leet A: screen tok_med 293 vs the leet-only sibling's 232 at target (36,43) — the +8-word
#   xq7 bonus over-fills under the ×4-5 leet blowup; retarget ×(235/293): (29, 34).
PAIR_CELL_NUDGE_OVERRIDES: dict[tuple[str, str], tuple[int, int]] = {
    ("substring_high_entropy__leetspeak", "A"): (29, 34),
    # leetspeak__ai_self_reference: the leet+ cells (A, held_f1pos) ran long under the transform
    # (committed tok_median 302/280 vs the leet− cells' ~237), so the identical-histogram held match
    # kept only 128/cell. ai_self_reference+ content resists the short leet target and overshoots, so
    # retarget the two leet+ cells shorter to converge their post-transform tokens with the leet−
    # cells (≈237): 34-41 × (237/302) ≈ 27-32 for A, × (237/280) ≈ 29-35 for held_f1pos.
    ("leetspeak__ai_self_reference", "A"): (26, 31),
    ("leetspeak__ai_self_reference", "held_f1pos"): (28, 34),
    # Harm pairs with a systematic per-pole length gap between the two HELD cells (accept ~0.94, so
    # a pure length-match discard, not a content limit): the transform/insertion side shifts its
    # held cell's committed-token median away from the harm-natural ~240-tok non-transform side, so
    # retarget the transform/insertion + cells to converge the two held cells at the source
    # (INSTRUCTIONS: fix at source, not brute-force buffer). Word targets scaled by
    # (240 / measured-committed-tok-median) of the too-short/too-long + cell.
    ("all_caps__harmful_vs_benign", "A"): (112, 133),          # caps+ over-shortened (207 tok -> ~240)
    ("all_caps__harmful_vs_benign", "held_f1pos"): (112, 133),
    ("leetspeak__harmful_vs_benign", "A"): (40, 45),           # leet+ over-shortened (208 tok -> ~240)
    ("leetspeak__harmful_vs_benign", "held_f1pos"): (40, 45),
    ("substring_rare_wordlike__harmful_vs_benign", "A"): (170, 200),        # rare+ insertion over-lengthened (266 -> ~240)
    ("substring_rare_wordlike__harmful_vs_benign", "held_f1pos"): (170, 200),
}


def newxnew_cell_length_nudge(pair_key: str, cell_key: str) -> str:
    """Word target combining BOTH sides' committed adjustments: insertion-trigger + cells add the
    feature's word bonus; transform + cells multiply by the measured 'en' token-blowup factor
    (bonus applied before the factor). No Spanish / code-partner variants exist in new×new pairs
    (no core partners). A transform cell that also hosts a middle-sentence insertion gets an
    explicit short-sentence hint (the ~40-word leet target must still carry >=4 sentences for the
    insertion rule)."""
    f1_name, f2_name = EXT.split_pair_key(pair_key)
    s1, s2 = _cell_signs(cell_key)
    if (pair_key, cell_key) in PAIR_CELL_NUDGE_OVERRIDES:
        lo, hi = PAIR_CELL_NUDGE_OVERRIDES[(pair_key, cell_key)]
    else:
        lo, hi = 170, 205
        for name, sign in ((f1_name, s1), (f2_name, s2)):
            if sign == "+" and name in EXT.INSERTION_WORD_BONUS:
                b = EXT.INSERTION_WORD_BONUS[name]
                lo, hi = lo + b, hi + b
        for name, sign in ((f1_name, s1), (f2_name, s2)):
            if sign == "+" and name in EXT.WORD_TARGET_FACTORS:
                f = EXT.WORD_TARGET_FACTORS[name]["en"]
                lo, hi = round(lo * f), round(hi * f)
    nudge = EB._word_nudge(lo, hi)
    # Features with a minimum-sentence structural requirement: insertion triggers need a MIDDLE
    # sentence (so >=3-4 sentences), entity_mention_trigger+ needs a >=4-sentence passage with
    # Marseille confined to <=2 of them. Under a short leet target (~40 words) the model otherwise
    # writes 2-3 long sentences and the rule fails, so force several short sentences.
    needs_min_sentences = any(
        s == "+" and (n in EXT.INSERTION_WORD_BONUS or n == "entity_mention_trigger")
        for n, s in ((f1_name, s1), (f2_name, s2)))
    if hi < 100 and needs_min_sentences:
        nudge += " Use at least five short sentences."
    return nudge


_TRANSFORM_FEATURES = ("all_caps", "leetspeak")


def _make_newxnew_transform_validate():
    """validate_text semantics for a new×new generate-then-transform pair (≥1 side is
    all_caps/leetspeak). A char transform corrupts the OTHER feature's surface regex on + cells
    (leet maps a/e/i/o/s→4/3/1/0/5; caps uppercases, breaking exact-cased trigger rules and the
    case-sensitive other-city scan), so every NON-transform feature's surface rule runs on the
    readable PRE-transform text, while each transform feature's own rule (caps_rule/leet_rule) runs
    on the FINAL text. Judged partners are judged on the FINAL text (all_caps stays readable to the
    judge; leet×judged-content never reaches here — it routes to v2_build's pretransform rung).

    Generalizes extension_build._make_transform_partner_validate to a JUDGE-FREE partner (a
    deterministic trigger) and to the all_caps__leetspeak double transform (both sides = transform,
    so both rules run on the final composed text)."""

    async def _validate(text, f1, f1_sign, f2, f2_sign, pretransform=None):
        readable = pretransform if pretransform is not None else text  # readable == text on − cells
        labels, matches, reasons = {}, {}, {}
        surface_matches: dict[str, bool] = {}
        for tag, prop, sign in (("f1", f1, f1_sign), ("f2", f2, f2_sign)):
            view = text if prop.name in _TRANSFORM_FEATURES else readable
            if prop.surface_rule:
                surface_matches[prop.name] = validation.SURFACE_RULES[prop.surface_rule](view, sign)
            if prop.name in validation.JUDGE_FREE_PROPERTIES:
                labels[tag] = sign if surface_matches.get(prop.name) else "unclear"
                reasons[tag] = "surface-rule-only (judge skipped)"
            else:
                j = await validation.judge_feature(text, prop)  # judged on the final text
                labels[tag] = j["label"]
                reasons[tag] = j["reason"]
            matches[tag] = labels[tag] == sign
        return validation.ValidationResult(
            accepted=matches["f1"] and matches["f2"] and all(surface_matches.values()),
            f1_label=labels["f1"], f2_label=labels["f2"],
            f1_match=matches["f1"], f2_match=matches["f2"], surface_matches=surface_matches,
            detail={"f1_reason": reasons["f1"], "f2_reason": reasons["f2"]})

    _validate.needs_pretransform = True
    return _validate


# ------------------------------------------------------------------- per-cell generation kwargs
def newxnew_cell_gen_kwargs(pair_key: str, cell_key: str) -> dict:
    f1_name, f2_name = EXT.split_pair_key(pair_key)
    s1, s2 = _cell_signs(cell_key)
    sides = ((f1_name, s1), (f2_name, s2))
    feats = {f1_name, f2_name}

    transforms = [EV.nfc]
    for name, sign in sides:
        t = EV.GEN_TRANSFORMS.get((name, sign))
        if t:
            transforms.append(t)

    kwargs: dict = {
        "length_nudge": newxnew_cell_length_nudge(pair_key, cell_key),
        "text_transform": EB._compose(*transforms),
    }
    for name, sign in sides:
        if name == "trailing_signature" and sign == "+":
            kwargs["dash_view"] = EV.strip_trailing_signature
        if name == "starts_with_string" and sign == "+":
            kwargs["dedup_view"] = EV.strip_leading_start_string

    judged = sorted(feats - set(validation.JUDGE_FREE_PROPERTIES))
    if "leetspeak" in feats and judged and "harmful_vs_benign" not in feats:
        raise NotImplementedError(
            f"{pair_key}: leet × judged-feature pairs resolve their judge via the screened ladder "
            "in lib/v2_build.py (use v2_build.v2_cell_gen_kwargs / the complete_graph driver)")
    if "harmful_vs_benign" in feats:
        kwargs["generator_model"] = CB.HARM_GENERATOR_MODEL
        kwargs["gen_call_uuid"] = CB.HARM_GEN_UUID
        kwargs["validate_fn"] = CB._make_partner_and_feature_judge_validate(
            f1_model=CB.HARM_JUDGE_MODEL if f1_name == "harmful_vs_benign" else None,
            f2_model=CB.HARM_JUDGE_MODEL if f2_name == "harmful_vs_benign" else None)
        return kwargs
    # generate-then-transform pairs: a char transform (all_caps/leetspeak) corrupts the OTHER
    # feature's surface rule on + cells, so the partner rule must run on the readable pre-transform
    # text (the extension-arm transform-partner pattern, generalized to a judge-free trigger
    # partner). Only needed when a NON-transform feature carries a surface rule — the all_caps ×
    # judged-content-without-a-surface-rule pairs (deception/topic_medical/ai_self_reference) keep
    # the plain judged path (all_caps stays judge-readable), and the double transform
    # all_caps__leetspeak keeps the plain path too (both rules are defined on the final text; its
    # composed identity is enforced in verify_v2_grid).
    has_transform = bool(feats & set(_TRANSFORM_FEATURES))
    partner_surface = any(EXT.lookup_property(n).surface_rule and n not in _TRANSFORM_FEATURES
                          for n in feats)
    if has_transform and partner_surface:
        kwargs["validate_fn"] = _make_newxnew_transform_validate()
    return kwargs


# ------------------------------------------------------------------- buffered generation
async def generate_newxnew_pair_buffer(pair_key: str, n_target: int, verbose: int = 1) -> list[dict]:
    """Generate all four cells of a new×new pair to data/buffer/. Cached + resumable."""
    import asyncio

    import generation
    import numpy as np

    f1, f2, seed_pool = newxnew_pair(pair_key)

    async def gen_cell(cell_key: str) -> dict:
        res = await generation.generate_cell_props(
            pair_key, cell_key, f1, f2, seed_pool, n_target=n_target, verbose=verbose,
            **newxnew_cell_gen_kwargs(pair_key, cell_key))
        dataset.save_cell(res["accepted"], pair_key, cell_key, base_dir=BUFFER_DIR)
        toks = np.array([r["num_tokens"] for r in res["accepted"]]) if res["accepted"] else np.array([0])
        return {"pair": pair_key, "cell": cell_key, "n_target": n_target, **res["stats"],
                "tok_median": float(np.median(toks))}

    return await asyncio.gather(*[gen_cell(ck) for ck, *_ in CELLS])


# ------------------------------------------------------------------- commit + verification
def commit_newxnew_pair(pair_key: str, target: int = config.TEXTS_PER_CELL, write: bool = True) -> dict:
    """Buffer → normalized → (matched held cells, full-balanced A/B) → data/grid/.
    Identical treatment to commit_extension_pair, with the dedup prefix-strip applied for a
    starts_with side regardless of orientation."""
    grid = EB.normalize_grid(dataset.load_grid(pair_key, base_dir=BUFFER_DIR))
    f1_name, f2_name = EXT.split_pair_key(pair_key)
    dedup_view = (EV.strip_leading_start_string
                  if "starts_with_string" in (f1_name, f2_name) else None)
    _committed, report = LM.commit_grid_flow(pair_key, grid, target, dedup_view=dedup_view,
                                             write=write, grid_dir=GRID_DIR)
    return report


def verify_newxnew_grid(pair_key: str, base_dir=GRID_DIR) -> dict:
    """Exact programmatic verification of a committed new×new pair (0 failures required):
    BOTH features' deterministic surface rules at each cell's signs (triggers, transforms, the
    entity AND-gate), dash discipline (with the trailing-signature exemption), the frozen token
    window, harm-pair generator provenance (Opus model + cache namespace on every record), and
    position spread for insertion triggers."""
    import numpy as np

    grid = dataset.load_grid(pair_key, base_dir=base_dir)
    f1_name, f2_name = EXT.split_pair_key(pair_key)
    rules = {}
    for name in (f1_name, f2_name):
        prop = EXT.lookup_property(name)
        if prop.surface_rule:
            rules[name] = validation.SURFACE_RULES[prop.surface_rule]
    is_harm = "harmful_vs_benign" in (f1_name, f2_name)
    needles = {n: EV.POSITION_NEEDLES[n] for n in (f1_name, f2_name) if n in EV.POSITION_NEEDLES}

    failures = []
    positions = {n: [] for n in needles}
    other_city_hits_minus = []
    for cell in LM.ALL:
        for r in grid[cell]:
            text, ci = r["text"], r.get("candidate_index", "?")
            signs = {f1_name: r["f1_sign"], f2_name: r["f2_sign"]}
            for name, rule in rules.items():
                if not rule(text, signs[name]):
                    failures.append((cell, ci, f"{name}_rule_{signs[name]}"))
            if "entity_mention_trigger" in signs and signs["entity_mention_trigger"] == "-":
                other_city_hits_minus.append(len(CV.other_city_hits(text)))
            dash_text = (EV.strip_trailing_signature(text)
                         if signs.get("trailing_signature") == "+" else text)
            if data_checks.has_forbidden_dash(dash_text):
                failures.append((cell, ci, "dash"))
            if not (config.LENGTH_MIN_TOKENS <= r["num_tokens"] <= config.LENGTH_MAX_TOKENS):
                failures.append((cell, ci, f"length={r['num_tokens']}"))
            if is_harm:
                if r.get("generator_model") != CB.HARM_GENERATOR_MODEL:
                    failures.append((cell, ci, f"harm_generator={r.get('generator_model')}"))
                if r.get("gen_uuid") != CB.HARM_GEN_UUID:
                    failures.append((cell, ci, f"harm_gen_uuid={r.get('gen_uuid')}"))
            for name, needle in needles.items():
                if signs[name] == "+":
                    pos = EV.trigger_sentence_index(text, needle)
                    if pos is not None:
                        positions[name].append(pos / max(1, len(EV.sentences(text)) - 1))

    report = {
        "pair": pair_key,
        "n_per_cell": {c: len(grid[c]) for c in LM.ALL},
        "n_failures": len(failures),
        "failures": failures[:20],
        "ok": not failures,
    }
    for name, rel in positions.items():
        if rel:
            arr = np.array(rel)
            report[f"plus_position_rel_{name}"] = {
                "mean": float(arr.mean()), "p10": float(np.percentile(arr, 10)),
                "p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90))}
    if other_city_hits_minus:
        report["minus_other_city_count"] = {
            "mean": float(np.mean(other_city_hits_minus)),
            "all_exactly_one": all(c == 1 for c in other_city_hits_minus)}
    return report
