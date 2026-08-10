"""Text generation for a 2x2 cell + the accept/redraw loop (INSTRUCTIONS §5).

Cell prompt = both feature values' generation instructions + combined avoid-words + one sampled
scenario seed (no few-shot examples; the descriptions carry the load). Candidates are accepted
only if they pass the cheap checks (length, dash, avoid-words, dedup) AND the independent judge /
code rule in validation.py.
"""

from __future__ import annotations

import asyncio

from attrs import frozen

import config
import data_checks
import diversify
from llm import call_llm
from properties import PAIRS, PROPERTIES, Pair, Property
from validation import validate_text

# gen-v2: extended per-candidate diversification (secondary narrative names, specificity
# people/places/orgs, imperative/question openings; per-cell decorrelation) + a gentle shared length
# nudge (INSTRUCTIONS §5, §7). The prompt text changes, so cached gen-v1 (de-risk/buildability)
# entries are not reused — expected; this phase regenerates at scale.
_GEN_UUID = "gen-v2"

_OPENING_NUDGE = "Vary your opening; do not begin with a textbook-style definition of the subject."
# Gentle, shared (feature-neutral) length nudge. The hard accept gate stays config.LENGTH_* tokens;
# this only narrows the prompt's word target so the four cells' raw length distributions sit closer
# (the longest cells, which hit the old ceiling, compress most), making the post-hoc length match
# (§5) far less lossy. Post-hoc matching is still the binding control.
_LENGTH_NUDGE = "Length: aim for roughly 170 to 205 words (keep close to this range)."


def build_cell_prompt(
    f1: Property, f1_sign: str, f2: Property, f2_sign: str, seed: str, candidate_index: int = 0,
    length_nudge: str = _LENGTH_NUDGE,
) -> str:
    v1, v2 = f1.value(f1_sign), f2.value(f2_sign)
    avoid = sorted(set(v1.avoid_words) | set(v2.avoid_words))
    avoid_line = ", ".join(f'"{w}"' for w in avoid)
    cell_tag = f"{f1_sign}{f2_sign}"  # decorrelates injected entities/openings across a pair's cells
    extras = (diversify.extra_lines(f1.name, f1_sign, cell_tag, candidate_index, partner=(f2.name, f2_sign))
              + diversify.extra_lines(f2.name, f2_sign, cell_tag, candidate_index, partner=(f1.name, f1_sign)))
    extras_block = ("".join(f"\n{e}" for e in extras)) if extras else ""
    return (
        "Write a single passage for a text dataset.\n\n"
        f"Subject: {seed}.\n\n"
        "The passage must satisfy ALL of the following:\n"
        f"- {v1.gen_instruction}\n"
        f"- {v2.gen_instruction}\n"
        f"- {length_nudge}\n"
        "- Do NOT use em dashes or en dashes (the characters — or –). Use commas, periods, "
        "or parentheses instead.\n"
        f"- Do NOT use any of these words anywhere in the prose: {avoid_line}.\n"
        f"- {_OPENING_NUDGE}\n"
        "- Do NOT announce or label what kind of passage this is; just write it."
        f"{extras_block}\n"
        "- Output ONLY the passage itself, with no title, preamble, or commentary."
    )


@frozen
class Candidate:
    text: str
    seed: str
    candidate_index: int
    gen_sample_idx: int


async def generate_candidate(prompt: str, sample_idx: int, model: str | None = None,
                             call_uuid: str = _GEN_UUID) -> str:
    return (
        await call_llm(
            model=model or config.GENERATOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.GENERATOR_MAX_TOKENS,
            temperature=config.GENERATOR_TEMPERATURE,
            sample_idx=sample_idx,
            call_uuid=call_uuid,
        )
    ).strip()


def _cell_signs(cell_key: str) -> tuple[str, str]:
    from properties import CELLS

    for key, f1_sign, f2_sign, _cls in CELLS:
        if key == cell_key:
            return f1_sign, f2_sign
    raise KeyError(cell_key)


async def _evaluate_candidate(
    cand: Candidate, f1: Property, f1_sign: str, f2: Property, f2_sign: str,
    text_transform=None, dash_view=None, validate_fn=None, text_assembler=None,
) -> dict:
    """Run the per-candidate checks that do NOT depend on the growing accepted set (length, dash,
    avoid-words, then the judge). Dedup is applied later, sequentially, by the caller.

    Extension hooks (all default to None = the original behavior, byte-identical for locked pairs;
    see extension_build.cell_gen_kwargs):
      * text_assembler(raw, seed): builds the final text from the raw generation (content arm:
        assembles the transcript with the byte-fixed user turn for the seed). Applied FIRST, before
        text_transform and all checks.
      * text_transform: applied to the (assembled) candidate BEFORE all checks except avoid-words;
        the final (transformed) text is what gets validated, stored, and read out. Avoid-words run
        on the RAW text (the giveaway concern is about the generated content; a transform like
        leetspeak would hide the giveaway from the check). If the final text differs from raw, the
        raw text is kept in record["text_pretransform"].
      * dash_view: sign-aware exemption view for the frozen no-dash cheap check (e.g. strip the
        trailing_signature + pole's own pinned signature); the check still rejects any other dash.
      * validate_fn: replaces validation.validate_text (used for the documented
        numerical_density × substring_high_entropy surface-rule exemption, and the content-arm
        transcript reply-view surface rules / harm judge amendment)."""
    raw = cand.text
    text = text_assembler(raw, cand.seed) if text_assembler else raw
    pre_transform = text  # assembled but untransformed (for validate_fns with needs_pretransform)
    if text_transform:
        text = text_transform(text)
    avoid = tuple(sorted(set(f1.value(f1_sign).avoid_words) | set(f2.value(f2_sign).avoid_words)))

    # The length check tokenizes (HF tokenizer, rust) — run it off the event loop so concurrent cells'
    # API calls overlap with it (otherwise the per-candidate tokenize serializes the whole loop).
    ok_len, n_tok = await asyncio.to_thread(data_checks.length_ok, text)
    reasons = []
    if not ok_len:
        reasons.append(f"length={n_tok}")
    if data_checks.has_forbidden_dash(dash_view(text) if dash_view else text):
        reasons.append("dash")
    aw = data_checks.contains_avoid_word(raw, avoid)
    if aw:
        reasons.append(f"avoid_word:{aw}")
    # Reject the two rare generation glitches the harm-wave QA surfaced (a repeated-symbol run, or
    # a text cut mid-clause) that the unicode-hygiene gate misses. Checked on the PRE-transform
    # surface (leet substitution erases ASCII terminal punctuation) — the reply portion for
    # transcript pairs, else the whole passage. Lazy import: extension_validators/content_validators
    # pull in heavier deps and generation is imported widely.
    import content_validators as _CV
    import extension_validators as _EV
    hygiene_surface = _CV.reply_view(pre_transform) if text_assembler else pre_transform
    if not _EV.well_formed_text(hygiene_surface):
        reasons.append("malformed_text")

    record = {
        "text": text, "seed": cand.seed, "candidate_index": cand.candidate_index,
        "gen_sample_idx": cand.gen_sample_idx, "num_tokens": n_tok,
    }
    if text != pre_transform:
        # The ASSEMBLED-but-untransformed text (== raw for passage pairs, so byte-identical for
        # every existing build). Storing `raw` here was a latent trap for transcript pairs: raw is
        # the bare reply, which the transcript-aware grid normalizer then mangled into a fake
        # 'User: <reply>' transcript. With `pre_transform` the field is only present when a real
        # text_transform ran, and for transcript pairs it is a well-formed transcript the
        # normalizer and the leet identity check can consume.
        record["text_pretransform"] = pre_transform
    if reasons:  # failed a cheap check; skip the (expensive) judge
        return {"passed_cheap": False, "passed": False, "reasons": reasons, "record": record}

    try:
        if getattr(validate_fn, "needs_pretransform", False):
            # v2 leet-content rung (c): the judge reads the readable PRE-transform text; the
            # deterministic transform + its identity check happen inside the validate_fn.
            val = await validate_fn(text, f1, f1_sign, f2, f2_sign, pretransform=pre_transform)
        else:
            val = await (validate_fn or validate_text)(text, f1, f1_sign, f2, f2_sign)
    except Exception as e:  # a malformed judge response shouldn't crash the whole cell — reject it
        return {"passed_cheap": True, "passed": False, "reasons": [f"judge_error:{type(e).__name__}"],
                "record": record}
    record["validation"] = {
        "f1_label": val.f1_label, "f2_label": val.f2_label,
        "f1_match": val.f1_match, "f2_match": val.f2_match,
        "surface_matches": val.surface_matches,
        **val.detail,
    }
    if not val.accepted:
        if not val.f1_match:
            reasons.append(f"judge_f1={val.f1_label}")
        if not val.f2_match:
            reasons.append(f"judge_f2={val.f2_label}")
        for pname, ok in val.surface_matches.items():
            if not ok:
                reasons.append(f"surface:{pname}")
    return {"passed_cheap": True, "passed": val.accepted, "reasons": reasons, "record": record}


async def generate_cell(
    pair_key: str,
    cell_key: str,
    n_target: int = config.TEXTS_PER_CELL,
    batch_size: int = 60,
    max_candidates: int = 4000,
    seed_pool: tuple[str, ...] | None = None,
    verbose: int = 1,
) -> dict:
    """Accept/redraw until n_target texts pass every check, for a pair in the locked pair lists.
    Thin wrapper over generate_cell_props (which extension pairs call with explicit properties)."""
    if pair_key in PAIRS:
        pair = PAIRS[pair_key]
    else:  # expansion pair (stage 6) — same Pair fields; lazy import keeps the original-32 path untouched
        from pairs_expanded import PAIRS_EXPANDED
        pair = PAIRS_EXPANDED[pair_key]
    f1, f2 = PROPERTIES[pair.f1], PROPERTIES[pair.f2]
    if seed_pool is None:
        seed_pool = PROPERTIES[pair.seed_property].seeds
    return await generate_cell_props(
        pair_key, cell_key, f1, f2, seed_pool,
        n_target=n_target, batch_size=batch_size, max_candidates=max_candidates, verbose=verbose,
    )


async def generate_cell_props(
    pair_key: str,
    cell_key: str,
    f1: Property,
    f2: Property,
    seed_pool: tuple[str, ...],
    n_target: int = config.TEXTS_PER_CELL,
    batch_size: int = 60,
    max_candidates: int = 4000,
    verbose: int = 1,
    collect_rejects: bool = False,
    length_nudge: str | None = None,
    text_transform=None,
    dash_view=None,
    dedup_view=None,
    validate_fn=None,
    prompt_builder=None,
    text_assembler=None,
    generator_model: str | None = None,
    gen_call_uuid: str | None = None,
) -> dict:
    """The accept/redraw core, taking explicit Property objects (so out-of-roster extension pairs
    reuse the exact frozen flow). Candidates are enumerated deterministically: candidate i uses
    seed_pool[i % S] and gen_sample_idx = i // S, so (seed, sample_idx) is reproducible and fully
    cached. Prompts and records are byte-identical to the pre-refactor generate_cell for locked
    pairs. collect_rejects additionally returns every rejected candidate + reasons (inspection).

    Extension hooks (all None = original behavior; wired per cell by extension_build.cell_gen_kwargs):
    length_nudge (per-cell word target), text_transform / dash_view / validate_fn (see
    _evaluate_candidate), and dedup_view (a view of the final text used ONLY for the near-duplicate
    comparison — e.g. the starts_with + pole strips its pinned shared prefix so dedup compares
    bodies rather than the identical opener).

    Content-arm hooks (stage 3; all None = original behavior):
      * prompt_builder(f1, f1_sign, f2, f2_sign, seed, candidate_index, length_nudge) -> str:
        replaces build_cell_prompt (the transcript builder passes its two-turn template here). The
        default keeps the passage prompt byte-identical.
      * generator_model / gen_call_uuid: use Opus (a distinct model + cache namespace) for the harm
        arm without touching the haiku default; stamped into each record's generator_model."""
    # Pre-warm the HF tokenizer on the CALLING thread before the to_thread length checks fan out:
    # transformers' lazy AutoTokenizer import is not thread-safe on its first call (protects every
    # generate_cell caller, not just the runners that remembered to warm it).
    data_checks.num_qwen_tokens("tokenizer warmup")
    f1_sign, f2_sign = _cell_signs(cell_key)
    S = len(seed_pool)

    accepted: list[dict] = []
    accepted_norm: list[str] = []
    rejected: list[dict] = []
    stats = {"cheap_fail": 0, "judge_fail": 0, "dedup_fail": 0, "n_candidates": 0,
             "fail_reasons": {}}
    next_i = 0
    nudge_kwargs = {} if length_nudge is None else {"length_nudge": length_nudge}
    _build = prompt_builder or build_cell_prompt
    while len(accepted) < n_target and next_i < max_candidates:
        idxs = list(range(next_i, min(next_i + batch_size, max_candidates)))
        next_i = idxs[-1] + 1
        prompts = [_build(f1, f1_sign, f2, f2_sign, seed_pool[i % S], i, **nudge_kwargs)
                   for i in idxs]
        cands = await asyncio.gather(*[
            generate_candidate(p, i // S, model=generator_model,
                               **({"call_uuid": gen_call_uuid} if gen_call_uuid else {}))
            for p, i in zip(prompts, idxs)])
        candidates = [Candidate(t, seed_pool[i % S], i, i // S) for t, i in zip(cands, idxs)]
        evals = await asyncio.gather(*[
            _evaluate_candidate(c, f1, f1_sign, f2, f2_sign, text_transform=text_transform,
                                dash_view=dash_view, validate_fn=validate_fn,
                                text_assembler=text_assembler)
            for c in candidates])
        stats["n_candidates"] += len(candidates)
        for ev in evals:
            if not ev["passed"]:
                if not ev["passed_cheap"]:
                    stats["cheap_fail"] += 1
                else:
                    stats["judge_fail"] += 1
                for r in ev["reasons"]:
                    key = r.split(":")[0].split("=")[0]
                    stats["fail_reasons"][key] = stats["fail_reasons"].get(key, 0) + 1
                if collect_rejects:
                    rejected.append({"reasons": ev["reasons"], **ev["record"]})
                continue
            text = ev["record"]["text"]
            dedup_text = dedup_view(text) if dedup_view else text
            # Dedup is O(n^2) difflib and grows with the accepted set; run it off the event loop so other
            # concurrent cells progress during the check (this cell's accepted_norm is stable here — it is
            # only appended below, after the check).
            if await asyncio.to_thread(data_checks.is_near_duplicate, dedup_text, accepted_norm):
                stats["dedup_fail"] += 1
                stats["fail_reasons"]["dedup"] = stats["fail_reasons"].get("dedup", 0) + 1
                if collect_rejects:
                    rejected.append({"reasons": ["dedup"], **ev["record"]})
                continue
            rec = {
                "pair": pair_key, "cell": cell_key,
                "f1": f1.name, "f1_sign": f1_sign, "f2": f2.name, "f2_sign": f2_sign,
                "generator_model": generator_model or config.GENERATOR_MODEL,
                "gen_uuid": gen_call_uuid or _GEN_UUID,
                **ev["record"],
            }
            accepted.append(rec)
            accepted_norm.append(data_checks.normalized_prefix(dedup_text))
            if len(accepted) >= n_target:
                break
        if verbose:
            print(f"  [{pair_key}/{cell_key}] accepted {len(accepted)}/{n_target} "
                  f"(candidates={stats['n_candidates']}, cheap_fail={stats['cheap_fail']}, "
                  f"judge_fail={stats['judge_fail']}, dedup_fail={stats['dedup_fail']})")

    accept_rate = len(accepted) / max(1, stats["n_candidates"])
    stats["accept_rate"] = accept_rate
    stats["n_accepted"] = len(accepted)
    out = {"accepted": accepted, "stats": stats}
    if collect_rejects:
        out["rejected"] = rejected
    return out


