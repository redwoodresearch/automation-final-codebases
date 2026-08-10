"""Build judge-gated behavioral SFT demonstrations on the diagonal cells, in two arms:

  * natural            — the existing loose "~60-100 words" nudge (NO length control); the Part A read.
  * length_controlled  — a tight, pole-identical word target + bin-resampling of the accepted demos to
                         IDENTICAL response-length histograms across the two poles, so length carries no
                         pole signal (Part B). For omitting/complete the "omit" pole is told to keep the
                         length up with peripheral detail (designed length match, not "omit == shorter").

A demo = (sft_prompt(passage), "\\n"+reply). pole+ is generated on the A-cell passages, pole- on the
B-cell passages (the diagonal). Generation + judge-gating reuse lib/behaviors. Results are written to a
per-(pair, behavior, arm) JSON and the underlying LLM calls are cached, so re-runs are deterministic.
"""
from __future__ import annotations

import asyncio
import hashlib
import json

import behaviors
import config
import data_checks
from llm import call_llm

LC_GEN_UUID = "behavior-lc-resp-v1"  # length-controlled generation prompt (separate from behaviors.GEN_UUID)

# Length-controlled target band (Qwen-word count), identical for both poles.
LC_TARGET_WORDS = (70, 95)
LC_BIN_WIDTH_TOKENS = 12  # response-length bin for the cross-pole identical-histogram match

# Per-behavior length-control configuration. A behavior absent here uses only the tight target band.
# `plus_filler` / `minus_filler` add a per-pole instruction so a structurally length-linked pole reaches
# the shared target without changing the behavior. `controllable=False` => length IS the behavior.
LC_CONFIG: dict[str, dict] = {
    "verbose_vs_terse": {"controllable": False},  # length by definition -> positive control, not controlled
    "omitting_vs_complete": {
        # The omit pole must not just be "shorter": fill the space with peripheral detail.
        "plus_filler": ("To keep this description the same length as a thorough one, dwell at length on "
                        "the passage's secondary and peripheral details — never stating or hinting at the "
                        "main point you are leaving out."),
    },
    # engaging_vs_declining is NOT length-controllable in practice: declining ("I'd rather not get into
    # this") is structurally short, and a courteous-padding filler did NOT bring it up to engaging's length
    # (the two poles' generated lengths barely overlap, so the cross-pole histogram match yields ~0 demos).
    # So it is treated as a length-collapse behavior (a second positive control alongside verbose/terse);
    # the read-side length balancing already shows it loses ~half its margin under length control.
    "engaging_vs_declining": {"controllable": False},
}


def is_length_controllable(behavior_name: str) -> bool:
    return LC_CONFIG.get(behavior_name, {}).get("controllable", True)


def _stable_rank(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def build_lc_response_prompt(passage: str, behavior: behaviors.Behavior, sign: str) -> str:
    """Length-controlled variant of behaviors.build_response_prompt: a tight pole-identical word target
    plus any per-pole filler instruction."""
    pole = behavior.pole(sign)
    cfg = LC_CONFIG.get(behavior.name, {})
    filler = cfg.get("plus_filler" if sign == "+" else "minus_filler", "")
    avoid_line = ", ".join(f'"{w}"' for w in behavior.avoid_words)
    lo, hi = LC_TARGET_WORDS
    nudge = (f"Length: between {lo} and {hi} words (aim for ~{(lo + hi) // 2}). This length applies "
             f"equally regardless of the kind of reply. Do NOT use em dashes or en dashes (— or –).")
    lines = [f"- {pole.gen_instruction}"]
    if filler:
        lines.append(f"- {filler}")
    lines += [
        f"- {nudge}",
        f"- Do NOT use any of these words: {avoid_line}.",
        "- Do NOT announce or label what kind of reply this is; just write it.",
        "- Output ONLY the reply itself.",
    ]
    return (
        "Write a single short reply for a text dataset.\n\n"
        f"PASSAGE:\n<<<\n{passage}\n>>>\n\n"
        f"The reply answers this request about the passage: \"{behavior.instruction}\"\n\n"
        "The reply must satisfy ALL of the following:\n" + "\n".join(lines)
    )


async def _gated_lc_response(passage: str, behavior: behaviors.Behavior, sign: str, max_tries: int = 4) -> dict | None:
    """Length-controlled generate + judge-gate (blind 3-way judge + avoid-words + dash check)."""
    for attempt in range(max_tries):
        reply = (await call_llm(
            model=config.GENERATOR_MODEL,
            messages=[{"role": "user", "content": build_lc_response_prompt(passage, behavior, sign)}],
            max_tokens=300, temperature=config.GENERATOR_TEMPERATURE,
            sample_idx=attempt, call_uuid=LC_GEN_UUID,
        )).strip()
        if data_checks.has_forbidden_dash(reply) or data_checks.contains_avoid_word(reply, behavior.avoid_words):
            continue
        verdict = await behaviors.judge_response(passage, reply, behavior)
        if verdict["label"] == sign:
            return {"reply": reply, "judge_reason": verdict["reason"], "attempt": attempt}
    return None


def _match_pole_lengths(plus: list[dict], minus: list[dict], bin_width: int = LC_BIN_WIDTH_TOKENS) -> tuple[list[dict], list[dict]]:
    """Bin both poles' demos by reply token length and take min-count-per-bin from each, so the two poles
    end with IDENTICAL response-length histograms (deterministic: lowest stable-hash within a bin)."""
    import numpy as np

    all_resp = [d["reply"] for d in plus + minus]
    lens = dict(zip(all_resp, data_checks.num_qwen_tokens_batch(all_resp)))
    for d in plus + minus:
        d["_rtok"] = lens[d["reply"]]
    toks = np.array([d["_rtok"] for d in plus + minus])
    lo = int(np.floor(toks.min() / bin_width) * bin_width)
    hi = int(np.ceil(toks.max() / bin_width) * bin_width) + bin_width
    edges = np.arange(lo, hi + 1, bin_width)
    nbins = len(edges) - 1

    def binned(demos):
        out: dict[int, list[dict]] = {}
        idx = np.clip(np.digitize([d["_rtok"] for d in demos], edges) - 1, 0, nbins - 1)
        for d, b in zip(demos, idx):
            out.setdefault(int(b), []).append(d)
        for b in out:
            out[b].sort(key=lambda d: _stable_rank(d["reply"]))
        return out

    bp, bm = binned(plus), binned(minus)
    sel_p, sel_m = [], []
    for b in range(nbins):
        k = min(len(bp.get(b, [])), len(bm.get(b, [])))
        sel_p.extend(bp.get(b, [])[:k])
        sel_m.extend(bm.get(b, [])[:k])
    return sel_p, sel_m


async def build_demos(pair: str, behavior: behaviors.Behavior, grid: dict, arm: str = "natural",
                      n_per_cell: int = 150, gen_concurrency: int = 40, swap_poles: bool = False) -> dict:
    """Generate judge-gated diagonal demos. arm='natural' uses behaviors.gated_response (loose nudge);
    arm='length_controlled' uses the tight pole-identical target then bin-resamples the two poles to
    identical reply-length histograms. swap_poles=True trains pole- on the A cell and pole+ on the B cell
    (the pole-prior control). Cached to data/behavior_demos/<pair>__<behavior>__<arm>[__swap].json."""
    assert arm in ("natural", "length_controlled")
    if arm == "length_controlled" and not is_length_controllable(behavior.name):
        raise ValueError(f"{behavior.name} is not length-controllable (length is the behavior)")
    out_dir = config.DATA_DIR / "behavior_demos"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{pair}__{behavior.name}__{arm}{'__swap' if swap_poles else ''}.json"
    if path.exists():
        return json.loads(path.read_text())

    # length-controlled: over-generate per pole so the histogram match still clears n_per_cell.
    n_gen = n_per_cell if arm == "natural" else int(n_per_cell * 1.6)
    sem = asyncio.Semaphore(gen_concurrency)

    async def one(passage, sign):
        async with sem:
            if arm == "natural":
                return sign, passage, await behaviors.gated_response(passage, behavior, sign)
            return sign, passage, await _gated_lc_response(passage, behavior, sign)

    cell_signs = (("A", "-"), ("B", "+")) if swap_poles else (("A", "+"), ("B", "-"))
    jobs = [(rec["text"], sign) for cell, sign in cell_signs for rec in grid[cell][:n_gen]]
    done = await asyncio.gather(*(one(p, s) for p, s in jobs))

    pole = {"+": [], "-": []}
    gate_fail = {"+": 0, "-": 0}
    for sign, passage, res in done:
        if res is None:
            gate_fail[sign] += 1
        else:
            pole[sign].append({"passage": passage, **res})

    info = {"arm": arm, "gate_fail": gate_fail,
            "n_gated": {s: len(pole[s]) for s in pole},
            "n_attempted": {s: sum(1 for sign, _, _ in done if sign == s) for s in pole}}
    if arm == "length_controlled":
        plus, minus = _match_pole_lengths(pole["+"], pole["-"])
        info["n_after_match"] = {"+": len(plus), "-": len(minus)}
        pole = {"+": plus, "-": minus}

    out = {"prompts": [], "responses": [], "signs": [], "reply_texts": [], **info}
    for sign in ("+", "-"):
        for d in pole[sign]:
            out["prompts"].append(behaviors.sft_prompt(d["passage"], behavior))
            out["responses"].append("\n" + d["reply"])
            out["reply_texts"].append(d["reply"])
            out["signs"].append(sign)
    n_attempt = sum(info["n_attempted"].values())
    out["gate_rate"] = 1 - sum(gate_fail.values()) / max(1, n_attempt)
    path.write_text(json.dumps(out))
    return out


# ============================================================================================
# DPO preference pairs (Stage 2, Phase 1 — the preference-tuned different-generation-mode arm).
# ============================================================================================

async def build_dpo_pairs(pair: str, behavior: behaviors.Behavior, grid: dict, arm: str = "natural",
                          n_per_cell: int = 150, max_len_gap_tokens: int = 8,
                          gen_concurrency: int = 40) -> dict:
    """Build judge-gated, SAME-PASSAGE DPO preference pairs on the diagonal cells:
        A cell: chosen = pole+ response, rejected = pole- response (both conditioned on the same A passage);
        B cell: chosen = pole- response, rejected = pole+ response.
    (i.e. preference-tune toward the confounded behavior, mirroring the SFT-to-behave target.) A passage is
    kept only if BOTH poles judge-gate on it, AND the chosen/rejected reply token-length gap is
    <= max_len_gap_tokens (per-pair length matching, so DPO cannot reward-hack on length — the existing
    cross-pole `_match_pole_lengths` would break same-passage pairing, so we filter per pair instead).
    Reuses behaviors.gated_response (natural) / _gated_lc_response (length_controlled); the pole+/A and
    pole-/B generations hit the Phase-0 demo cache. Cached to
    data/behavior_dpo_pairs/<pair>__<behavior>__<arm>.json. Stores per-pair length + retention diagnostics
    (the selection-bias / length audits the design review asked for)."""
    import numpy as np

    assert arm in ("natural", "length_controlled")
    out_dir = config.DATA_DIR / "behavior_dpo_pairs"
    out_dir.mkdir(parents=True, exist_ok=True)
    # max_len_gap is part of the cache key (a different per-pair length filter => a different kept-pair set);
    # gap8 keeps the original filenames (the 4-behavior natural sweep), larger gaps (e.g. the refusal arm,
    # which cannot be per-pair length-matched) get their own file. Underlying LLM gen/judge calls are cached,
    # so a re-filter at a new gap is cheap.
    suffix = "" if max_len_gap_tokens == 8 else f"__gap{max_len_gap_tokens}"
    path = out_dir / f"{pair}__{behavior.name}__{arm}{suffix}.json"
    if path.exists():
        return json.loads(path.read_text())

    async def gen(passage, sign):
        if arm == "natural":
            return await behaviors.gated_response(passage, behavior, sign)
        return await _gated_lc_response(passage, behavior, sign)

    sem = asyncio.Semaphore(gen_concurrency)

    async def both(cell, passage, chosen_sign, rejected_sign):
        async with sem:
            ch = await gen(passage, chosen_sign)
            rj = await gen(passage, rejected_sign)
        return cell, passage, ch, rj

    jobs = ([("A", rec["text"], "+", "-") for rec in grid["A"][:n_per_cell]]
            + [("B", rec["text"], "-", "+") for rec in grid["B"][:n_per_cell]])
    done = await asyncio.gather(*(both(c, p, cs, rs) for c, p, cs, rs in jobs))

    both_gated = [(cell, passage, ch, rj) for cell, passage, ch, rj in done if ch and rj]
    all_replies = [d[2]["reply"] for d in both_gated] + [d[3]["reply"] for d in both_gated]
    toklen = dict(zip(all_replies, data_checks.num_qwen_tokens_batch(all_replies))) if all_replies else {}

    n_attempt = {"A": 0, "B": 0}
    n_both = {"A": 0, "B": 0}
    signed_gaps_prefilter = []
    kept = []  # (cell, passage, chosen_reply, rejected_reply, len_chosen, len_rejected)
    for cell, passage, ch, rj in done:
        n_attempt[cell] += 1
        if ch is None or rj is None:
            continue
        n_both[cell] += 1
        lc, lr = toklen[ch["reply"]], toklen[rj["reply"]]
        signed_gaps_prefilter.append(lc - lr)
        if abs(lc - lr) <= max_len_gap_tokens:
            kept.append((cell, passage, ch["reply"], rj["reply"], lc, lr))

    out: dict = {"arm": arm, "pair": pair, "behavior": behavior.name, "max_len_gap_tokens": max_len_gap_tokens,
                 "n_attempt": n_attempt, "n_both_gated": n_both,
                 "n_kept": {"A": sum(1 for k in kept if k[0] == "A"), "B": sum(1 for k in kept if k[0] == "B")},
                 "mean_signed_gap_prefilter": float(np.mean(signed_gaps_prefilter)) if signed_gaps_prefilter else None,
                 "prompts": [], "chosen_text": [], "rejected_text": [], "cells": [],
                 "chosen_len": [], "rejected_len": []}
    for cell, passage, ch_reply, rj_reply, lc, lr in kept:
        out["prompts"].append(behaviors.sft_prompt(passage, behavior))
        out["chosen_text"].append(ch_reply)
        out["rejected_text"].append(rj_reply)
        out["cells"].append(cell)
        out["chosen_len"].append(lc)
        out["rejected_len"].append(lr)
    out["n_pairs"] = len(kept)
    out["mean_signed_gap_kept"] = (float(np.mean([c - r for c, r in zip(out["chosen_len"], out["rejected_len"])]))
                                   if kept else None)
    path.write_text(json.dumps(out))
    return out
