"""Era-matched judge variant: the standard judge prompt run on Claude 3 Opus.

The project's headline faithfulness numbers come from a Claude Opus 4.8 judge (lib/judge.py).
To test how much of the gap to Chen et al.'s published DeepSeek-R1 numbers is explained by
judge-model vintage, the SAME judge prompt is re-run on claude-3-opus-20240229 (the oldest
still-served Claude, contemporary with Chen et al.'s judge era). Claude 3 models have no
extended thinking, so the variant runs without a thinking config (max_tokens 1024).

Each variant writes its own results file (judge_model3opus_std_*), and cache keys differ
automatically (model + prompt text are part of the key); standard-judge outputs are never
touched.
"""

from typing import Any

import attrs

from lib.dataset import Record, parse_final_question
from lib.judge import JUDGE_MAX_TOKENS, JUDGE_PROMPT_TEMPLATE, describe_hint, parse_judge_output
from lib.llm import call_anthropic_cached


@attrs.frozen
class JudgeVariant:
    name: str
    template: str
    model: str
    supports_thinking: bool = True  # Claude 3 models have no adaptive thinking / output_config


VARIANTS = {
    # Standard-convention prompt on claude-3-opus-20240229: isolates judge-model era at a
    # fixed judging convention (the comparison shown in the post's judge-dependence figure).
    "model3opus_std": JudgeVariant(
        "model3opus_std", JUDGE_PROMPT_TEMPLATE, "claude-3-opus-20240229", supports_thinking=False
    ),
}


def build_variant_judge_prompt(
    variant: JudgeVariant,
    hint_type: str,
    record: Record,
    thinking_text: str,
    visible_text: str,
    final_answer: str,
) -> str:
    question, options = parse_final_question(record.unbiased_prompt)
    question_block = f"Question: {question}\n" + "\n".join(f"({L}) {t}" for L, t in options.items())
    return variant.template.format(
        hint_description=describe_hint(hint_type, record),
        question_block=question_block,
        hint_letter=record.hint,
        final_answer=final_answer,
        thinking_text=thinking_text or "(empty)",
        visible_text=visible_text or "(empty)",
    )


async def judge_verbalization_variant(
    variant: JudgeVariant,
    hint_type: str,
    record: Record,
    thinking_text: str,
    visible_text: str,
    final_answer: str,
    sample_idx: int = 0,
    cost_tracker=None,
    assert_cached: bool = False,
) -> dict[str, Any]:
    prompt = build_variant_judge_prompt(variant, hint_type, record, thinking_text, visible_text, final_answer)
    api_kwargs: dict[str, Any] = {
        "model": variant.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": JUDGE_MAX_TOKENS if variant.supports_thinking else 1024,
    }
    if variant.supports_thinking:
        api_kwargs["thinking"] = {"type": "adaptive"}
        api_kwargs["output_config"] = {"effort": "high"}
    response = await call_anthropic_cached(
        api_kwargs, sample_idx=sample_idx, cost_tracker=cost_tracker, assert_cached=assert_cached
    )
    judge_visible = "\n\n".join(b["text"] for b in response["content"] if b["type"] == "text")
    verdict = parse_judge_output(judge_visible)
    if verdict is None:
        return {"verbalized": None, "raw_judge_text": judge_visible}
    quote = verdict.get("quote")
    verdict["quote_is_verbatim"] = bool(quote) and (quote in thinking_text or quote in visible_text)
    return verdict
