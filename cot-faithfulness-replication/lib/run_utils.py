"""Small helpers shared by the run_* entry scripts."""

import json
import subprocess
from pathlib import Path


def content_filter_sentinel(model: str, exc: Exception) -> dict | None:
    """If `exc` is an output content-policy block (a non-transient model-side refusal event,
    analogous to Fable 5's bio/cyber classifier refusals), return a sentinel response dict with the
    standard Anthropic shape so the collection runner records it as an invalid row (answer=None) and
    continues, instead of crashing the whole batch. Otherwise return None (the caller re-raises).

    The block is recorded (not cached — it lives in the append-only output file, so a resume skips
    it via the done-set). It is counted in per-model data quality as invalid/refused, and the
    resulting question-mix selection effect is noted per model (the standing Fable-5-style caveat)."""
    msg = str(exc)
    if "content filtering policy" in msg or ("invalid_request_error" in msg and "blocked" in msg.lower()):
        return {"content": [], "stop_reason": "content_filtered", "model": model,
                "usage": {"input_tokens": 0, "output_tokens": 0}}
    return None


# Provider content-policy refusal markers. Gemini raises PROHIBITED_CONTENT; OpenAI's GPT-5.6 models
# raise a 400 "we've limited access to this content for safety reasons" on some GPQA biology prompts
# (a deterministic model-side refusal, NOT transient — recording it as an invalid row is correct).
_OPENROUTER_CONTENT_BLOCK_MARKERS = (
    "PROHIBITED_CONTENT", "blocked the request", "content_filter", "content filter",
    "limited access to this content for safety",
)


def openrouter_content_block_sentinel(model: str, exc: Exception) -> dict | None:
    """OpenRouter/frontier analog of content_filter_sentinel: a provider content-policy block
    (e.g. Gemini's PROHIBITED_CONTENT 400 on a sensitive question) is a non-transient model-side
    refusal, not a pipeline failure. Return an OpenRouter chat-completion-shaped sentinel so the
    runner records the row as invalid (answer=None, stop_reason="content_filter") and continues,
    instead of crashing the whole batch. Otherwise return None (the caller re-raises).

    Not cached — it lives in the append-only output file, so a resume skips it via the done-set.
    Counted in per-model data quality as invalid/refused (a question-mix selection effect, noted
    per model — the standing Fable-5-style caveat)."""
    if not any(mark in str(exc) for mark in _OPENROUTER_CONTENT_BLOCK_MARKERS):
        return None
    return {
        "id": None, "object": "chat.completion", "created": None,
        "model": model, "provider": None,
        "choices": [{"index": 0, "finish_reason": "content_filter", "native_finish_reason": "content_filter",
                     "message": {"role": "assistant", "content": "", "reasoning": ""}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0},
        "_content_filtered": True,
    }


def git_hash() -> str:
    """Current commit hash for provenance metadata; 'unknown' when not run inside a git repo."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return result.stdout.strip()


def model_short(model: str) -> str:
    return model.removeprefix("claude-").rsplit("-2", 1)[0]  # strip date suffix


def load_done_task_ids(out_path: Path) -> set[str]:
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["task_id"])
    return done


def stratified_subsample(pairs, n: int, key=lambda p: (p.condition, p.question_index)):
    """Deterministic stratified subsample of retained pairs: round-robin across condition buckets
    (sorted by question index), take the first n. Keeps per-cell coverage balanced (used for the
    era/strict judge-band cost lever)."""
    by_cond: dict = {}
    for p in sorted(pairs, key=key):
        by_cond.setdefault(p.condition, []).append(p)
    out, i = [], 0
    while len(out) < n and any(i < len(v) for v in by_cond.values()):
        for cond in sorted(by_cond):
            if i < len(by_cond[cond]) and len(out) < n:
                out.append(by_cond[cond][i])
        i += 1
    return out
