"""Central configuration: models, paths, label tokens, prompt templates, the global
property ordering used to orient the logprob difference, and per-text check bounds.

Kept deliberately small; hand-authored property content lives in properties.py.
"""

from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------------------- models
SUBJECT_MODEL = "Qwen/Qwen3-1.7B"  # the model we probe and fine-tune
GENERATOR_MODEL = "claude-haiku-4-5-20251001"  # data generator (capable + cheap)
# Judge must be *stronger than the generator* (INSTRUCTIONS §5 check 4). Sonnet 4.6 with
# extended thinking is the default; we spot-check it against a hand-labeled sample.
JUDGE_MODEL = "claude-sonnet-4-6"

GENERATOR_TEMPERATURE = 1.0
GENERATOR_MAX_TOKENS = 700

# ----------------------------------------------------------------------------- paths
# lib/ holds the shared spine; data/results/plots/writeups live at the repo root one level up.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
WRITEUPS_DIR = ROOT / "writeups"
CACHE_DIR = ROOT / "file_cache_dir"
COST_FILE = ROOT / "total_cost.jsonl"


def results_dir(experiment: str) -> Path:
    """Per-experiment results directory (results/<experiment>/), created on demand. Use this
    instead of dumping into one flat results/ folder."""
    d = RESULTS_DIR / experiment
    d.mkdir(parents=True, exist_ok=True)
    return d


def plots_dir(experiment: str) -> Path:
    """Per-experiment plots directory (plots/<experiment>/), created on demand."""
    d = PLOTS_DIR / experiment
    d.mkdir(parents=True, exist_ok=True)
    return d


# NB: directories are created on demand (results_dir / plots_dir / dataset.save_cell), never at
# import time — importing config has no filesystem side effects.

# ----------------------------------------------------------------------------- reproducibility
SEED = 0  # base seed for deterministic train/eval splits and tokenization

# ----------------------------------------------------------------------------- per-text checks (INSTRUCTIONS §5)
LENGTH_MIN_TOKENS = 150
LENGTH_MAX_TOKENS = 320
# Within-cell near-duplicate drop: difflib ratio on lowercased, whitespace-normalized
# first 200 chars (INSTRUCTIONS §5 check 3).
DEDUP_PREFIX_CHARS = 200
DEDUP_RATIO_THRESHOLD = 0.80
DASH_CHARS = ("—", "–")  # em dash, en dash — a generator tell

# Texts accepted per cell for the test pairs (INSTRUCTIONS §5).
TEXTS_PER_CELL = 300

# ----------------------------------------------------------------------------- label tokens (INSTRUCTIONS §2b, §8)
# Neutral tokens unrelated to the features. Each candidate is a (token_a, token_b) pair
# written WITH a leading space, because the label is emitted right after "Answer:" where
# the model produces a space-prefixed token. We assert each is a single Qwen3 token and
# pick the pair whose un-fine-tuned base prior is closest to 50/50 (INSTRUCTIONS §2b).
LABEL_TOKEN_CANDIDATES: list[tuple[str, str]] = [
    (" A", " B"),
    (" X", " Y"),
    (" M", " N"),
    (" red", " blue"),
]
# The pinned default. Chosen by the §2b base-prior measurement (run_baseline_prior): X/Y is the
# most balanced candidate (mean |0.5 - P(A)| = 0.082, vs 0.157 for A/B which leans ~71/29). The §2b
# robustness grid still tests A/B and M/N; the sign is stable across all of them.
DEFAULT_LABEL_TOKENS: tuple[str, str] = (" X", " Y")

# ----------------------------------------------------------------------------- classification prompt (INSTRUCTIONS §2b, §7, §8)
# Templates wrap the text and end right before the label position (no trailing space, so
# the next token is the space-prefixed label). {text}, {a}, {b} are filled in. We test
# multiple wordings for robustness (INSTRUCTIONS §2b). The label characters in the prompt
# are the stripped tokens (e.g. "A"/"B").
PROMPT_TEMPLATES: dict[str, str] = {
    "classify": (
        "{text}\n\n"
        "The passage above belongs to one of two categories, {a} or {b}.\n"
        "Answer with a single letter.\n"
        "Answer:"
    ),
    "group": (
        "Passage:\n{text}\n\n"
        "Which group does this passage belong to, {a} or {b}? "
        "Respond with just the group label.\n"
        "Group:"
    ),
    "label": (
        "Text:\n{text}\n\n"
        "Sort this text into bucket {a} or bucket {b}. Reply with one character only.\n"
        "Bucket:"
    ),
}
DEFAULT_PROMPT_TEMPLATE = "classify"  # pinned default; confirmed by the §2b check

# ----------------------------------------------------------------------------- Modal
MODAL_GPU = "B200"  # default GPU; faster than A100 for this workload

# Client-side cap on concurrent Modal GPU calls; Modal autoscales containers and queues the rest.
# Set None for no cap, or override per run with --concurrency.
DEFAULT_CONCURRENCY = 100
