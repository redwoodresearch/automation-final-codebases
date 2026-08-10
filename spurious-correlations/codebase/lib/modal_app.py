"""Modal app: Qwen3-1.7B on a GPU via HuggingFace transformers (INSTRUCTIONS §3).

All three measurements (probe, fine-tune, un-fine-tuned baseline) run here so tokenization and
the forward pass are identical across them. Model weights are cached to a Modal Volume so they
download once. The shared metric module (metric.py) is baked into the image and used remotely so
the logprob difference is computed by the same code everywhere.

Remote entry points:
  * ``extract_activations``  — all-layer last-non-pad-token hidden states (the reusable primitive;
    activations are produced on demand and never persisted to disk).
  * ``probe_logprob_diff``   — fit the linear probe (all layers x seeds) on training cells and
    evaluate the held-out logprob difference; returns only small numbers.
  * ``baseline_logprob_diff``— un-fine-tuned model's held-out logprob difference (the §2a reference).
  * ``finetune_and_eval``    — full fine-tune on training cells; training-signal + held-out curves.

Heavy imports (torch/transformers/sklearn) stay inside functions so this module also imports
locally (for the app/image objects and cache keys).
"""

from __future__ import annotations

import os
import time

import modal

import metric  # baked into the image; numpy+attrs only, imports fine locally too

MODEL_NAME = "Qwen/Qwen3-1.7B"
GPU = "B200"  # faster than A100; 1.7B fine-tune fits easily
DTYPE = "float32"  # fp32 PARAMETERS/optimizer; forward matmuls run under bf16 autocast (see NOTES.md)
HF_CACHE_VOL = modal.Volume.from_name("derisk-hf-cache", create_if_missing=True)
HF_CACHE_DIR = "/hf_cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    # torch from the CUDA 12.8 index so it carries Blackwell (sm_100) kernels for B200 GPUs.
    .pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "transformers==4.56.2",
        "accelerate==1.10.1",
        "scikit-learn==1.7.2",
        "numpy==2.2.6",
        "attrs==25.4.0",
    )
    .env({"HF_HOME": HF_CACHE_DIR, "HF_HUB_ENABLE_HF_TRANSFER": "0", "TOKENIZERS_PARALLELISM": "false"})
    # __file__-relative (not CWD-relative), so entry points can be launched from any directory.
    .add_local_file(str(os.path.join(os.path.dirname(os.path.abspath(__file__)), "metric.py")),
                    "/root/metric.py", copy=True)
)

app = modal.App("spurious-correlations")


# ============================================================================= shared helpers (remote)
def _load_model_and_tokenizer(dtype: str = DTYPE, attn: str = "eager", model_name: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = model_name or MODEL_NAME  # None => the pinned default (Qwen3-1.7B); else cross-model read
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # eager for activation extraction (exact hidden states); sdpa for the fine-tune (much faster).
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=HF_CACHE_DIR, dtype=getattr(torch, dtype), attn_implementation=attn
    ).to("cuda")
    return model, tokenizer


def _set_determinism(seed: int):
    # Fix seeds for weight init / data shuffling / split. We do NOT force deterministic CuBLAS
    # kernels: that needs CUBLAS_WORKSPACE_CONFIG (an image env change) and is much slower, while
    # the residual GPU nondeterminism is tiny next to the seed-to-seed variance we measure (§7).
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _wrap_text(tokenizer, text: str, wrap: str) -> tuple[list[int], bool]:
    """Return (token_ids, add_special_tokens_flag) for the probe representation of a raw text.

    wrap='none' -> raw text, no chat template, no special tokens (pinned default; INSTRUCTIONS §8).
    wrap='chat' -> the model's chat template (the §8 equivalence check).
    """
    if wrap == "chat":
        wrapped = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
        return tokenizer(wrapped, add_special_tokens=False)["input_ids"], False
    assert wrap == "none", f"unknown wrap {wrap!r}"
    return tokenizer(text, add_special_tokens=False)["input_ids"], False


def _chat_masked_ids(tokenizer, text: str) -> tuple[list[int], list[int]]:
    """Token ids of the chat-templated text (user turn = the text, add_generation_prompt=True —
    the same rendering as wrap='chat') plus a 0/1 CONTENT mask selecting exactly the tokens whose
    character span overlaps the text's own span inside the template. Role scaffolding and the
    generation cue get mask 0; a boundary token that merges a template character with content
    characters counts as content (quantified in the phase's token-layout inspection).
    Used by the additive 'chat_masked' mean-pooled read (phase chat_meanpooled_reads); the
    pre-existing reads are untouched."""
    wrapped = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )
    start = wrapped.find(text)
    assert start != -1, "chat template altered the text"
    assert wrapped.find(text, start + 1) == -1, "text occurs twice in the wrapped string"
    end = start + len(text)
    enc = tokenizer(wrapped, add_special_tokens=False, return_offsets_mapping=True)
    mask = [1 if (o[1] > start and o[0] < end and o[1] > o[0]) else 0 for o in enc["offset_mapping"]]
    assert sum(mask) > 0, "empty content mask"
    return enc["input_ids"], mask


def _extract_all_layer_repr(model, tokenizer, texts: list[str], wrap: str = "none", representation: str = "last",
                            prompt_template: str | None = None, a_char: str | None = None,
                            b_char: str | None = None, batch_size: int = 16):
    """Return float32 ndarray [n_texts, n_hidden_states, hidden] of per-layer activations, summarized
    over tokens by `representation`:
      * "last"     = last non-pad token of the (raw or chat-wrapped) text (the pilot default);
      * "mean"     = mean over all non-pad tokens (guards against a last-token-position artifact);
      * "prompted" = last non-pad token of ``text + prompt_template`` (with a_char/b_char filled in) —
        exactly the position the fine-tune / baseline read before emitting the label token. This is the
        cleanest apples-to-apples representation for the RQ1 probe↔fine-tune comparison (INSTRUCTIONS §5).
    last/mean behaviour is byte-identical to before (the pinned probe path); only "prompted" is new.
    Batched forward passes; no activations persisted to disk."""
    import numpy as np
    import torch

    content_masks = None
    if representation == "prompted":
        assert prompt_template is not None, "prompted representation needs a prompt_template"
        all_ids = [tokenizer(_format_prompt(prompt_template, t, a_char, b_char),
                             add_special_tokens=False)["input_ids"] for t in texts]
        pooling = "last"
    elif wrap == "chat_masked":
        # additive read (phase chat_meanpooled_reads): chat-templated text, mean-pooled over
        # CONTENT tokens only. Existing wraps/representations take the unchanged paths below.
        assert representation == "mean", "chat_masked is a masked mean-pool read"
        masked = [_chat_masked_ids(tokenizer, t) for t in texts]
        all_ids = [ids for ids, _ in masked]
        content_masks = [m for _, m in masked]
        pooling = "mean"
    else:
        all_ids = [_wrap_text(tokenizer, t, wrap)[0] for t in texts]
        pooling = representation  # 'last' or 'mean'
    pad_id = tokenizer.pad_token_id
    out_chunks = []
    for start in range(0, len(all_ids), batch_size):
        batch = all_ids[start : start + batch_size]
        maxlen = max(len(x) for x in batch)
        input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for i, ids in enumerate(batch):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            attn[i, : len(ids)] = 1
        input_ids, attn = input_ids.to("cuda"), attn.to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
        stacked = torch.stack(out.hidden_states, dim=1).float()  # [batch, n_hs, seq, hidden]
        if pooling == "last":
            last_idx = attn.sum(1) - 1  # last non-pad position (right padding)
            rows = torch.arange(stacked.shape[0], device="cuda")
            picked = stacked[rows, :, last_idx, :]  # [batch, n_hs, hidden]
        elif pooling == "mean":
            if content_masks is not None:  # chat_masked: pool over content tokens only
                pool = torch.zeros_like(attn)
                for i, cm in enumerate(content_masks[start : start + batch_size]):
                    pool[i, : len(cm)] = torch.tensor(cm, dtype=torch.long, device="cuda")
            else:
                pool = attn
            m = pool[:, None, :, None].float()  # [batch,1,seq,1]
            picked = (stacked * m).sum(2) / m.sum(2).clamp(min=1)  # [batch, n_hs, hidden]
        else:
            raise ValueError(f"unknown representation {representation!r}")
        out_chunks.append(picked.cpu().numpy())
    return np.concatenate(out_chunks, axis=0)


def _format_prompt(template: str, text: str, a_char: str, b_char: str) -> str:
    return template.format(text=text, a=a_char, b=b_char)


def _label_ids(tokenizer, tok_a: str, tok_b: str) -> tuple[int, int]:
    a = tokenizer(tok_a, add_special_tokens=False)["input_ids"]
    b = tokenizer(tok_b, add_special_tokens=False)["input_ids"]
    assert len(a) == 1 and len(b) == 1, f"label tokens must be single Qwen3 tokens: {tok_a!r}->{a}, {tok_b!r}->{b}"
    return a[0], b[0]


def _eval_label_logits_pair(
    model, tokenizer, texts, template, a_char, b_char, a_id, b_id, batch_size=16
):
    """Return float32 ndarray [n_texts, 2] of (logit_a, logit_b) at the label position (last token
    of text+prompt). Softmax normalization cancels in the difference, so raw logits give the metric's
    d = logP(A) - logP(B) directly."""
    import numpy as np
    import torch

    prompts = [_format_prompt(template, t, a_char, b_char) for t in texts]
    all_ids = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]
    pad_id = tokenizer.pad_token_id
    out_rows = []
    for start in range(0, len(all_ids), batch_size):
        batch = all_ids[start : start + batch_size]
        maxlen = max(len(x) for x in batch)
        input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for i, ids in enumerate(batch):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            attn[i, : len(ids)] = 1
        input_ids, attn = input_ids.to("cuda"), attn.to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=input_ids, attention_mask=attn).logits
        last_idx = attn.sum(1) - 1
        rows = torch.arange(logits.shape[0], device="cuda")
        last_logits = logits[rows, last_idx, :]  # [batch, vocab]
        pair = torch.stack([last_logits[:, a_id], last_logits[:, b_id]], dim=1)  # [batch, 2]
        out_rows.append(pair.float().cpu().numpy())
    return np.concatenate(out_rows, axis=0)


def _training_signal(logits_pair, labels):
    """Diagonal (training-cell) signal used to choose LR/steps/checkpoint — NEVER the held-out cells.
    labels: 0 for class A, 1 for class B. Returns (accuracy, mean correct-label logprob)."""
    import numpy as np

    labels = np.asarray(labels)
    # log-softmax over the two label logits
    m = logits_pair.max(axis=1, keepdims=True)
    logZ = m[:, 0] + np.log(np.exp(logits_pair - m).sum(axis=1))
    logp = logits_pair - logZ[:, None]  # [n,2], col0=logP(A), col1=logP(B)
    pred = logits_pair.argmax(axis=1)  # 0=A,1=B
    acc = float((pred == labels).mean())
    correct_logp = float(logp[np.arange(len(labels)), labels].mean())
    return acc, correct_logp


def _heldout_summary(logits_pair, f1_signs):
    """Apply the shared metric to held-out (logit_a, logit_b) pairs. Returns the metric summary plus
    the raw oriented d-values for inspection."""
    import numpy as np

    d = logits_pair[:, 0] - logits_pair[:, 1]
    res = metric.logprob_difference(d, f1_signs)
    return {
        "raw": res.raw,
        "bounded_prob": res.bounded_prob,
        "hard_accuracy": res.hard_accuracy,
        "n": res.n,
        "d_values": [float(x) for x in d],  # per-text decision values (small; for data inspection)
    }


# ============================================================================= remote: activations
@app.function(image=image, gpu=GPU, volumes={HF_CACHE_DIR: HF_CACHE_VOL}, timeout=60 * 30)
def extract_activations(texts: list[str], wrap: str = "none", representation: str = "last") -> dict:
    """Reusable primitive (for later segments, e.g. RQ3 re-extraction): per-layer activations (last
    token or mean-pooled) for all layers. Returns the array as a nested list — WARNING: this is
    [n_texts, 29, 2048] floats, so call only on SMALL text batches; the probe uses probe_logprob_diff
    to fit on-GPU and avoid transferring big arrays."""
    start = time.monotonic()
    model, tokenizer = _load_model_and_tokenizer()
    model.eval()
    HF_CACHE_VOL.commit()
    acts = _extract_all_layer_repr(model, tokenizer, texts, wrap, representation)
    return {"activations": acts.tolist(), "shape": list(acts.shape), "elapsed": time.monotonic() - start}


@app.function(image=image, gpu=GPU, volumes={HF_CACHE_DIR: HF_CACHE_VOL}, timeout=60 * 30)
def extract_activations_blob(
    groups: dict[str, list[str]],
    representation: str = "last",
    wrap: str = "none",
    prompt_template: str | None = None,
    a_char: str | None = None,
    b_char: str | None = None,
    batch_size: int = 16,
    model_id: str | None = None,
    dtype: str | None = None,
) -> dict:
    """Extract all-layer activations for several named text groups in ONE call (model loads once),
    returning them as compressed .npz BYTES (one float32 [n_texts, 29, 2048] array per group).

    This is the slow/fast-split primitive (INSTRUCTIONS §4): extract once on the GPU, then fit every
    probe variant (controls, C, seeds, layers) locally on CPU. The activations are transient — the
    caller caches the .npz blob locally for iteration and never persists it as an artifact / to a Volume
    (§3). `representation` ∈ {last, mean, prompted}; for "prompted" pass the classify prompt_template +
    a_char/b_char so the read position matches the fine-tune."""
    import io

    import numpy as np

    start = time.monotonic()
    model, tokenizer = _load_model_and_tokenizer(dtype=dtype or DTYPE, model_name=model_id)
    model.eval()
    HF_CACHE_VOL.commit()
    arrays, shapes = {}, {}
    for name, txts in groups.items():
        acts = _extract_all_layer_repr(
            model, tokenizer, txts, wrap, representation, prompt_template, a_char, b_char, batch_size
        )
        arrays[name] = acts.astype(np.float32)
        shapes[name] = list(acts.shape)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return {"npz": buf.getvalue(), "shapes": shapes, "elapsed": time.monotonic() - start}


@app.function(image=image, gpu=GPU, volumes={HF_CACHE_DIR: HF_CACHE_VOL}, timeout=60 * 30)
def extract_signature_stats(
    groups: dict[str, list[str]],
    keep_layers: list[int] | None = None,
    wrap: str = "none",
    batch_size: int = 16,
) -> dict:
    """Compact per-cell activation summaries for the signature pipeline. ONE forward pass
    per batch; for each named group returns, per hidden-state layer, the MEAN activation under BOTH
    the last-non-pad-token read and the mean-over-tokens read (so a pair's whole signature comes from
    a single pass instead of two extract_activations_blob calls). Also returns per-text activations at
    `keep_layers` (both reads) for the within-cell text bootstrap (predictor stability / tie-band /
    sample-size curve). Tokenization is byte-identical to _extract_all_layer_repr (wrap='none',
    add_special_tokens=False) so the means match the pinned probe's read positions.

    Payload is small (means are [n_layers, hidden] per cell); the big per-text arrays are limited to
    `keep_layers`. Returned as compressed .npz BYTES; the caller caches the blob in the local
    activation cache and never persists it as an artifact (§3)."""
    import io

    import numpy as np
    import torch

    start = time.monotonic()
    model, tokenizer = _load_model_and_tokenizer()
    model.eval()
    HF_CACHE_VOL.commit()
    keep = list(keep_layers) if keep_layers is not None else []
    pad_id = tokenizer.pad_token_id
    arrays: dict[str, np.ndarray] = {}
    for name, texts in groups.items():
        all_ids = [_wrap_text(tokenizer, t, wrap)[0] for t in texts]
        sum_last = sum_mean = None
        n = 0
        pt_last, pt_mean = [], []
        for s in range(0, len(all_ids), batch_size):
            batch = all_ids[s : s + batch_size]
            maxlen = max(len(x) for x in batch)
            input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
            attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
            for i, ids in enumerate(batch):
                input_ids[i, : len(ids)] = torch.tensor(ids)
                attn[i, : len(ids)] = 1
            input_ids, attn = input_ids.to("cuda"), attn.to("cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
            stacked = torch.stack(out.hidden_states, dim=1).float()  # [b, n_hs, seq, H]
            last_idx = attn.sum(1) - 1
            rows = torch.arange(stacked.shape[0], device="cuda")
            last = stacked[rows, :, last_idx, :]  # [b, n_hs, H]
            m = attn[:, None, :, None].float()
            mean = (stacked * m).sum(2) / m.sum(2).clamp(min=1)  # [b, n_hs, H]
            sl, sm = last.sum(0).cpu().numpy(), mean.sum(0).cpu().numpy()
            sum_last = sl if sum_last is None else sum_last + sl
            sum_mean = sm if sum_mean is None else sum_mean + sm
            n += last.shape[0]
            if keep:
                pt_last.append(last[:, keep, :].cpu().numpy())
                pt_mean.append(mean[:, keep, :].cpu().numpy())
        arrays[f"{name}__mean_last"] = (sum_last / n).astype(np.float32)  # [n_hs, H]
        arrays[f"{name}__mean_mean"] = (sum_mean / n).astype(np.float32)
        arrays[f"{name}__n"] = np.array([n], dtype=np.int64)
        if keep:
            arrays[f"{name}__pertext_last"] = np.concatenate(pt_last, 0).astype(np.float32)  # [n, |keep|, H]
            arrays[f"{name}__pertext_mean"] = np.concatenate(pt_mean, 0).astype(np.float32)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return {"npz": buf.getvalue(), "elapsed": time.monotonic() - start, "keep_layers": keep,
            "n_by_group": {name: len(texts) for name, texts in groups.items()}}


@app.function(image=image, volumes={HF_CACHE_DIR: HF_CACHE_VOL}, timeout=60 * 10)
def chat_masked_layout(texts: list[str]) -> dict:
    """AUTHORITATIVE token layout of the 'chat_masked' read — runs with the image's pinned
    transformers + the cached tokenizer (CPU-only, no model load): per text, the wrapped chat
    string, tokens, offsets, and the content mask. Used by the phase's token-layout inspection
    guardrail before the B1 wave (phase chat_meanpooled_reads)."""
    from transformers import AutoTokenizer

    start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)
    per_text = []
    for t in texts:
        ids, mask = _chat_masked_ids(tokenizer, t)
        enc = tokenizer(
            tokenizer.apply_chat_template([{"role": "user", "content": t}],
                                          tokenize=False, add_generation_prompt=True),
            add_special_tokens=False, return_offsets_mapping=True)
        per_text.append({
            "tokens": tokenizer.convert_ids_to_tokens(ids),
            "offsets": [list(o) for o in enc["offset_mapping"]],
            "mask": mask,
            "n_content_tokens": sum(mask),
            "n_raw_text_tokens": len(tokenizer(t, add_special_tokens=False)["input_ids"]),
        })
    return {"per_text": per_text, "elapsed": time.monotonic() - start}


# ============================================================================= remote: probe
@app.function(image=image, gpu=GPU, volumes={HF_CACHE_DIR: HF_CACHE_VOL}, timeout=60 * 30)
def probe_logprob_diff(
    train_texts: list[str],
    train_labels: list[int],          # 0 = class A, 1 = class B
    heldout_texts: list[str],
    heldout_f1_signs: list[int],      # +1 if f1 is at its + value in the text, else -1
    seeds: list[int],
    wrap: str = "none",
    representation: str = "last",
) -> dict:
    """Fit LogisticRegression(C=1.0, max_iter=1000), no scaling, on a 75/25 split of the training
    cells; for every layer report the held-in validation cross-entropy/margin/accuracy (for global
    layer selection) and the held-out logprob difference. 5 seeds (INSTRUCTIONS §7)."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    start = time.monotonic()
    model, tokenizer = _load_model_and_tokenizer()
    model.eval()
    HF_CACHE_VOL.commit()

    # Extract once (activations don't depend on the probe seed), reuse across seeds.
    train_acts = _extract_all_layer_repr(model, tokenizer, train_texts, wrap, representation)  # [n_tr, L, H]
    held_acts = _extract_all_layer_repr(model, tokenizer, heldout_texts, wrap, representation)  # [n_ho, L, H]
    n_layers = train_acts.shape[1]
    y = np.asarray(train_labels)
    signs = np.asarray(heldout_f1_signs)

    per_seed = []
    for seed in seeds:
        tr_idx, val_idx = train_test_split(
            np.arange(len(y)), test_size=0.25, random_state=seed, stratify=y
        )
        layers = []
        for layer in range(n_layers):
            Xtr, Xval = train_acts[tr_idx, layer, :], train_acts[val_idx, layer, :]
            Xho = held_acts[:, layer, :]
            clf = LogisticRegression(C=1.0, max_iter=1000)
            clf.fit(Xtr, y[tr_idx])
            # Held-in validation: continuous tie-break for layer selection (accuracy saturates).
            val_dec = clf.decision_function(Xval)  # w·x + b, oriented toward class label 1 (=B)
            # Orient decision value toward class A: A is label 0, so logP(A)-logP(B) = -(w·x+b).
            # sklearn's decision_function is positive toward the '1' class (B). Flip sign so positive=A.
            eps = 1e-12
            p_b = np.clip(1.0 / (1.0 + np.exp(-val_dec)), eps, 1 - eps)  # P(class B), clamped
            val_ce = float(-np.mean(
                (y[val_idx] == 1) * np.log(p_b) + (y[val_idx] == 0) * np.log(1 - p_b)
            ))
            val_acc = float((clf.predict(Xval) == y[val_idx]).mean())
            val_margin = float(np.mean(np.abs(val_dec)))  # larger = more separated
            # Held-out: d = logP(A) - logP(B) = -(w·x+b)
            d = -clf.decision_function(Xho)
            res = metric.logprob_difference(d, signs)
            layers.append({
                "layer": layer,
                "val_ce": val_ce,
                "val_acc": val_acc,
                "val_margin": val_margin,
                "heldout_raw": res.raw,
                "heldout_bounded": res.bounded_prob,
                "heldout_acc": res.hard_accuracy,
            })
        per_seed.append({"seed": seed, "layers": layers})

    return {
        "per_seed": per_seed,
        "n_layers": int(n_layers),
        "n_train": len(train_texts),
        "n_heldout": len(heldout_texts),
        "elapsed": time.monotonic() - start,
    }


# ============================================================================= remote: baseline
@app.function(image=image, gpu=GPU, volumes={HF_CACHE_DIR: HF_CACHE_VOL}, timeout=60 * 30)
def baseline_logprob_diff(
    heldout_texts: list[str],
    heldout_f1_signs: list[int],
    template: str,
    tok_a: str,
    tok_b: str,
    train_texts: list[str] | None = None,
    train_labels: list[int] | None = None,
) -> dict:
    """Un-fine-tuned model's held-out logprob difference (the §2a reference) with the same prompt and
    tokens as the fine-tune. Optionally also report the base model's training-cell signal and its
    label prior."""
    import numpy as np

    start = time.monotonic()
    model, tokenizer = _load_model_and_tokenizer()
    model.eval()
    HF_CACHE_VOL.commit()
    a_char, b_char = tok_a.strip(), tok_b.strip()
    a_id, b_id = _label_ids(tokenizer, tok_a, tok_b)

    held_pair = _eval_label_logits_pair(model, tokenizer, heldout_texts, template, a_char, b_char, a_id, b_id)
    result = {"heldout": _heldout_summary(held_pair, heldout_f1_signs), "elapsed": time.monotonic() - start}
    pairs_for_prior = [held_pair]
    if train_texts is not None:
        tr_pair = _eval_label_logits_pair(model, tokenizer, train_texts, template, a_char, b_char, a_id, b_id)
        acc, logp = _training_signal(tr_pair, train_labels)
        result["train_acc"] = acc
        result["train_correct_logp"] = logp
        pairs_for_prior.append(tr_pair)
    # Base label prior: mean P(A) at the label position over the evaluated texts (token lopsidedness).
    all_pair = np.concatenate(pairs_for_prior, axis=0)
    d_all = all_pair[:, 0] - all_pair[:, 1]
    result["base_mean_p_a"] = float((1.0 / (1.0 + np.exp(-d_all))).mean())
    return result


# ============================================================================= remote: fine-tune
@app.function(image=image, gpu=GPU, volumes={HF_CACHE_DIR: HF_CACHE_VOL}, timeout=60 * 60)
def finetune_and_eval(
    train_texts: list[str],
    train_labels: list[int],          # 0 = class A, 1 = class B
    heldout_texts: list[str],
    heldout_f1_signs: list[int],
    template: str,
    tok_a: str,
    tok_b: str,
    lr: float,
    max_steps: int,
    seed: int,
    batch_size: int = 16,
    eval_steps: list[int] | None = None,
    head_only: bool = False,          # True => freeze all but lm_head (labeled sanity-check only)
    sample_weights: list[float] | None = None,  # per-train-example loss weights (length-bin reweighting, §6)
) -> dict:
    """Full fine-tune (all params) on the training cells: text+prompt -> one neutral label token,
    loss on the label token only. Reports, at each checkpoint, the training-cell (diagonal) signal —
    used to choose LR/steps/checkpoint — and the held-out logprob difference (NOT used for selection).

    sample_weights (INSTRUCTIONS §6, the length-handling tie-check): per-training-example loss weights.
    None => the canonical mean cross-entropy via HF's `.loss` (the full-A/B path, unchanged). When given,
    the per-example label-token CE is weighted by these (inverse-propensity length-bin weights computed
    on the host via probe_fit.length_bin_weights — the SAME control the probe uses, so the two methods'
    length controls are apples-to-apples). Weight-0 examples (class-pure length bins, a positivity
    violation) contribute no gradient — i.e. the reweighting "drops" them; the host reports the retained
    fraction. The loss is a per-batch WEIGHTED MEAN so its scale matches the unweighted run.
    """
    import numpy as np
    import torch

    start = time.monotonic()
    _set_determinism(seed)
    model, tokenizer = _load_model_and_tokenizer(attn="sdpa")  # fast attention for training
    HF_CACHE_VOL.commit()
    a_char, b_char = tok_a.strip(), tok_b.strip()
    a_id, b_id = _label_ids(tokenizer, tok_a, tok_b)
    label_token_ids = [a_id, b_id]
    if eval_steps is None:
        eval_steps = sorted(set([0, 25, 50, 100, 150, 200, 300, 400, max_steps]))
    eval_steps = [s for s in eval_steps if s <= max_steps]

    if head_only:
        for name, p in model.named_parameters():
            p.requires_grad = "lm_head" in name or "embed_tokens" in name  # tied embeddings in Qwen3

    # Build training sequences: input = prompt_ids + [label_id]; loss only on the label token.
    pad_id = tokenizer.pad_token_id
    seqs, lbls = [], []
    for text, label in zip(train_texts, train_labels):
        prompt = _format_prompt(template, text, a_char, b_char)
        pid = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        label_id = label_token_ids[label]
        seqs.append(pid + [label_id])
        lbls.append([-100] * len(pid) + [label_id])

    def make_batch(indices):
        batch_ids = [seqs[i] for i in indices]
        batch_lbl = [lbls[i] for i in indices]
        maxlen = max(len(x) for x in batch_ids)
        input_ids = torch.full((len(indices), maxlen), pad_id, dtype=torch.long)
        labels = torch.full((len(indices), maxlen), -100, dtype=torch.long)
        attn = torch.zeros((len(indices), maxlen), dtype=torch.long)
        for i, (ids, lb) in enumerate(zip(batch_ids, batch_lbl)):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            labels[i, : len(lb)] = torch.tensor(lb)
            attn[i, : len(ids)] = 1
        return input_ids.to("cuda"), attn.to("cuda"), labels.to("cuda")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    sw = None if sample_weights is None else torch.tensor(sample_weights, dtype=torch.float32, device="cuda")
    assert sw is None or len(sample_weights) == len(seqs), "sample_weights must align with train texts"
    rng = np.random.default_rng(seed)
    order: list[int] = []

    def next_indices(n):
        nonlocal order
        out = []
        while len(out) < n:
            if not order:
                order = list(rng.permutation(len(seqs)))
            out.append(order.pop())
        return out

    def evaluate(step):
        model.eval()
        tr_pair = _eval_label_logits_pair(model, tokenizer, train_texts, template, a_char, b_char, a_id, b_id)
        acc, logp = _training_signal(tr_pair, train_labels)
        held_pair = _eval_label_logits_pair(model, tokenizer, heldout_texts, template, a_char, b_char, a_id, b_id)
        ho = _heldout_summary(held_pair, heldout_f1_signs)
        model.train()
        return {
            "step": step,
            "train_acc": acc,
            "train_correct_logp": logp,
            "heldout_raw": ho["raw"],
            "heldout_bounded": ho["bounded_prob"],
            "heldout_acc": ho["hard_accuracy"],
            # Per-text decision values d = logit_a - logit_b at THIS checkpoint. Kept at EVERY eval step
            # (was final-only) so the per-held-cell / XOR-aware readout (INSTRUCTIONS §7) and the per-text
            # inspection can be computed locally at the pinned (training-signal) checkpoint, not just step
            # max_steps. heldout_texts is ordered [held_f1pos..., held_f1neg...] (dataset.build_inputs), so
            # the caller splits by heldout_f1_signs to get the two cells. Small (~n_held floats per step).
            "d_values": ho["d_values"],
        }

    curve = []
    train_losses = []
    model.train()
    if 0 in eval_steps:
        curve.append(evaluate(0))
    for step in range(1, max_steps + 1):
        idx = next_indices(batch_size)
        input_ids, attn, labels = make_batch(idx)
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):  # bf16 matmuls, fp32 params/optimizer
            if sw is None:
                loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss
            else:
                # Weighted CE on the single label token per example (length-bin reweighting). Replicate
                # HF's causal shift, take the one non-ignored (label) token's CE per example, and form a
                # per-batch WEIGHTED MEAN. Weight-0 examples drop out of both numerator and denominator.
                logits = model(input_ids=input_ids, attention_mask=attn).logits
                ce = torch.nn.functional.cross_entropy(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100, reduction="none",
                ).view(len(idx), -1).sum(1)  # exactly one non-ignored token per example
                w = sw[torch.tensor(idx, device="cuda")]
                loss = (ce * w).sum() / w.sum().clamp(min=1e-6)
        loss.backward()
        opt.step()
        train_losses.append(float(loss.item()))
        if step in eval_steps:
            curve.append(evaluate(step))

    return {
        "curve": curve,
        "train_losses": train_losses,
        "lr": lr,
        "max_steps": max_steps,
        "seed": seed,
        "batch_size": batch_size,
        "head_only": head_only,
        "n_train": len(train_texts),
        "n_heldout": len(heldout_texts),
        "elapsed": time.monotonic() - start,
    }


# ============================================================================= K-class (multi-class) entry points
# The K-class analogs of the binary probe/baseline/fine-tune (INSTRUCTIONS §6). New functions (the binary
# ones are untouched / still cached). modal_app.py is NOT baked into the image, so adding these does not
# change image.object_id; each new function gets its own call_uuid in the host cache key.
def _label_ids_kway(tokenizer, label_tokens: list[str]) -> list[int]:
    ids = []
    for t in label_tokens:
        x = tokenizer(t, add_special_tokens=False)["input_ids"]
        assert len(x) == 1, f"label token must be a single Qwen3 token: {t!r}->{x}"
        ids.append(x[0])
    return ids


def _eval_label_logits_kway(model, tokenizer, texts, template, label_ids, batch_size=16):
    """Return float32 [n_texts, K] of the K label-token logits at the label position (last token of
    text+prompt). `template` is the K-class classify template (label letters baked in, only {text})."""
    import numpy as np
    import torch

    prompts = [template.format(text=t) for t in texts]
    all_ids = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]
    pad_id = tokenizer.pad_token_id
    label_ids_t = torch.tensor(label_ids, device="cuda")
    out_rows = []
    for start in range(0, len(all_ids), batch_size):
        batch = all_ids[start : start + batch_size]
        maxlen = max(len(x) for x in batch)
        input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for i, ids in enumerate(batch):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            attn[i, : len(ids)] = 1
        input_ids, attn = input_ids.to("cuda"), attn.to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=input_ids, attention_mask=attn).logits
        last_idx = attn.sum(1) - 1
        rows = torch.arange(logits.shape[0], device="cuda")
        picked = logits[rows, last_idx, :][:, label_ids_t]  # [batch, K]
        out_rows.append(picked.float().cpu().numpy())
    return np.concatenate(out_rows, axis=0)


def _training_signal_kway(class_logits, labels):
    """K-way diagonal (training-cell) signal for checkpoint/LR selection — NEVER the held-out cells.
    labels in 0..K-1. Returns (accuracy, mean correct-label logprob over the K label tokens)."""
    import numpy as np

    labels = np.asarray(labels)
    m = class_logits.max(axis=1, keepdims=True)
    logZ = (m[:, 0] + np.log(np.exp(class_logits - m).sum(axis=1)))
    logp = class_logits - logZ[:, None]
    pred = class_logits.argmax(axis=1)
    return float((pred == labels).mean()), float(logp[np.arange(len(labels)), labels].mean())


def _base_label_probs(class_logits):
    """Mean softmax over the K label tokens (the base K-way prior; lopsidedness check, §6)."""
    import numpy as np

    m = class_logits.max(axis=1, keepdims=True)
    sm = np.exp(class_logits - m)
    sm = sm / sm.sum(axis=1, keepdims=True)
    return sm.mean(axis=0).tolist()


@app.function(image=image, gpu=GPU, volumes={HF_CACHE_DIR: HF_CACHE_VOL}, timeout=60 * 20)
def kclass_label_prior(texts: list[str], template: str, label_tokens: list[str]) -> dict:
    """Base model's mean softmax over the K label tokens at the classify position (no fine-tune). Used to
    pick K neutral label tokens the base model is roughly balanced over (INSTRUCTIONS §6, like Stage 0's
    X/Y prior)."""
    start = time.monotonic()
    model, tokenizer = _load_model_and_tokenizer()
    model.eval()
    HF_CACHE_VOL.commit()
    label_ids = _label_ids_kway(tokenizer, label_tokens)
    logits = _eval_label_logits_kway(model, tokenizer, texts, template, label_ids)
    return {"base_label_probs": _base_label_probs(logits), "n": len(texts),
            "label_tokens": label_tokens, "elapsed": time.monotonic() - start}


@app.function(image=image, gpu=GPU, volumes={HF_CACHE_DIR: HF_CACHE_VOL}, timeout=60 * 30)
def kclass_baseline_logprob_diff(
    heldout_texts: list[str],
    heldout_f1_class: list[int],
    heldout_f2_class: list[int],
    template: str,
    label_tokens: list[str],
    train_texts: list[str] | None = None,
    train_labels: list[int] | None = None,
) -> dict:
    """Un-fine-tuned model's held-out multi-class logprob difference (the §7 reference), same prompt +
    K label tokens as the fine-tune. Returns the pooled metric, the per-text K class-logits (for the host
    per-cell readout), the base K-way prior, and optionally the base model's K-way training-cell signal."""
    import numpy as np

    start = time.monotonic()
    model, tokenizer = _load_model_and_tokenizer()
    model.eval()
    HF_CACHE_VOL.commit()
    label_ids = _label_ids_kway(tokenizer, label_tokens)
    held = _eval_label_logits_kway(model, tokenizer, heldout_texts, template, label_ids)
    res = metric.kclass_logprob_difference(held, heldout_f1_class, heldout_f2_class)
    result = {
        "heldout": {"raw": res.raw, "bounded_prob": res.bounded_prob, "hard_accuracy": res.hard_accuracy,
                    "n": res.n, "class_logits": held.tolist()},
        "elapsed": time.monotonic() - start,
    }
    pools = [held]
    if train_texts is not None:
        tr = _eval_label_logits_kway(model, tokenizer, train_texts, template, label_ids)
        acc, logp = _training_signal_kway(tr, train_labels)
        result["train_acc"] = acc
        result["train_correct_logp"] = logp
        pools.append(tr)
    result["base_label_probs"] = _base_label_probs(np.concatenate(pools, axis=0))
    return result


@app.function(image=image, gpu=GPU, volumes={HF_CACHE_DIR: HF_CACHE_VOL}, timeout=60 * 60)
def kclass_finetune_and_eval(
    train_texts: list[str],
    train_labels: list[int],           # 0..K-1 (diagonal class)
    heldout_texts: list[str],
    heldout_f1_class: list[int],
    heldout_f2_class: list[int],
    template: str,
    label_tokens: list[str],
    lr: float,
    max_steps: int,
    seed: int,
    batch_size: int = 16,
    eval_steps: list[int] | None = None,
) -> dict:
    """Full K-way fine-tune (all params) on the diagonal cells: text+prompt -> one of the K neutral label
    tokens, loss on the label token only. Reports, at each checkpoint, the K-way training-cell signal
    (for LR/steps selection) and the held-out multi-class logprob difference + per-text K class-logits
    (NOT used for selection). The K-class analog of finetune_and_eval."""
    import numpy as np
    import torch

    start = time.monotonic()
    _set_determinism(seed)
    model, tokenizer = _load_model_and_tokenizer(attn="sdpa")
    HF_CACHE_VOL.commit()
    label_ids = _label_ids_kway(tokenizer, label_tokens)
    if eval_steps is None:
        eval_steps = sorted(set([0, 25, 50, 75, 100, 150, 200, max_steps]))
    eval_steps = [s for s in eval_steps if s <= max_steps]

    pad_id = tokenizer.pad_token_id
    seqs, lbls = [], []
    for text, label in zip(train_texts, train_labels):
        pid = tokenizer(template.format(text=text), add_special_tokens=False)["input_ids"]
        lid = label_ids[label]
        seqs.append(pid + [lid])
        lbls.append([-100] * len(pid) + [lid])

    def make_batch(indices):
        batch_ids = [seqs[i] for i in indices]
        batch_lbl = [lbls[i] for i in indices]
        maxlen = max(len(x) for x in batch_ids)
        input_ids = torch.full((len(indices), maxlen), pad_id, dtype=torch.long)
        labels = torch.full((len(indices), maxlen), -100, dtype=torch.long)
        attn = torch.zeros((len(indices), maxlen), dtype=torch.long)
        for i, (ids, lb) in enumerate(zip(batch_ids, batch_lbl)):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            labels[i, : len(lb)] = torch.tensor(lb)
            attn[i, : len(ids)] = 1
        return input_ids.to("cuda"), attn.to("cuda"), labels.to("cuda")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = np.random.default_rng(seed)
    order: list[int] = []

    def next_indices(n):
        nonlocal order
        out = []
        while len(out) < n:
            if not order:
                order = list(rng.permutation(len(seqs)))
            out.append(order.pop())
        return out

    def evaluate(step):
        model.eval()
        tr = _eval_label_logits_kway(model, tokenizer, train_texts, template, label_ids)
        acc, logp = _training_signal_kway(tr, train_labels)
        held = _eval_label_logits_kway(model, tokenizer, heldout_texts, template, label_ids)
        res = metric.kclass_logprob_difference(held, heldout_f1_class, heldout_f2_class)
        model.train()
        return {"step": step, "train_acc": acc, "train_correct_logp": logp,
                "heldout_raw": res.raw, "heldout_bounded": res.bounded_prob, "heldout_acc": res.hard_accuracy,
                "class_logits": held.tolist()}  # per-text K-logits at THIS step (host per-cell readout)

    curve, train_losses = [], []
    model.train()
    if 0 in eval_steps:
        curve.append(evaluate(0))
    for step in range(1, max_steps + 1):
        idx = next_indices(batch_size)
        input_ids, attn, labels = make_batch(idx)
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss
        loss.backward()
        opt.step()
        train_losses.append(float(loss.item()))
        if step in eval_steps:
            curve.append(evaluate(step))

    return {"curve": curve, "train_losses": train_losses, "lr": lr, "max_steps": max_steps, "seed": seed,
            "batch_size": batch_size, "n_train": len(train_texts), "n_heldout": len(heldout_texts),
            "K": len(label_tokens), "elapsed": time.monotonic() - start}


# ============================================================================= remote: token NLL
@app.function(image=image, gpu=GPU, volumes={HF_CACHE_DIR: HF_CACHE_VOL}, timeout=60 * 30)
def token_nll(texts: list[str], batch_size: int = 16) -> dict:
    """Per-token base-model negative log-likelihood (natural log) for each raw text (no chat
    template, no special tokens — the probe/fine-tune 'none' wrapping). Returns, per text, the
    token strings, their character offsets, and their NLL (the model's surprisal at predicting
    each token given the prefix; the first token has no prediction context and is omitted, i.e.
    offsets/nll align to tokens[1:]). Used by the footprint/anomaly regression to measure the mean
    surprisal of a feature's distinctive tokens. Small numbers only; no activations persisted."""
    import numpy as np
    import torch

    start = time.monotonic()
    model, tokenizer = _load_model_and_tokenizer()
    model.eval()
    HF_CACHE_VOL.commit()
    pad_id = tokenizer.pad_token_id
    out = []
    for s in range(0, len(texts), batch_size):
        batch = texts[s : s + batch_size]
        encs = [tokenizer(t, add_special_tokens=False, return_offsets_mapping=True) for t in batch]
        ids = [e["input_ids"] for e in encs]
        maxlen = max(len(x) for x in ids)
        input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for i, x in enumerate(ids):
            input_ids[i, : len(x)] = torch.tensor(x)
            attn[i, : len(x)] = 1
        input_ids, attn = input_ids.to("cuda"), attn.to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=input_ids, attention_mask=attn).logits.float()
        logp = torch.log_softmax(logits, dim=-1)
        for i, x in enumerate(ids):
            n = len(x)
            # NLL of token j (j>=1) = -logp[i, j-1, x[j]]
            tgt = torch.tensor(x[1:], device="cuda")
            nll = -logp[i, torch.arange(n - 1, device="cuda"), tgt]
            toks = tokenizer.convert_ids_to_tokens(x)
            out.append({
                "tokens": toks[1:],
                "offsets": [list(o) for o in encs[i]["offset_mapping"][1:]],
                "nll": nll.cpu().numpy().round(4).tolist(),
            })
    return {"per_text": out, "n": len(texts), "elapsed": time.monotonic() - start}


if __name__ == "__main__":
    # quick smoke of the real entry points on toy data
    from pathlib import Path
    from cost_tracker import CostTracker

    tracker = CostTracker(Path("total_cost.jsonl"), run_description="modal_app self-test")
    toy_train = [
        "def f(x): return x*2", "The cat sat on the mat quietly.",
        "for i in range(3): print(i)", "Birds sing in the morning light.",
        "import os; print(os.getcwd())", "She walked along the river bank.",
        "while True: break", "The garden was full of roses.",
    ]
    toy_labels = [0, 1, 0, 1, 0, 1, 0, 1]
    toy_held = ["x = [1,2,3]; print(sum(x))", "A quiet afternoon by the lake.", "y = 5\nprint(y+1)", "Snow fell softly."]
    toy_signs = [1, -1, 1, -1]
    with modal.enable_output(), app.run():
        with tracker.track_modal_gpu(gpu=GPU, is_sandbox=False) as t:
            r = probe_logprob_diff.remote(toy_train, toy_labels, toy_held, toy_signs, seeds=[0, 1])
            t.elapsed = r["elapsed"]
        print("probe n_layers:", r["n_layers"], "seed0 layer0:", r["per_seed"][0]["layers"][0])
        with tracker.track_modal_gpu(gpu=GPU, is_sandbox=False) as t:
            b = baseline_logprob_diff.remote(toy_held, toy_signs, "{text}\n\nCategory {a} or {b}?\nAnswer:", " A", " B")
            t.elapsed = b["elapsed"]
        print("baseline heldout raw:", b["heldout"]["raw"])
    print(f"cost: ${tracker.run_cost:.4f}")
