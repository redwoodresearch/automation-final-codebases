"""Modal app for the Qwen3.6-27B scale-check (qwen36_common.py). ADDITIVE: a separate app +
image; the study image (transformers 4.56.2) is untouched.

Why a second image: Qwen3.6-27B is model_type "qwen3_5" (hybrid Gated DeltaNet linear attention
3:1 with gated full attention, multimodal wrapper), supported by transformers >= 5.2 only
("the latest transformers is required" per the model card; TRT-LLM tracking issue pins v5.2+).
Pins here: torch==2.8.0 (cu128, B200), transformers==5.14.1, flash-linear-attention (fla triton
kernels for the DeltaNet path; without them transformers silently falls back to a much slower
pure-torch path), bitsandbytes (8-bit AdamW for the 27B full fine-tune), and the SAME
scikit-learn==1.7.2 / numpy==2.2.6 as the study so the probe fit instrument is bit-identical.

Text-only loading uses Qwen3_5ForCausalLM (the documented text path for multimodal Qwen3.5/3.6
checkpoints; the vision tower + MTP weights are simply not loaded).

Precision/optimizer at 27B (single B200, 192 GB): the study's fp32-params-AdamW recipe needs
~432 GB at 27B, so the 27B fine-tune runs bf16 params (the checkpoint's native dtype) +
bitsandbytes AdamW8bit + gradient checkpointing (~162 GB states). The instrument-validation gate
quantifies this deviation on 1.7B: (a) study recipe (fp32 + torch AdamW) under THIS image vs the
committed Δs isolates the image/version change; (b) bf16 + AdamW8bit on the same pairs isolates
the precision/optimizer change. Everything else (classify template, " X"/" Y" single-token label
loss after "Answer:", batch order rng, plateau-checkpoint convention, eval code) is ported
verbatim from knockout_modal.py / lib/modal_app.py.
"""

from __future__ import annotations

import os
import time

import modal

GPU = "B200"
HF_CACHE_VOL = modal.Volume.from_name("derisk-hf-cache", create_if_missing=True)
HF_CACHE_DIR = "/hf_cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "transformers==5.14.1",
        "accelerate==1.14.0",
        "scikit-learn==1.7.2",   # study pin: probe instrument identical
        "numpy==2.2.6",          # study pin
        "attrs==25.4.0",
        "bitsandbytes==0.50.0",
        "flash-linear-attention==0.5.1",
        "huggingface_hub[hf_transfer]",
    )
    .env({
        "HF_HOME": HF_CACHE_DIR,
        "HF_XET_HIGH_PERFORMANCE": "1",     # 56 GB checkpoint; fast Xet path for the one-time prefetch
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
    })
    .add_local_python_source("probe_fit", "metric")
)

app = modal.App("spurious-qwen36-27b")

# No HF auth: the model is public and the HF_TOKEN in ~/automation/.env is EXPIRED (an expired
# token causes 401s even on public repos — verified on the first run attempt).
SECRETS: list = []

TRAIN_LOSS_PLATEAU_THRESH = 1e-3   # study plateau convention (lib/finetune.py)
PLATEAU_WINDOW = 10


# ============================================================================= helpers (remote)
def _load_model_and_tokenizer(model_name: str, dtype: str, grad_ckpt: bool):
    """Text-only causal LM. Qwen3.6-27B needs the Qwen3_5ForCausalLM text path; everything else
    (Qwen3-1.7B for validation) goes through AutoModelForCausalLM. sdpa attention (the study's
    fine-tune path used sdpa; qwen3_5's DeltaNet layers use fla kernels when available)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = dict(cache_dir=HF_CACHE_DIR, dtype=getattr(torch, dtype), attn_implementation="sdpa")
    if "qwen3.5" in model_name.lower() or "qwen3.6" in model_name.lower():
        from transformers import Qwen3_5ForCausalLM
        model = Qwen3_5ForCausalLM.from_pretrained(model_name, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model = model.to("cuda")
    if grad_ckpt:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    return model, tokenizer


def _set_determinism(seed: int):
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _format_prompt(template: str, text: str, a_char: str, b_char: str) -> str:
    return template.format(text=text, a=a_char, b=b_char)


def _jsonable(obj):
    """JSON round-trip a return value REMOTELY so no torch/numpy/transformers object can leak
    into the Modal result payload (the local driver env has no torch; a non-plain object fails
    HERE with a clear TypeError instead of a local DeserializationError)."""
    import json

    return json.loads(json.dumps(obj))


def _encode_ids(tokenizer, text: str) -> list[int]:
    """Tokenize with the study convention (add_special_tokens=False) and coerce to a plain
    list[int] — transformers 5.x tokenizer outputs are not guaranteed to be python lists, and
    torch objects must never leak into Modal return values (the local driver env has no torch)."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [int(x) for x in ids]


def _label_ids(tokenizer, tok_a: str, tok_b: str) -> tuple[int, int]:
    a = _encode_ids(tokenizer, tok_a)
    b = _encode_ids(tokenizer, tok_b)
    assert len(a) == 1 and len(b) == 1, f"label tokens must be single tokens: {tok_a!r}->{a}, {tok_b!r}->{b}"
    return a[0], b[0]


def _eval_label_logits_pair(model, tokenizer, texts, template, a_char, b_char, a_id, b_id, batch_size=16):
    """Study convention (modal_app._eval_label_logits_pair): (logit_a, logit_b) at the label
    position (last token of text+prompt). Ported verbatim."""
    import numpy as np
    import torch

    prompts = [_format_prompt(template, t, a_char, b_char) for t in texts]
    all_ids = [_encode_ids(tokenizer, p) for p in prompts]
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
        last_logits = logits[rows, last_idx, :]
        pair = torch.stack([last_logits[:, a_id], last_logits[:, b_id]], dim=1)
        out_rows.append(pair.float().cpu().numpy())
    return np.concatenate(out_rows, axis=0)


def _training_signal(logits_pair, labels):
    import numpy as np

    labels = np.asarray(labels)
    m = logits_pair.max(axis=1, keepdims=True)
    logZ = m[:, 0] + np.log(np.exp(logits_pair - m).sum(axis=1))
    logp = logits_pair - logZ[:, None]
    pred = logits_pair.argmax(axis=1)
    acc = float((pred == labels).mean())
    correct_logp = float(logp[np.arange(len(labels)), labels].mean())
    return acc, correct_logp


# ============================================================================= prefetch + env report
@app.function(image=image, volumes={HF_CACHE_DIR: HF_CACHE_VOL}, secrets=SECRETS, timeout=60 * 60, cpu=8, memory=16384)
def prefetch_model(model_name: str) -> dict:
    """One-time snapshot download to the shared HF cache volume (so 10+ concurrent GPU
    containers don't each pull 56 GB from the Hub)."""
    from huggingface_hub import snapshot_download

    start = time.monotonic()
    path = snapshot_download(model_name)
    HF_CACHE_VOL.commit()
    return _jsonable({"model": model_name, "path": str(path), "elapsed": time.monotonic() - start})


@app.function(image=image, volumes={HF_CACHE_DIR: HF_CACHE_VOL}, secrets=SECRETS, timeout=15 * 60, cpu=4, memory=8192)
def env_and_tokenizer_report(model_name: str, tok_a: str, tok_b: str, template: str) -> dict:
    """Version report + label-token single-token check + template render check under the NEW image."""
    import importlib

    import transformers
    import torch
    from transformers import AutoTokenizer

    versions = {"transformers": transformers.__version__, "torch": torch.__version__}
    for mod in ("fla", "bitsandbytes", "sklearn", "numpy"):
        try:
            m = importlib.import_module(mod)
            versions[mod] = getattr(m, "__version__", "present")
        except Exception as e:  # noqa: BLE001
            versions[mod] = f"MISSING ({e})"

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    HF_CACHE_VOL.commit()
    a = _encode_ids(tokenizer, tok_a)
    b = _encode_ids(tokenizer, tok_b)
    rendered = template.format(text="SAMPLE.", a=tok_a.strip(), b=tok_b.strip())
    raw_ids = tokenizer(tok_a, add_special_tokens=False)["input_ids"]
    return _jsonable({
        "model": model_name,
        "versions": versions,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_output_type": type(raw_ids).__name__,   # diagnosis: what leaked torch earlier
        "label_ids": {tok_a: a, tok_b: b},
        "labels_single_token": bool(len(a) == 1 and len(b) == 1),
        "pad_token": str(tokenizer.pad_token),
        "rendered_template_tail": rendered[-90:],
    })


# ============================================================================= fine-tune + eval
@app.function(image=image, gpu=GPU, volumes={HF_CACHE_DIR: HF_CACHE_VOL}, secrets=SECRETS,
              timeout=3 * 60 * 60, memory=98304, retries=modal.Retries(max_retries=2, initial_delay=30.0))
def finetune_and_eval(
    model_name: str,
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
    dtype: str = "bfloat16",          # "float32" reproduces the study recipe (1.7B validation)
    optimizer: str = "adamw8bit",     # "adamw_fp32" = torch AdamW (study recipe)
    grad_ckpt: bool = True,
    batch_size: int = 16,
    eval_steps: list[int] | None = None,
) -> dict:
    """The study's fine-tune recipe (knockout_modal.finetune_save_and_eval), generalized over
    model/precision/optimizer, WITHOUT checkpoint saving: training stops at the plateau step
    (first eval step whose trailing-10 mean per-batch loss < 1e-3) and the Δ at that step is the
    committed readout. The step-0 eval doubles as the un-fine-tuned baseline anchor. Training
    math, seeding, batch order and eval code are the study's, verbatim."""
    import numpy as np
    import torch

    start = time.monotonic()
    _set_determinism(seed)
    model, tokenizer = _load_model_and_tokenizer(model_name, dtype, grad_ckpt)
    HF_CACHE_VOL.commit()
    load_elapsed = time.monotonic() - start
    a_char, b_char = tok_a.strip(), tok_b.strip()
    a_id, b_id = _label_ids(tokenizer, tok_a, tok_b)
    label_token_ids = [a_id, b_id]
    if eval_steps is None:
        eval_steps = [0, 25, 50, 75, 100, 150, 200]  # finetune.FT_EVAL_STEPS
    eval_steps = [s for s in eval_steps if s <= max_steps]

    pad_id = tokenizer.pad_token_id
    seqs, lbls = [], []
    for text, label in zip(train_texts, train_labels):
        prompt = _format_prompt(template, text, a_char, b_char)
        pid = _encode_ids(tokenizer, prompt)
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

    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer == "adamw_fp32":
        opt = torch.optim.AdamW(params, lr=lr)
    elif optimizer == "adamw8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=lr)
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")

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
        d = (held_pair[:, 0] - held_pair[:, 1])
        oriented = d * np.asarray(heldout_f1_signs, dtype=np.float64)
        model.train()
        return {
            "step": step,
            "train_acc": acc,
            "train_correct_logp": logp,
            "heldout_raw": float(oriented.mean()),
            "heldout_bounded": float((1.0 / (1.0 + np.exp(-oriented))).mean()),
            "heldout_acc": float((oriented > 0).mean()),
            "d_values": [float(x) for x in d],
        }

    curve, train_losses = [], []
    plateau_step = None
    model.train()
    if 0 in eval_steps:
        curve.append(evaluate(0))
    for step in range(1, max_steps + 1):
        idx = next_indices(batch_size)
        input_ids, attn, labels = make_batch(idx)
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss
        assert torch.isfinite(loss), f"non-finite training loss at step {step}"
        loss.backward()
        opt.step()
        train_losses.append(float(loss.item()))
        if step in eval_steps:
            curve.append(evaluate(step))
            tl = np.asarray(train_losses)
            if tl[max(0, step - PLATEAU_WINDOW): step].mean() < TRAIN_LOSS_PLATEAU_THRESH:
                plateau_step = step
                break

    saved_step = plateau_step if plateau_step is not None else max_steps
    return _jsonable({
        "model": model_name,
        "curve": curve,
        "train_losses": train_losses,
        "plateau_step": plateau_step,
        "saved_step": saved_step,
        "lr": lr,
        "max_steps": max_steps,
        "seed": seed,
        "dtype": dtype,
        "optimizer": optimizer,
        "grad_ckpt": grad_ckpt,
        "batch_size": batch_size,
        "n_train": len(train_texts),
        "n_heldout": len(heldout_texts),
        "load_elapsed": load_elapsed,
        "max_mem_gib": float(torch.cuda.max_memory_allocated() / 2**30),
        "elapsed": time.monotonic() - start,
    })


# ============================================================================= probes (base model)
_MODEL_CACHE: dict = {}  # warm-container reuse across pairs (base model is never mutated)


@app.function(image=image, gpu=GPU, volumes={HF_CACHE_DIR: HF_CACHE_VOL}, secrets=SECRETS,
              timeout=2 * 60 * 60, memory=98304, max_containers=8, retries=modal.Retries(max_retries=2, initial_delay=30.0))
def probe_extract_fit(
    model_name: str,
    pair: str,
    groups: dict[str, list[str]],
    lengths: dict[str, list[int]],
    seeds_by_cell: dict[str, list],
    signs: list[int],
    template: str,
    tok_a: str,
    tok_b: str,
    probe_c: float = 0.1,
    n_seeds: int = 5,
    batch_size: int = 16,
) -> dict:
    """BASE-model probes for one pair, both study representations, in one call (activations never
    leave the container — the scaffoldpool_full pattern):

      * "meanpooled" — bare passage text (add_special_tokens=False), mean over all real tokens,
        every hidden state (embedding + all layers). The study's mean_pooled read.
      * "decision"   — rendered classify prompt, last real token (the study's 'prompted' /
        decision-token read). This pass also yields the un-fine-tuned baseline held logprob diff
        (logit(" X") - logit(" Y") at the decision position) as the sanity anchor.

    Fit = the frozen probe instrument: probe_fit.fit_grid, uncontrolled, C=probe_c, n_seeds seeds,
    every hidden state; reported layer chosen by held-in val-CE (NaN CE sorts last, the
    scaffoldpool_full convention)."""
    import numpy as np
    import torch

    import metric
    import probe_fit

    start = time.monotonic()
    key = model_name
    if key not in _MODEL_CACHE:
        _MODEL_CACHE.clear()
        _MODEL_CACHE[key] = _load_model_and_tokenizer(model_name, "bfloat16", grad_ckpt=False)
        HF_CACHE_VOL.commit()
    model, tokenizer = _MODEL_CACHE[key]
    model.eval()
    a_char, b_char = tok_a.strip(), tok_b.strip()
    a_id, b_id = _label_ids(tokenizer, tok_a, tok_b)
    pad_id = tokenizer.pad_token_id
    cells = list(groups.keys())

    def forward_pool(all_ids, pooling: str, want_label_logits: bool):
        """Batched forwards -> (pooled [n, n_hs, hidden] float32, label logit pairs or None)."""
        pooled_chunks, logit_rows = [], []
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
            stacked = torch.stack(out.hidden_states, dim=1).float()  # [b, n_hs, seq, hidden]
            last_idx = attn.sum(1) - 1
            rows = torch.arange(stacked.shape[0], device="cuda")
            if pooling == "last":
                picked = stacked[rows, :, last_idx, :]
            else:  # mean over real tokens
                m = attn[:, None, :, None].float()
                picked = (stacked * m).sum(2) / m.sum(2).clamp(min=1)
            pooled_chunks.append(picked.cpu().numpy())
            if want_label_logits:
                last_logits = out.logits[rows, last_idx, :]
                pr = torch.stack([last_logits[:, a_id], last_logits[:, b_id]], dim=1)
                logit_rows.append(pr.float().cpu().numpy())
            del out, stacked
        pooled = np.concatenate(pooled_chunks, axis=0).astype(np.float32)
        logits = np.concatenate(logit_rows, axis=0) if want_label_logits else None
        return pooled, logits

    # ---- extraction ------------------------------------------------------------------
    acts = {"meanpooled": {}, "decision": {}}
    held_d: dict[str, list[float]] = {}
    for cell in cells:
        texts = groups[cell]
        bare_ids = [_encode_ids(tokenizer, t) for t in texts]
        acts["meanpooled"][cell], _ = forward_pool(bare_ids, "mean", want_label_logits=False)
        prompt_ids = [_encode_ids(tokenizer, _format_prompt(template, t, a_char, b_char))
                      for t in texts]
        acts["decision"][cell], lg = forward_pool(prompt_ids, "last", want_label_logits=True)
        held_d[cell] = [float(x) for x in (lg[:, 0] - lg[:, 1])]
    extract_elapsed = time.monotonic() - start

    # ---- un-fine-tuned baseline anchor (held cells, study metric) ---------------------
    d_held = np.array(held_d["held_f1pos"] + held_d["held_f1neg"])
    base = metric.logprob_difference(d_held, np.asarray(signs))
    baseline = {"raw": base.raw, "bounded": base.bounded_prob, "hard": base.hard_accuracy, "n": base.n}

    # ---- frozen probe fit per representation ------------------------------------------
    def fit_rep(rep: str) -> dict:
        a = acts[rep]
        n_hs = a["A"].shape[1]
        g = probe_fit.fit_grid(
            a, lengths, seeds_by_cell, np.asarray(signs),
            [probe_fit.Control("uncontrolled")], [probe_c], list(range(n_hs)), n_seeds=n_seeds,
        )
        layers = g["uncontrolled"][f"C={probe_c:g}"]["layers"]
        per_layer = {int(L): {"raw_mean": v["raw_mean"], "raw_std": v["raw_std"],
                              "val_ce_mean": v["val_ce_mean"], "val_acc_mean": v["val_acc_mean"],
                              "raw_seeds": v["raw_seeds"]}
                     for L, v in layers.items()}

        def _ce_key(L):  # NaN CE sorts last (scaffoldpool_full convention)
            ce = per_layer[L]["val_ce_mean"]
            return (1, 0.0) if np.isnan(ce) else (0, ce)

        sel = min(per_layer, key=_ce_key)
        return {
            "selected_layer": sel,
            "n_hidden_states": n_hs,
            "raw_selected": per_layer[sel]["raw_mean"],
            "raw_selected_std": per_layer[sel]["raw_std"],
            "raw_selected_seeds": per_layer[sel]["raw_seeds"],
            "val_acc_selected": per_layer[sel]["val_acc_mean"],
            "val_ce_selected": per_layer[sel]["val_ce_mean"],
            "per_layer": per_layer,
        }

    reps_out = {rep: fit_rep(rep) for rep in ("meanpooled", "decision")}

    return _jsonable({
        "pair": pair,
        "model": model_name,
        "representations": reps_out,
        "baseline_heldout": baseline,
        "n_per_cell": {c: len(groups[c]) for c in cells},
        "probe_c": probe_c,
        "n_probe_seeds": n_seeds,
        "elapsed_extract": extract_elapsed,
        "elapsed": time.monotonic() - start,
    })


# ============================================================================= named (explicit-prompting) mean-pooled probe
@app.function(image=image, gpu=GPU, volumes={HF_CACHE_DIR: HF_CACHE_VOL}, secrets=SECRETS,
              timeout=2 * 60 * 60, memory=98304, max_containers=8, retries=modal.Retries(max_retries=2, initial_delay=30.0))
def probe_named_meanpooled(
    model_name: str,
    pair: str,
    rendered_groups: dict[str, list[str]],
    lengths: dict[str, list[int]],
    seeds_by_cell: dict[str, list],
    signs: list[int],
    probe_c: float = 0.1,
    n_seeds: int = 5,
    batch_size: int = 16,
) -> dict:
    """Explicit-prompting (named) BASE-model MEAN-POOLED probe for one pair, mirroring the 1.7B
    inoculation probe (experiments/complete_graph/run_inoculation_probe.py): the naming prompt for
    the pair's safety behavior is already baked into every text of `rendered_groups` (via
    naming.render / the instruction-first builder LOCALLY); this call mean-pools the base model's
    hidden states over ALL real tokens of the rendered text (add_special_tokens=False), every hidden
    state, and fits the SAME frozen probe instrument as the neutral run (probe_fit.fit_grid,
    uncontrolled, C=probe_c, n_seeds seeds). No training, no label logits — meanpooled only.

    Returns the full per-layer fit (raw_mean oriented toward f1, val-CE, val-acc) so the caller can
    read the named margin at the neutral run's val-CE-selected layer (matched-layer shift) AND at
    the named condition's own val-CE-selected layer."""
    import numpy as np
    import torch

    import probe_fit

    start = time.monotonic()
    key = model_name
    if key not in _MODEL_CACHE:
        _MODEL_CACHE.clear()
        _MODEL_CACHE[key] = _load_model_and_tokenizer(model_name, "bfloat16", grad_ckpt=False)
        HF_CACHE_VOL.commit()
    model, tokenizer = _MODEL_CACHE[key]
    model.eval()
    pad_id = tokenizer.pad_token_id
    cells = list(rendered_groups.keys())

    def mean_pool(all_ids):
        chunks = []
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
            stacked = torch.stack(out.hidden_states, dim=1).float()  # [b, n_hs, seq, hidden]
            m = attn[:, None, :, None].float()
            picked = (stacked * m).sum(2) / m.sum(2).clamp(min=1)     # mean over real tokens
            chunks.append(picked.cpu().numpy())
            del out, stacked
        return np.concatenate(chunks, axis=0).astype(np.float32)

    acts = {}
    for cell in cells:
        ids = [_encode_ids(tokenizer, t) for t in rendered_groups[cell]]
        acts[cell] = mean_pool(ids)
    extract_elapsed = time.monotonic() - start

    n_hs = acts["A"].shape[1]
    g = probe_fit.fit_grid(
        acts, lengths, seeds_by_cell, np.asarray(signs),
        [probe_fit.Control("uncontrolled")], [probe_c], list(range(n_hs)), n_seeds=n_seeds,
    )
    layers = g["uncontrolled"][f"C={probe_c:g}"]["layers"]
    per_layer = {int(L): {"raw_mean": v["raw_mean"], "raw_std": v["raw_std"],
                          "val_ce_mean": v["val_ce_mean"], "val_acc_mean": v["val_acc_mean"],
                          "raw_seeds": v["raw_seeds"]}
                 for L, v in layers.items()}

    def _ce_key(L):  # NaN CE sorts last (scaffoldpool_full convention)
        ce = per_layer[L]["val_ce_mean"]
        return (1, 0.0) if np.isnan(ce) else (0, ce)

    sel = min(per_layer, key=_ce_key)
    return _jsonable({
        "pair": pair,
        "model": model_name,
        "meanpooled": {
            "selected_layer": sel,
            "n_hidden_states": n_hs,
            "raw_selected": per_layer[sel]["raw_mean"],
            "raw_selected_std": per_layer[sel]["raw_std"],
            "raw_selected_seeds": per_layer[sel]["raw_seeds"],
            "val_acc_selected": per_layer[sel]["val_acc_mean"],
            "val_ce_selected": per_layer[sel]["val_ce_mean"],
            "per_layer": per_layer,
        },
        "n_per_cell": {c: len(rendered_groups[c]) for c in cells},
        "probe_c": probe_c,
        "n_probe_seeds": n_seeds,
        "elapsed_extract": extract_elapsed,
        "elapsed": time.monotonic() - start,
    })
