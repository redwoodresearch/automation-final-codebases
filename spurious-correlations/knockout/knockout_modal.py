"""Modal app for the attention-knockout follow-up experiment (knockout/).

Additive to the study codebase (../codebase); no existing files are modified. The image pins are
byte-identical to lib/modal_app.py so tokenization + forward numerics match the study:
torch==2.7.1 (cu128), transformers==4.56.2, fp32 params with bf16-autocast matmuls, B200.

Remote entry points:
  * ``finetune_save_and_eval`` — Phase A: the study's own fine-tune recipe (same prompt template /
    label-token loss / lr / rng conventions as modal_app.finetune_and_eval with sample_weights=None),
    plus ONLINE detection of the training-loss-plateau checkpoint (finetune.loss_plateau_step
    convention: first eval step whose trailing-10 mean per-batch loss < 1e-3). The model is saved to
    the "spurious-knockout-ckpts" Volume at that step and training stops early (later steps cannot
    change the plateau step, and evals consume no rng, so the saved weights are exactly the study's
    pinned-checkpoint weights).
  * ``knockout_eval`` — Phase B: load a saved checkpoint (or the base model), run the held-out
    disagreement prompts, and read the frozen log-odds d = logit(X) - logit(Y) at the decision
    position while masking the DECISION ROW's attention at EVERY layer: it may only attend to
    {position 0} ∪ {last k positions}. All other rows untouched. Also a TARGETED knockout that
    blocks the decision row's attention to exactly the marker token span. Custom 4D additive masks
    (transformers 4.56 passes 4D masks through untouched); a built-in equivalence check asserts the
    unmasked custom-mask path reproduces the plain 2D-mask baseline.
  * ``head_attribution`` — Phase C: per (layer, head, source position) decision-position
    contribution c_{h,t} = attn_weight(decision -> t, head h) * <w, W_O^h v_t^h> with
    w = unembed(tok_a) - unembed(tok_b); aggregated per head plus the attention / contribution
    mass on the marker span.
"""

from __future__ import annotations

import os
import time

import modal

MODEL_NAME = "Qwen/Qwen3-1.7B"
GPU = "B200"
DTYPE = "float32"
HF_CACHE_VOL = modal.Volume.from_name("derisk-hf-cache", create_if_missing=True)
HF_CACHE_DIR = "/hf_cache"
CKPT_VOL = modal.Volume.from_name("spurious-knockout-ckpts", create_if_missing=True)
CKPT_DIR = "/ckpts"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.7.1", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "transformers==4.56.2",
        "accelerate==1.10.1",
        "scikit-learn==1.7.2",
        "numpy==2.2.6",
        "attrs==25.4.0",
    )
    .env({"HF_HOME": HF_CACHE_DIR, "HF_HUB_ENABLE_HF_TRANSFER": "0", "TOKENIZERS_PARALLELISM": "false"})
)

app = modal.App("spurious-knockout")

VOLUMES = {HF_CACHE_DIR: HF_CACHE_VOL, CKPT_DIR: CKPT_VOL}

# The study's plateau-checkpoint convention (lib/finetune.py).
TRAIN_LOSS_PLATEAU_THRESH = 1e-3
PLATEAU_WINDOW = 10


# ============================================================================= shared helpers (remote)
def _load_model_and_tokenizer(ckpt: str | None = None, dtype: str = DTYPE, attn: str = "eager"):
    """Base model (ckpt=None) or a fine-tuned checkpoint saved under CKPT_DIR/<ckpt>. The tokenizer
    is always the pinned base tokenizer (checkpoints don't change it)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    path = MODEL_NAME if ckpt is None else os.path.join(CKPT_DIR, ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        path, cache_dir=HF_CACHE_DIR, dtype=getattr(torch, dtype), attn_implementation=attn
    ).to("cuda")
    return model, tokenizer


def _set_determinism(seed: int):
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _format_prompt(template: str, text: str, a_char: str, b_char: str) -> str:
    return template.format(text=text, a=a_char, b=b_char)


def _label_ids(tokenizer, tok_a: str, tok_b: str) -> tuple[int, int]:
    a = tokenizer(tok_a, add_special_tokens=False)["input_ids"]
    b = tokenizer(tok_b, add_special_tokens=False)["input_ids"]
    assert len(a) == 1 and len(b) == 1, f"label tokens must be single Qwen3 tokens: {tok_a!r}->{a}, {tok_b!r}->{b}"
    return a[0], b[0]


def _eval_label_logits_pair(model, tokenizer, texts, template, a_char, b_char, a_id, b_id, batch_size=16):
    """Study convention (modal_app._eval_label_logits_pair): (logit_a, logit_b) at the label position
    (last token of text+prompt)."""
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


def _tokenize_prompts_with_marker_spans(tokenizer, texts, marker_spans, template, a_char, b_char):
    """Tokenize each rendered classify prompt (add_special_tokens=False — the study convention) and
    map each text's marker char span (in PROMPT coordinates; None if no marker) to the set of token
    indices whose char span overlaps it."""
    all_ids, marker_tok_idx = [], []
    for text, span in zip(texts, marker_spans):
        prompt = _format_prompt(template, text, a_char, b_char)
        enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        all_ids.append(enc["input_ids"])
        if span is None:
            marker_tok_idx.append(None)
        else:
            s, e = span
            idx = [i for i, (o0, o1) in enumerate(enc["offset_mapping"]) if o1 > s and o0 < e and o1 > o0]
            assert idx, f"marker span {span} matched no tokens"
            marker_tok_idx.append(idx)
    return all_ids, marker_tok_idx


# ============================================================================= CHAT-format fine-tune
def _chat_prompt_ids(tokenizer, prompt: str) -> list[int]:
    """Token ids of the classify prompt placed as a USER turn (apply_chat_template,
    add_generation_prompt=True). The Qwen3 template ends '<|im_start|>assistant\\n' with no <think>
    scaffold, so the next-token position is exactly the assistant's X/Y answer."""
    wrapped = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    return tokenizer(wrapped, add_special_tokens=False)["input_ids"]


def _eval_label_logits_pair_chat(model, tokenizer, texts, template, a_char, b_char, a_id, b_id, batch_size=16):
    """Chat variant of _eval_label_logits_pair: (logit_a, logit_b) read at the assistant's first-token
    position (last token of the chat-wrapped user turn)."""
    import numpy as np
    import torch

    all_ids = [_chat_prompt_ids(tokenizer, _format_prompt(template, t, a_char, b_char)) for t in texts]
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


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=60 * 60)
def finetune_and_eval_chat(
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
) -> dict:
    """The study's fine-tune recipe, but the classify prompt is delivered in the model's CHAT format:
    each passage+question is a user turn and the assistant answers with the single label token
    (' X'/' Y'), loss on that answer token only; the held-out delta is read at the assistant's
    first-token position. Training math / seeding / batch order match finetune_save_and_eval; only the
    prompt rendering (raw -> chat) changes. No checkpoint is saved (this is a read-only robustness
    check). heldout_raw is the study's Delta in nats toward f1 at each eval step."""
    import numpy as np
    import torch

    start = time.monotonic()
    _set_determinism(seed)
    model, tokenizer = _load_model_and_tokenizer(attn="sdpa")
    HF_CACHE_VOL.commit()
    a_char, b_char = tok_a.strip(), tok_b.strip()
    a_id, b_id = _label_ids(tokenizer, tok_a, tok_b)
    label_token_ids = [a_id, b_id]
    if eval_steps is None:
        eval_steps = [0, 25, 50, 75, 100, 150, 200]
    eval_steps = [s for s in eval_steps if s <= max_steps]

    pad_id = tokenizer.pad_token_id
    seqs, lbls = [], []
    for text, label in zip(train_texts, train_labels):
        pid = _chat_prompt_ids(tokenizer, _format_prompt(template, text, a_char, b_char))
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
        tr_pair = _eval_label_logits_pair_chat(model, tokenizer, train_texts, template, a_char, b_char, a_id, b_id)
        acc, logp = _training_signal(tr_pair, train_labels)
        held_pair = _eval_label_logits_pair_chat(model, tokenizer, heldout_texts, template, a_char, b_char, a_id, b_id)
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

    return {"curve": curve, "train_losses": train_losses, "wrap": "chat", "seed": seed,
            "n_train": len(train_texts), "n_heldout": len(heldout_texts),
            "elapsed": time.monotonic() - start}


# ============================================================================= raw fine-tune (no save)
@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=60 * 60)
def finetune_and_eval_raw(
    train_texts: list[str],
    train_labels: list[int],
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
) -> dict:
    """The study's raw-template fine-tune (finetune_save_and_eval math, verbatim) but WITHOUT saving a
    checkpoint — for cheap read-only sweeps that only need the held-out delta (e.g. alternate label
    tokens: pass tok_a/tok_b = ' A'/' B', ' yes'/' no', ' good'/' bad'). heldout_raw is the study's
    Delta in nats toward f1 at each eval step."""
    import numpy as np
    import torch

    start = time.monotonic()
    _set_determinism(seed)
    model, tokenizer = _load_model_and_tokenizer(attn="sdpa")
    HF_CACHE_VOL.commit()
    a_char, b_char = tok_a.strip(), tok_b.strip()
    a_id, b_id = _label_ids(tokenizer, tok_a, tok_b)
    label_token_ids = [a_id, b_id]
    if eval_steps is None:
        eval_steps = [0, 25, 50, 75, 100, 150, 200]
    eval_steps = [s for s in eval_steps if s <= max_steps]

    pad_id = tokenizer.pad_token_id
    seqs, lbls = [], []
    for text, label in zip(train_texts, train_labels):
        pid = tokenizer(_format_prompt(template, text, a_char, b_char), add_special_tokens=False)["input_ids"]
        label_id = label_token_ids[label]
        seqs.append(pid + [label_id])
        lbls.append([-100] * len(pid) + [label_id])

    def make_batch(indices):
        batch_ids = [seqs[i] for i in indices]; batch_lbl = [lbls[i] for i in indices]
        maxlen = max(len(x) for x in batch_ids)
        input_ids = torch.full((len(indices), maxlen), pad_id, dtype=torch.long)
        labels = torch.full((len(indices), maxlen), -100, dtype=torch.long)
        attn = torch.zeros((len(indices), maxlen), dtype=torch.long)
        for i, (ids, lb) in enumerate(zip(batch_ids, batch_lbl)):
            input_ids[i, : len(ids)] = torch.tensor(ids); labels[i, : len(lb)] = torch.tensor(lb); attn[i, : len(ids)] = 1
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
        tr_pair = _eval_label_logits_pair(model, tokenizer, train_texts, template, a_char, b_char, a_id, b_id)
        acc, logp = _training_signal(tr_pair, train_labels)
        held_pair = _eval_label_logits_pair(model, tokenizer, heldout_texts, template, a_char, b_char, a_id, b_id)
        d = (held_pair[:, 0] - held_pair[:, 1])
        oriented = d * np.asarray(heldout_f1_signs, dtype=np.float64)
        model.train()
        return {"step": step, "train_acc": acc, "train_correct_logp": logp,
                "heldout_raw": float(oriented.mean()),
                "heldout_bounded": float((1.0 / (1.0 + np.exp(-oriented))).mean()),
                "heldout_acc": float((oriented > 0).mean())}

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
        loss.backward(); opt.step()
        train_losses.append(float(loss.item()))
        if step in eval_steps:
            curve.append(evaluate(step))

    return {"curve": curve, "train_losses": train_losses, "tok_a": tok_a, "tok_b": tok_b, "seed": seed,
            "n_train": len(train_texts), "n_heldout": len(heldout_texts), "elapsed": time.monotonic() - start}


# ============================================================================= reward-model scalar head
def _scalar_token_ids(tokenizer, texts):
    """Raw-text token ids (wrap='none', add_special_tokens=False) — the probe's representation; what a
    reward model scores."""
    return [tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]


def _scalar_reads(model, v_head, tokenizer, token_id_lists, batch_size=16):
    """r(text) = v_head(final-layer last-non-pad-token hidden state), no_grad, float32 numpy [n]."""
    import numpy as np
    import torch

    pad_id = tokenizer.pad_token_id
    out = []
    for start in range(0, len(token_id_lists), batch_size):
        batch = token_id_lists[start:start + batch_size]
        maxlen = max(len(x) for x in batch)
        input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for i, ids in enumerate(batch):
            input_ids[i, : len(ids)] = torch.tensor(ids); attn[i, : len(ids)] = 1
        input_ids, attn = input_ids.to("cuda"), attn.to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hs = model.model(input_ids=input_ids, attention_mask=attn).last_hidden_state
            last_idx = attn.sum(1) - 1
            rows = torch.arange(hs.shape[0], device="cuda")
            r = v_head(hs[rows, last_idx, :].float()).squeeze(-1)
        out.append(r.float().cpu().numpy())
    return np.concatenate(out)


def _scalar_diagonal_signal(r_A, r_B):
    """Training-cell separation for checkpoint gating (NEVER the held cells). AUC = P(r_A>r_B); margin."""
    import numpy as np
    rA = np.asarray(r_A)[:, None]; rB = np.asarray(r_B)[None, :]
    return float((rA > rB).mean() + 0.5 * (rA == rB).mean()), float(rA.mean() - rB.mean())


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=60 * 60)
def finetune_scalar_and_eval(
    train_texts: list[str],
    train_labels: list[int],          # 0 = class A (scores higher), 1 = class B
    heldout_texts: list[str],
    heldout_f1_signs: list[int],
    lr_backbone: float,
    lr_head: float,
    max_steps: int,
    seed: int,
    warmup_steps: int = 50,
    batch_size: int = 16,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    eval_steps: list[int] | None = None,
) -> dict:
    """Reward-model-style readout: full fine-tune of the backbone + a fresh zero-init linear VALUE HEAD,
    Bradley-Terry loss  -log σ(r(A)-r(B))  over random 1:1 (A,B) diagonal pairs. The head reads the raw
    text's final-layer last-token hidden state (what a reward model scores). LP-FT: head-only warmup,
    then unfreeze the backbone. heldout_raw = mean(r·sign) toward f1 (same orientation as the study)."""
    import numpy as np
    import torch

    start = time.monotonic()
    _set_determinism(seed)
    model, tokenizer = _load_model_and_tokenizer(attn="sdpa")
    HF_CACHE_VOL.commit()
    hidden = model.config.hidden_size
    v_head = torch.nn.Linear(hidden, 1, bias=False).to("cuda").float()
    if eval_steps is None:
        eval_steps = [0, 50, 75, 100, 150, 200]
    eval_steps = [s for s in eval_steps if s <= max_steps]

    a_ids = [i for i, l in enumerate(train_labels) if l == 0]
    b_ids = [i for i, l in enumerate(train_labels) if l == 1]
    tok = _scalar_token_ids(tokenizer, train_texts)
    held_tok = _scalar_token_ids(tokenizer, heldout_texts)
    train_A_tok = [tok[i] for i in a_ids]; train_B_tok = [tok[i] for i in b_ids]
    pad_id = tokenizer.pad_token_id

    def set_backbone_trainable(flag):
        for p in model.parameters():
            p.requires_grad = flag

    set_backbone_trainable(False)
    opt = torch.optim.AdamW(
        [{"params": list(model.parameters()), "lr": lr_backbone, "weight_decay": weight_decay},
         {"params": list(v_head.parameters()), "lr": lr_head, "weight_decay": weight_decay}])
    rng = np.random.default_rng(seed)
    order_A: list[int] = []; order_B: list[int] = []

    def next_pair_indices(n):
        nonlocal order_A, order_B
        oa, ob = [], []
        while len(oa) < n:
            if not order_A:
                order_A = list(rng.permutation(len(train_A_tok)))
            oa.append(order_A.pop())
        while len(ob) < n:
            if not order_B:
                order_B = list(rng.permutation(len(train_B_tok)))
            ob.append(order_B.pop())
        return oa, ob

    def forward_r(token_lists):
        maxlen = max(len(x) for x in token_lists)
        input_ids = torch.full((len(token_lists), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(token_lists), maxlen), dtype=torch.long)
        for i, ids in enumerate(token_lists):
            input_ids[i, : len(ids)] = torch.tensor(ids); attn[i, : len(ids)] = 1
        input_ids, attn = input_ids.to("cuda"), attn.to("cuda")
        hs = model.model(input_ids=input_ids, attention_mask=attn).last_hidden_state
        last_idx = attn.sum(1) - 1; rows = torch.arange(hs.shape[0], device="cuda")
        return v_head(hs[rows, last_idx, :].float()).squeeze(-1)

    def evaluate(step):
        model.eval()
        r_A = _scalar_reads(model, v_head, tokenizer, train_A_tok)
        r_B = _scalar_reads(model, v_head, tokenizer, train_B_tok)
        auc, margin = _scalar_diagonal_signal(r_A, r_B)
        r_held = _scalar_reads(model, v_head, tokenizer, held_tok)
        oriented = np.asarray(r_held) * np.asarray(heldout_f1_signs, dtype=np.float64)
        model.train()
        return {"step": step, "diag_auc": auc, "diag_margin": margin,
                "heldout_raw": float(oriented.mean()),
                "heldout_bounded": float((1.0 / (1.0 + np.exp(-oriented))).mean()),
                "heldout_acc": float((oriented > 0).mean())}

    n_pair = max(1, batch_size // 2)
    curve, train_losses = [], []
    model.train()
    if 0 in eval_steps:
        curve.append(evaluate(0))
    for step in range(1, max_steps + 1):
        if step == warmup_steps + 1:
            set_backbone_trainable(True)
        oa, ob = next_pair_indices(n_pair)
        batch_tokens = [train_A_tok[i] for i in oa] + [train_B_tok[i] for i in ob]
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            r = forward_r(batch_tokens)
            r_A, r_B = r[:n_pair], r[n_pair:]
            loss = torch.nn.functional.softplus(-(r_A - r_B).float()).mean()  # Bradley-Terry
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad] + list(v_head.parameters()), grad_clip)
        opt.step()
        train_losses.append(float(loss.item()))
        if step in eval_steps:
            curve.append(evaluate(step))

    return {"curve": curve, "train_losses": train_losses, "lr_backbone": lr_backbone, "lr_head": lr_head,
            "warmup_steps": warmup_steps, "seed": seed, "n_train": len(train_texts),
            "n_heldout": len(heldout_texts), "elapsed": time.monotonic() - start}


# ============================================================================= Phase A: fine-tune + save
@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=60 * 60)
def finetune_save_and_eval(
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
    ckpt_name: str,
    batch_size: int = 16,
    eval_steps: list[int] | None = None,
) -> dict:
    """The study's fine-tune recipe (modal_app.finetune_and_eval, sample_weights=None path) with the
    plateau checkpoint saved to the ckpt Volume and early stopping at that step. Training math,
    seeding, batch order and eval code are copied verbatim so the plateau-step weights are exactly
    what the study's pipeline would have produced."""
    import numpy as np
    import torch

    start = time.monotonic()
    _set_determinism(seed)
    model, tokenizer = _load_model_and_tokenizer(attn="sdpa")
    HF_CACHE_VOL.commit()
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
        loss.backward()
        opt.step()
        train_losses.append(float(loss.item()))
        if step in eval_steps:
            curve.append(evaluate(step))
            # finetune.loss_plateau_step convention: first eval step whose trailing-window smoothed
            # per-batch training loss is below threshold. Save + stop there (training-signal only).
            tl = np.asarray(train_losses)
            if tl[max(0, step - PLATEAU_WINDOW): step].mean() < TRAIN_LOSS_PLATEAU_THRESH:
                plateau_step = step
                break

    saved_step = plateau_step if plateau_step is not None else max_steps
    save_path = os.path.join(CKPT_DIR, ckpt_name)
    model.save_pretrained(save_path, safe_serialization=True)  # fp32, exactly the trained params
    CKPT_VOL.commit()

    return {
        "curve": curve,
        "train_losses": train_losses,
        "plateau_step": plateau_step,
        "saved_step": saved_step,
        "ckpt_name": ckpt_name,
        "lr": lr,
        "max_steps": max_steps,
        "seed": seed,
        "batch_size": batch_size,
        "n_train": len(train_texts),
        "n_heldout": len(heldout_texts),
        "elapsed": time.monotonic() - start,
    }


# ============================================================================= Phase B: attention knockout
@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=60 * 60)
def knockout_eval(
    ckpt: str | None,
    texts: list[str],
    marker_spans: list[tuple[int, int] | None],  # char span of the marker in the RENDERED PROMPT (or None)
    template: str,
    tok_a: str,
    tok_b: str,
    k_list: list[int],
    batch_size: int = 8,
) -> dict:
    """Frozen-checkpoint inference with the decision row's attention masked at EVERY layer.

    Conditions:
      * "baseline2d" — plain 2D attention mask (the study's exact readout path; sanity reference).
      * "full"       — custom 4D causal mask, no knockout (must equal baseline2d; asserted).
      * "k<k>"       — decision position may only attend to {position 0} ∪ {last k real positions}
                       (the last k INCLUDE the decision position itself). Position 0 stands in for
                       BOS: Qwen3's 'none' wrapping adds no special token, so index 0 (the
                       attention-sink position) is kept attendable in every condition.
      * "targeted"   — decision position blocked from exactly the marker token span (texts with
                       marker_spans=None are left unmasked).
    Returns per-condition per-text d = logit(tok_a) - logit(tok_b) at the decision position.
    """
    import numpy as np
    import torch

    start = time.monotonic()
    model, tokenizer = _load_model_and_tokenizer(ckpt, attn="eager")
    model.eval()
    HF_CACHE_VOL.commit()
    a_char, b_char = tok_a.strip(), tok_b.strip()
    a_id, b_id = _label_ids(tokenizer, tok_a, tok_b)
    all_ids, marker_tok_idx = _tokenize_prompts_with_marker_spans(
        tokenizer, texts, marker_spans, template, a_char, b_char
    )
    lens = [len(x) for x in all_ids]
    NEG = torch.finfo(torch.float32).min

    def run_condition(condition) -> list[float]:
        """condition: 'baseline2d' | 'full' | ('k', k) | 'targeted'."""
        d_out = []
        for s in range(0, len(all_ids), batch_size):
            batch = all_ids[s : s + batch_size]
            blens = lens[s : s + batch_size]
            maxlen = max(blens)
            input_ids = torch.full((len(batch), maxlen), tokenizer.pad_token_id, dtype=torch.long)
            attn2d = torch.zeros((len(batch), maxlen), dtype=torch.long)
            for i, ids in enumerate(batch):
                input_ids[i, : len(ids)] = torch.tensor(ids)
                attn2d[i, : len(ids)] = 1
            input_ids = input_ids.to("cuda")
            if condition == "baseline2d":
                mask = attn2d.to("cuda")
            else:
                # additive float causal mask [b, 1, L, L]; pad keys masked; then knock out the
                # decision row (query index len-1) as the condition requires.
                m = torch.zeros((len(batch), 1, maxlen, maxlen), dtype=torch.float32)
                causal = torch.triu(torch.ones(maxlen, maxlen, dtype=torch.bool), diagonal=1)
                m[:, 0, causal] = NEG
                for i, L in enumerate(blens):
                    m[i, 0, :, L:] = NEG  # pad keys
                    dec = L - 1
                    if condition == "full":
                        pass
                    elif condition == "targeted":
                        if marker_tok_idx[s + i] is not None:
                            m[i, 0, dec, torch.tensor(marker_tok_idx[s + i])] = NEG
                    else:
                        _, k = condition
                        allowed = torch.zeros(maxlen, dtype=torch.bool)
                        allowed[0] = True
                        allowed[max(0, L - k): L] = True
                        row = torch.full((maxlen,), NEG, dtype=torch.float32)
                        row[allowed] = 0.0
                        m[i, 0, dec, :] = row
                mask = m.to("cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids, attention_mask=mask).logits
            for i, L in enumerate(blens):
                row = logits[i, L - 1, :]
                d_out.append(float(row[a_id].float() - row[b_id].float()))
        return d_out

    conditions: dict[str, list[float]] = {}
    conditions["baseline2d"] = run_condition("baseline2d")
    conditions["full"] = run_condition("full")
    # equivalence check: the custom 4D mask path must reproduce the plain 2D path
    diff = float(np.max(np.abs(np.array(conditions["full"]) - np.array(conditions["baseline2d"]))))
    assert diff < 0.15, f"custom 4D mask does not reproduce the 2D baseline (max |Δd| = {diff})"
    for k in k_list:
        conditions[f"k{k}"] = run_condition(("k", k))
    conditions["targeted"] = run_condition("targeted")

    return {
        "ckpt": ckpt,
        "conditions": conditions,
        "mask_equiv_max_dd": diff,
        "seq_lens": lens,
        "n_marker_tokens": [len(x) if x is not None else 0 for x in marker_tok_idx],
        # distance (in tokens) from the decision position to the NEAREST marker token — tells us
        # which k windows would reach the marker.
        "marker_dist_from_decision": [
            (lens[i] - 1 - max(x)) if x is not None else None for i, x in enumerate(marker_tok_idx)
        ],
        "elapsed": time.monotonic() - start,
    }


# ============================================================================= Phase C: head attribution
@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=60 * 60)
def head_attribution(
    ckpt: str | None,
    texts: list[str],
    marker_spans: list[tuple[int, int] | None],
    template: str,
    tok_a: str,
    tok_b: str,
    batch_size: int = 4,
) -> dict:
    """Decision-position head attribution: c_{h,t} = attn(decision -> t, head h) * <w, W_O^h v_t^h>,
    w = unembed(tok_a) - unembed(tok_b). GQA: query head h reads kv head h // (H // H_kv).

    Returns per-(layer, head) aggregates (means over texts):
      * abs_contrib      — sum_t |c_{h,t}|   (the ranking statistic)
      * signed_contrib   — sum_t c_{h,t}     (net push along +w = toward tok_a)
      * attn_marker_frac — attention mass on the marker span (marker texts only)
      * contrib_marker_frac — |c| mass on the marker span (marker texts only)
    """
    import numpy as np
    import torch

    start = time.monotonic()
    model, tokenizer = _load_model_and_tokenizer(ckpt, attn="eager")
    model.eval()
    HF_CACHE_VOL.commit()
    a_char, b_char = tok_a.strip(), tok_b.strip()
    a_id, b_id = _label_ids(tokenizer, tok_a, tok_b)
    all_ids, marker_tok_idx = _tokenize_prompts_with_marker_spans(
        tokenizer, texts, marker_spans, template, a_char, b_char
    )

    cfg = model.config
    n_layers, n_heads, n_kv = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)
    group = n_heads // n_kv

    with torch.no_grad():
        w = (model.lm_head.weight[a_id] - model.lm_head.weight[b_id]).float()  # [hidden]
        # u[l, h] = (W_O^h)^T w : o_proj.weight is [hidden, n_heads*head_dim]
        u = torch.stack(
            [
                (model.model.layers[l].self_attn.o_proj.weight.float().T @ w).view(n_heads, head_dim)
                for l in range(n_layers)
            ]
        )  # [n_layers, n_heads, head_dim]

    v_cache: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx):
        def hook(module, inp, out):
            v_cache[layer_idx] = out.detach()  # [b, L, n_kv*head_dim]
        return hook

    handles = [model.model.layers[l].self_attn.v_proj.register_forward_hook(make_hook(l)) for l in range(n_layers)]

    abs_contrib = np.zeros((n_layers, n_heads))
    signed_contrib = np.zeros((n_layers, n_heads))
    attn_marker = np.zeros((n_layers, n_heads))
    contrib_marker = np.zeros((n_layers, n_heads))
    n_texts = 0
    n_marker_texts = 0

    pad_id = tokenizer.pad_token_id
    for s in range(0, len(all_ids), batch_size):
        batch = all_ids[s : s + batch_size]
        blens = [len(x) for x in batch]
        maxlen = max(blens)
        input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
        attn2d = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for i, ids in enumerate(batch):
            input_ids[i, : len(ids)] = torch.tensor(ids)
            attn2d[i, : len(ids)] = 1
        input_ids, attn2d = input_ids.to("cuda"), attn2d.to("cuda")
        v_cache.clear()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, attention_mask=attn2d, output_attentions=True)
        for i, L in enumerate(blens):
            dec = L - 1
            midx = marker_tok_idx[s + i]
            n_texts += 1
            if midx is not None:
                n_marker_texts += 1
            for l in range(n_layers):
                a = out.attentions[l][i, :, dec, :L].float()          # [n_heads, L]
                v = v_cache[l][i, :L].float().view(L, n_kv, head_dim)  # [L, n_kv, head_dim]
                # dot(v_t^{kv(h)}, u_{l,h}) for every q head h. HF repeat_kv layout: q head
                # h = k*group + g reads kv head k, and u[l].view(n_kv, group, head_dim)[k, g]
                # is exactly u[l, k*group + g].
                u_l = u[l].view(n_kv, group, head_dim).to(v.device)
                vu = torch.einsum("tkd,kgd->kgt", v, u_l).reshape(n_heads, L)
                c = a * vu  # [n_heads, L]
                ca = c.abs()
                abs_contrib[l] += ca.sum(1).cpu().numpy()
                signed_contrib[l] += c.sum(1).cpu().numpy()
                if midx is not None:
                    mt = torch.tensor(midx, device=a.device)
                    attn_marker[l] += a[:, mt].sum(1).cpu().numpy()
                    contrib_marker[l] += (ca[:, mt].sum(1) / ca.sum(1).clamp(min=1e-12)).cpu().numpy()

    for h in handles:
        h.remove()

    return {
        "ckpt": ckpt,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "abs_contrib": (abs_contrib / max(n_texts, 1)).tolist(),
        "signed_contrib": (signed_contrib / max(n_texts, 1)).tolist(),
        "attn_marker_frac": (attn_marker / max(n_marker_texts, 1)).tolist(),
        "contrib_marker_frac": (contrib_marker / max(n_marker_texts, 1)).tolist(),
        "n_texts": n_texts,
        "n_marker_texts": n_marker_texts,
        "elapsed": time.monotonic() - start,
    }
