"""Modal entry points added by the human-side experiments: LM-loss SFT + generation (behavioral
arm) and long-template ICL scoring.

SELF-CONTAINED on purpose: the container re-imports the module that defines a function, and
modal_app.py is not baked into the image — importing it here at module level crashes remotely
(ModuleNotFoundError). So this module carries its own app/image (same spec; metric.py baked the
same way) and its own copies of the small remote helpers. modal_app.py and its caches are
untouched. Runners open THIS module's app via AsyncModalSession(app_module=modal_gen).
"""
from __future__ import annotations

import time

import modal

SFT_GEN_CALL_UUID = "sft-gen-v2"      # v2: self-contained module (remote import fix)
ICL_SCORE_CALL_UUID = "icl-score-v2"
DPO_BEHAVE_CALL_UUID = "dpo-behave-v1"  # preference-tuned (DPO) behavioral arm — matched to sft_lm_and_generate

MODEL_NAME = "Qwen/Qwen3-1.7B"
GPU = "B200"  # match modal_app: faster than A100, 1.7B fits easily
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
    .add_local_file("metric.py", "/root/metric.py", copy=True)
)

app = modal.App("spurious-correlations-gen")


# --------------------------------------------------------------------- remote helpers (copies)
def _load_model_and_tokenizer(attn: str = "sdpa"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_CACHE_DIR)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, cache_dir=HF_CACHE_DIR, dtype=torch.float32, attn_implementation=attn
    ).to("cuda")
    return model, tokenizer


def _set_determinism(seed: int):
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _label_ids(tokenizer, tok_a: str, tok_b: str) -> tuple[int, int]:
    a = tokenizer(tok_a, add_special_tokens=False)["input_ids"]
    b = tokenizer(tok_b, add_special_tokens=False)["input_ids"]
    assert len(a) == 1 and len(b) == 1, f"label tokens must be single Qwen3 tokens: {tok_a!r}->{a}, {tok_b!r}->{b}"
    return a[0], b[0]


def _eval_label_logits_pair(model, tokenizer, texts, template, a_char, b_char, a_id, b_id, batch_size=8):
    import numpy as np
    import torch

    prompts = [template.format(text=t, a=a_char, b=b_char) for t in texts]
    out = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start: start + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**enc).logits
        last = enc["attention_mask"].sum(1) - 1
        rows = torch.arange(len(chunk))
        pair = torch.stack([logits[rows, last, a_id], logits[rows, last, b_id]], dim=1)
        out.append(pair.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def _heldout_summary(held_pair, heldout_f1_signs) -> dict:
    import sys

    sys.path.insert(0, "/root")
    import metric

    d = held_pair[:, 0] - held_pair[:, 1]
    res = metric.logprob_difference(d, heldout_f1_signs)
    return {"raw": res.raw, "bounded_prob": res.bounded_prob, "hard_accuracy": res.hard_accuracy,
            "n": res.n, "d_values": [float(x) for x in d]}


# --------------------------------------------------------------------- ICL scoring (sdpa, small batch)
@app.function(image=image, gpu=GPU, volumes={HF_CACHE_DIR: HF_CACHE_VOL}, timeout=60 * 30)
def icl_logprob_diff(
    heldout_texts: list[str],
    heldout_f1_signs: list[int],
    template: str,
    tok_a: str,
    tok_b: str,
    batch_size: int = 8,
) -> dict:
    """Label-logit scoring for LONG (k-shot) templates: sdpa attention (eager materializes the full
    attention matrix and OOMs at ~4k-token prompts) and a small eval batch. Same metric and
    label-position read as the baseline/fine-tune scorers."""
    start = time.monotonic()
    model, tokenizer = _load_model_and_tokenizer(attn="sdpa")
    model.eval()
    HF_CACHE_VOL.commit()
    a_char, b_char = tok_a.strip(), tok_b.strip()
    a_id, b_id = _label_ids(tokenizer, tok_a, tok_b)
    held_pair = _eval_label_logits_pair(model, tokenizer, heldout_texts, template, a_char, b_char,
                                        a_id, b_id, batch_size=batch_size)
    return {"heldout": _heldout_summary(held_pair, heldout_f1_signs),
            "elapsed": time.monotonic() - start}


# --------------------------------------------------------------------- behavioral SFT + generation
@app.function(image=image, gpu=GPU, volumes={HF_CACHE_DIR: HF_CACHE_VOL}, timeout=60 * 60)
def sft_lm_and_generate(
    train_prompts: list[str],
    train_responses: list[str],
    heldout_prompts: list[str],
    lr: float,
    max_steps: int,
    seed: int,
    batch_size: int = 8,
    gen_temperature: float = 0.7,
    gen_max_new_tokens: int = 130,
    plateau_rel_improve: float = 0.02,
    plateau_window: int = 25,
    min_steps: int = 50,
) -> dict:
    """Full fine-tune: input = prompt + response, LM loss on response tokens only. Early-stops on a
    relative training-loss plateau (training-signal only). Then samples one response per held-out
    prompt at fixed temperature/seed."""
    import numpy as np
    import torch

    start = time.monotonic()
    _set_determinism(seed)
    model, tokenizer = _load_model_and_tokenizer(attn="sdpa")
    HF_CACHE_VOL.commit()
    pad_id = tokenizer.pad_token_id

    seqs, lbls = [], []
    for prompt, resp in zip(train_prompts, train_responses):
        pid = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        rid = tokenizer(resp, add_special_tokens=False)["input_ids"]
        seqs.append(pid + rid)
        lbls.append([-100] * len(pid) + rid)

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

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
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

    model.train()
    train_losses: list[float] = []
    stop_step = max_steps
    for step in range(1, max_steps + 1):
        input_ids, attn, labels = make_batch(next_indices(batch_size))
        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss
        loss.backward()
        opt.step()
        train_losses.append(float(loss.item()))
        if step >= max(min_steps, 2 * plateau_window) and step % plateau_window == 0:
            prev = float(np.mean(train_losses[-2 * plateau_window: -plateau_window]))
            cur = float(np.mean(train_losses[-plateau_window:]))
            if prev - cur < plateau_rel_improve * prev:
                stop_step = step
                break

    model.eval()
    torch.manual_seed(seed)
    tokenizer.padding_side = "left"
    generated: list[str] = []
    bs = 16
    for i in range(0, len(heldout_prompts), bs):
        chunk = heldout_prompts[i: i + bs]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(
                **enc, do_sample=True, temperature=gen_temperature, top_p=0.95,
                max_new_tokens=gen_max_new_tokens, pad_token_id=pad_id,
            )
        for row in out:
            new_tokens = row[enc["input_ids"].shape[1]:]
            generated.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

    return {
        "train_losses": train_losses,
        "stop_step": stop_step,
        "generated": generated,
        "n_train": len(train_prompts),
        "n_heldout": len(heldout_prompts),
        "elapsed": time.monotonic() - start,
    }


# --------------------------------------------------------------------- DPO (preference-tuned) behavior arm
# Standard DPO with a frozen reference = the BASE model. The reference sequence-logprobs are precomputed
# ONCE before training (the reference never changes), so only one model is held in memory. Sequence-logprob
# = sum of per-token log-softmax over the RESPONSE tokens (the prompt is masked). Ported from the working
# salience_flip copy so the main pipeline can run the SFT-vs-DPO comparison (blog "DPO" appendix claim);
# generation is IDENTICAL to sft_lm_and_generate (same temp/top_p/max_new_tokens/seed) so the two arms
# differ ONLY in the training objective.
#   L = -log σ( β·[ (logπ_θ(y_c|x) - logπ_θ(y_r|x)) - (logπ_ref(y_c|x) - logπ_ref(y_r|x)) ] ) + nll_weight·NLL(y_c)
# nll_weight>0 (the RPO / Llama-3 chosen-likelihood anchor) is what makes the model actually GENERATE the
# chosen pole from base; pure ranking loss (nll_weight=0) drives the preference without inducing generation.
def _dpo_pretokenize(tokenizer, prompts, responses, append_eos):
    eos = [tokenizer.eos_token_id] if append_eos else []
    seqs, plens = [], []
    for p, r in zip(prompts, responses):
        pid = tokenizer(p, add_special_tokens=False)["input_ids"]
        rid = tokenizer(r, add_special_tokens=False)["input_ids"] + eos
        seqs.append(pid + rid)
        plens.append(len(pid))
    return seqs, plens


def _seq_logprob_batch(model, seqs, plens, indices, pad_id):
    """Summed log P(response tokens) for the given example indices, batched. Grad flows iff not under
    no_grad (used both for the no-grad reference precompute and the grad training step)."""
    import torch

    batch = [seqs[i] for i in indices]
    bplens = [plens[i] for i in indices]
    maxlen = max(len(x) for x in batch)
    input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
    for i, ids in enumerate(batch):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        attn[i, : len(ids)] = 1
    input_ids, attn = input_ids.to("cuda"), attn.to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=input_ids, attention_mask=attn).logits
    logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    tgt = input_ids[:, 1:]
    tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    out = []
    for i in range(len(batch)):
        plen, L = bplens[i], len(batch[i])
        out.append(tok_logp[i, plen - 1: L - 1].sum())  # response tokens only (prompt masked)
    return torch.stack(out)


def _precompute_ref_logprobs(model, seqs, plens, pad_id, batch_size=8):
    import torch

    model.eval()
    refs = []
    with torch.no_grad():
        for start in range(0, len(seqs), batch_size):
            idx = list(range(start, min(start + batch_size, len(seqs))))
            refs.append(_seq_logprob_batch(model, seqs, plens, idx, pad_id))
    return torch.cat(refs).detach()


def _dpo_train_loop(model, tokenizer, prompts, chosen, rejected, lr, beta, max_steps, seed,
                    batch_size, plateau_rel_improve, plateau_window, min_steps, append_eos,
                    grad_clip=1.0, nll_weight=0.0):
    import numpy as np
    import torch

    pad_id = tokenizer.pad_token_id
    c_seqs, c_plens = _dpo_pretokenize(tokenizer, prompts, chosen, append_eos)
    r_seqs, r_plens = _dpo_pretokenize(tokenizer, prompts, rejected, append_eos)
    ref_c = _precompute_ref_logprobs(model, c_seqs, c_plens, pad_id)
    ref_r = _precompute_ref_logprobs(model, r_seqs, r_plens, pad_id)
    c_counts = torch.tensor([max(1, len(s) - p) for s, p in zip(c_seqs, c_plens)],
                            dtype=torch.float32, device="cuda")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    n = len(prompts)
    order: list[int] = []

    def next_idx(k):
        nonlocal order
        out = []
        while len(out) < k:
            if not order:
                order = list(rng.permutation(n))
            out.append(order.pop())
        return out

    model.train()
    losses, pref_accs, margins = [], [], []
    stop_step = max_steps
    for step in range(1, max_steps + 1):
        idx = next_idx(batch_size)
        opt.zero_grad()
        pi_c = _seq_logprob_batch(model, c_seqs, c_plens, idx, pad_id)
        pi_r = _seq_logprob_batch(model, r_seqs, r_plens, idx, pad_id)
        logits = beta * ((pi_c - pi_r) - (ref_c[idx] - ref_r[idx]))
        loss = -torch.nn.functional.logsigmoid(logits).mean()
        if nll_weight:
            nll_chosen = -(pi_c / c_counts[idx]).mean()  # per-token NLL of the chosen response
            loss = loss + nll_weight * nll_chosen
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        losses.append(float(loss.item()))
        pref_accs.append(float((logits.detach() > 0).float().mean().item()))
        margins.append(float(logits.detach().mean().item()))
        if step >= max(min_steps, 2 * plateau_window) and step % plateau_window == 0:
            prev = float(np.mean(losses[-2 * plateau_window: -plateau_window]))
            cur = float(np.mean(losses[-plateau_window:]))
            if prev - cur < plateau_rel_improve * prev:
                stop_step = step
                break
    return {"train_losses": losses, "pref_accs": pref_accs, "reward_margins": margins, "stop_step": stop_step}


@app.function(image=image, gpu=GPU, volumes={HF_CACHE_DIR: HF_CACHE_VOL}, timeout=60 * 60)
def dpo_behave_and_generate(
    train_prompts: list[str],
    train_chosen: list[str],
    train_rejected: list[str],
    heldout_prompts: list[str],
    lr: float,
    beta: float,
    max_steps: int,
    seed: int,
    batch_size: int = 4,
    gen_temperature: float = 0.7,
    gen_max_new_tokens: int = 130,
    plateau_rel_improve: float = 0.02,
    plateau_window: int = 25,
    min_steps: int = 50,
    grad_clip: float = 1.0,
    nll_weight: float = 1.0,
    append_eos: bool = True,
) -> dict:
    """Preference-tune (DPO) the subject to BEHAVE the trained pole, then sample one response per held-out
    prompt with the SAME generation settings as sft_lm_and_generate (do_sample, temperature, top_p=0.95,
    max_new_tokens, seed) so the read-out is matched: the SFT and DPO arms differ ONLY in the objective."""
    import torch

    start = time.monotonic()
    _set_determinism(seed)
    model, tokenizer = _load_model_and_tokenizer(attn="sdpa")
    HF_CACHE_VOL.commit()
    pad_id = tokenizer.pad_token_id

    tr = _dpo_train_loop(model, tokenizer, train_prompts, train_chosen, train_rejected, lr, beta,
                         max_steps, seed, batch_size, plateau_rel_improve, plateau_window, min_steps,
                         append_eos=append_eos, grad_clip=grad_clip, nll_weight=nll_weight)

    # Generation block: identical to sft_lm_and_generate.
    model.eval()
    torch.manual_seed(seed)
    tokenizer.padding_side = "left"
    generated: list[str] = []
    bs = 16
    for i in range(0, len(heldout_prompts), bs):
        chunk = heldout_prompts[i: i + bs]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(
                **enc, do_sample=(gen_temperature > 0), temperature=gen_temperature, top_p=0.95,
                max_new_tokens=gen_max_new_tokens, pad_token_id=pad_id,
            )
        for row in out:
            new_tokens = row[enc["input_ids"].shape[1]:]
            generated.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

    return {
        "train_losses": tr["train_losses"], "pref_accs": tr["pref_accs"],
        "reward_margins": tr["reward_margins"], "stop_step": tr["stop_step"],
        "generated": generated,
        "n_train": len(train_prompts),
        "n_heldout": len(heldout_prompts),
        "elapsed": time.monotonic() - start,
    }
