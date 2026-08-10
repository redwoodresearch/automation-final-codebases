"""Local, transient activation cache (phase probe_pipeline_and_readout_controls).

All-layer activation arrays are large (~70 MB / cell / representation as float32), so they bypass the
JSON-based FileCache and are stored as ``.npz`` files keyed by a sha256 of the extraction request. This
directory is **gitignored scratch, not a deliverable**: per INSTRUCTIONS §3 raw activations are never
committed and never put on a Modal Volume — they exist only to make the local control/C/layer iteration
fast (extract once on the GPU, fit many variants on CPU). It is safe to delete after the phase; the
scale-up phase re-extracts on demand rather than caching tens of thousands of texts.

Entries are written atomically (temp-then-hardlink) and never modified/deleted, so the cache is safe to
share across processes (same contract as FileCache).
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

ACT_CACHE_DIR = Path(__file__).resolve().parent / "activation_cache"

# Bump only if the extraction CODE changes in a way that alters activations (the image id is already in
# the key, so a pip/env/metric.py change invalidates automatically; this covers logic changes to
# _extract_all_layer_repr that don't change the image).
EXTRACT_CALL_UUID = "extract-blob-v1"


def make_key(groups: dict[str, list[str]], representation: str, wrap: str, image_id: str,
             prompt_template: str | None = None, a_char: str | None = None, b_char: str | None = None,
             model_id: str | None = None, dtype: str | None = None) -> dict:
    """Cache key capturing everything that determines the activations. The (potentially large) text
    lists are hashed rather than stored inline."""
    groups_blob = json.dumps({k: groups[k] for k in sorted(groups)}, sort_keys=True).encode()
    key = {
        "groups_sha": hashlib.sha256(groups_blob).hexdigest(),
        "group_names": sorted(groups),
        "representation": representation,
        "wrap": wrap,
        "prompt_template": prompt_template,
        "a_char": a_char,
        "b_char": b_char,
        "image_id": image_id,
        "call_uuid": EXTRACT_CALL_UUID,
    }
    if model_id:  # only added for cross-model reads, so the pinned Qwen3-1.7B cache keys are unchanged
        key["model_id"] = model_id
    if dtype:  # ditto: only added when a non-default dtype is requested, so committed keys are unchanged
        key["dtype"] = dtype
    return key


# Signature-stats extraction (stage 5, v2_activation_interp): a distinct cache namespace from the
# per-text blob extraction above. The key adds keep_layers (the per-text layers stored) since that
# changes the payload. Bump SIG_CALL_UUID only if extract_signature_stats' pooling logic changes.
SIG_CALL_UUID = "sig-stats-v1"


def make_sig_key(groups: dict[str, list[str]], keep_layers: list[int], wrap: str, image_id: str) -> dict:
    groups_blob = json.dumps({k: groups[k] for k in sorted(groups)}, sort_keys=True).encode()
    return {
        "groups_sha": hashlib.sha256(groups_blob).hexdigest(),
        "group_names": sorted(groups),
        "keep_layers": list(keep_layers),
        "wrap": wrap,
        "image_id": image_id,
        "call_uuid": SIG_CALL_UUID,
    }


def _path(key: dict) -> Path:
    h = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    return ACT_CACHE_DIR / f"{h}.npz"


def path(key: dict) -> Path:
    """Public: the .npz path for a key (recorded in the extraction manifest so the CPU fitting step can
    load activations without a Modal session)."""
    return _path(key)


def load_path(npz_path: str | Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as z:
        return {name: z[name] for name in z.files}


def get(key: dict) -> dict[str, np.ndarray] | None:
    """Return {group_name: activations} or None on miss."""
    p = _path(key)
    if not p.exists():
        return None
    with np.load(p) as z:
        return {name: z[name] for name in z.files}


def put_npz_bytes(key: dict, npz_bytes: bytes) -> None:
    ACT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _path(key)
    if p.exists():
        return
    fd, tmp = tempfile.mkstemp(dir=ACT_CACHE_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(npz_bytes)
        os.link(tmp, p)
    except FileExistsError:
        pass
    finally:
        os.unlink(tmp)
