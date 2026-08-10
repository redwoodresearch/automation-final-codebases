"""Thin wrapper around Modal GPU calls: FileCache caching (keyed on the image id + all args) plus
Modal cost tracking. Enter a session once, then make many cached calls inside it.

AsyncModalSession parallelizes many GPU jobs (fine-tune grid, probe over pairs) via .remote.aio.
"""

from __future__ import annotations

import modal

import act_cache
import config
import modal_app as M
from cost_tracker import CostTracker
from file_cache import FileCache

_cache: FileCache | None = None


def _get_cache() -> FileCache:
    """Lazily construct the FileCache (it mkdirs CACHE_DIR) so importing this module has no
    filesystem side effects."""
    global _cache
    if _cache is None:
        _cache = FileCache(str(config.CACHE_DIR))
    return _cache


def _cache_key(fn_name: str, kwargs: dict, image_id: str, call_uuid: str) -> dict:
    return {"fn": fn_name, "kwargs": kwargs, "image_id": image_id, "this_call_uuid": call_uuid}


class AsyncModalSession:
    """Opens the Modal app once (async), resolves the image id for cache keys, and runs cached GPU
    calls concurrently. Cost is tracked only on cache miss."""

    def __init__(self, tracker: CostTracker, assert_cached: bool = False, app_module=None):
        self.tracker = tracker
        self.assert_cached = assert_cached
        self.image_id: str | None = None
        self.M = app_module if app_module is not None else M

    async def __aenter__(self) -> "AsyncModalSession":
        self._out = modal.enable_output()  # sync-only context, nested inside the async app run
        self._out.__enter__()
        self._run = self.M.app.run.aio()
        await self._run.__aenter__()
        self.image_id = self.M.image.object_id
        return self

    async def __aexit__(self, *exc):
        await self._run.__aexit__(*exc)
        self._out.__exit__(*exc)

    async def call(self, fn_name: str, modal_fn, kwargs: dict, call_uuid: str, gpu: str = M.GPU) -> tuple[dict, bool]:
        key = _cache_key(fn_name, kwargs, self.image_id, call_uuid)

        async def compute():
            result = await modal_fn.remote.aio(**kwargs)
            self.tracker.add_modal_gpu_cost(wall_seconds=result["elapsed"], gpu=gpu, is_sandbox=False)
            return result

        return await _get_cache().aget_or_compute_set(key, compute, assert_cached=self.assert_cached)

    async def extract_blob(self, groups: dict[str, list[str]], representation: str, wrap: str = "none",
                           prompt_template: str | None = None, a_char: str | None = None,
                           b_char: str | None = None, gpu: str = M.GPU,
                           model_id: str | None = None, dtype: str | None = None) -> tuple[dict, bool]:
        """Extract all-layer activations for named text groups, caching the (large) .npz blob in the
        local activation cache (NOT the JSON FileCache — bytes don't JSON-serialize). Returns
        ({group_name: ndarray}, computed). model_id/dtype select the model (cross-model reads); None =
        the pinned Qwen3-1.7B (and the cache key is then unchanged)."""
        key = act_cache.make_key(groups, representation, wrap, self.image_id, prompt_template, a_char, b_char,
                                 model_id=model_id, dtype=dtype)
        cached = act_cache.get(key)
        if cached is not None:
            return cached, False
        if self.assert_cached:
            raise RuntimeError(f"assert_cached=True but activation cache miss: repr={representation} groups={list(groups)}")
        result = await M.extract_activations_blob.remote.aio(
            groups, representation=representation, wrap=wrap,
            prompt_template=prompt_template, a_char=a_char, b_char=b_char,
            model_id=model_id, dtype=dtype,
        )
        self.tracker.add_modal_gpu_cost(wall_seconds=result["elapsed"], gpu=gpu, is_sandbox=False)
        act_cache.put_npz_bytes(key, result["npz"])
        return act_cache.get(key), True

    async def extract_signature_stats(self, groups: dict[str, list[str]], keep_layers: list[int],
                                      wrap: str = "none", gpu: str = M.GPU) -> tuple[dict, bool]:
        """Cached call to extract_signature_stats (signature pipeline): per-cell per-layer
        means under BOTH reads + per-text acts at keep_layers, in one forward pass. Blob cached
        locally (bytes don't JSON-serialize, so this bypasses the FileCache like extract_blob)."""
        key = act_cache.make_sig_key(groups, keep_layers, wrap, self.image_id)
        cached = act_cache.get(key)
        if cached is not None:
            return cached, False
        if self.assert_cached:
            raise RuntimeError(f"assert_cached=True but signature-stats cache miss: groups={list(groups)}")
        result = await self.M.extract_signature_stats.remote.aio(groups, keep_layers=keep_layers, wrap=wrap)
        self.tracker.add_modal_gpu_cost(wall_seconds=result["elapsed"], gpu=gpu, is_sandbox=False)
        act_cache.put_npz_bytes(key, result["npz"])
        return act_cache.get(key), True
