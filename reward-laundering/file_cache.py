"""
Simple file-based cache that stores each entry as an individual file.

This module is intended to be copied into your project and modified as needed.
You may want to customize the serialize/deserialize functions for your use case
(e.g., use pickle for Python objects, msgpack for performance, etc.).

NEVER delete or modify cache entry files - this ensures the cache is safe to
share between multiple agents/processes. This implementation never
deletes/modifies entry files itself, so it's trivial to follow this advice if
you only interface with the cache through this code. If you MUST invalidate old
cache entries due to changing the code being cached (not just changing inputs),
change the UUID in your cache key rather than deleting files.

## Concurrent Access

This cache is safe for concurrent access from multiple processes without locks.
The get_or_set / get_or_compute_set methods use write-to-temp-then-hardlink, which is
atomic at the filesystem level - content is always complete, and only one process can
create a given cache entry.

Dependencies:
- aiofiles (for async operations): pip install aiofiles
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Awaitable

# Optional import for async operations
try:
    import aiofiles
except ImportError:
    aiofiles = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import aiofiles

T = TypeVar("T")


# Modify this function if you need different serialization (e.g., pickle, msgpack).
def serialize(value: Any, *, deterministic: bool = False) -> str:
    return json.dumps(value, sort_keys=deterministic)


# Modify this function to match your serialize() implementation.
def deserialize(data: str) -> Any:
    return json.loads(data)


class FileCache:
    """
    A simple file-based cache that stores each entry in a separate file.

    Keys are hashed to create filenames, values are serialized to file contents.

    Safe for concurrent access from multiple processes via write-to-temp-then-
    hardlink, which is atomic at the filesystem level. If multiple processes
    race to cache the same key, exactly one wins and all others use that result.

    Example:
        cache = FileCache("./file_cache_dir/")

        # With a compute function (most common)
        async def fetch_data():
            return await api_call()
        result, computed = await cache.aget_or_compute_set(key, fetch_data, assert_cached=assert_cached)

        # With a value (when you need control over caching)
        cached = await cache.aget(key)
        if cached is not None:
            return cached
        if assert_cached:
            raise ...
        # ... do operation
        result, was_set = await cache.aget_or_set(key, value)
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_filename(self, key: Any) -> Path:
        key_str = serialize(key, deterministic=True)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def _atomic_write(self, filepath: Path, data: str) -> bool:
        """Write data to filepath atomically, only if it doesn't already exist.

        Uses write-to-temp-then-hardlink: writes to a temp file first, then
        os.link() to the target (atomic and fails if target exists).
        Returns True if written, False if file already existed.
        """
        fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            os.link(tmp_path, filepath)
            return True
        except FileExistsError:
            return False
        finally:
            os.unlink(tmp_path)

    async def _async_atomic_write(self, filepath: Path, data: str) -> bool:
        """Async version of _atomic_write."""
        fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        try:
            os.close(fd)
            async with aiofiles.open(tmp_path, "w", encoding="utf-8") as f:
                await f.write(data)
            os.link(tmp_path, filepath)
            return True
        except FileExistsError:
            return False
        finally:
            os.unlink(tmp_path)

    def get(self, key: Any, default: T = None) -> Any | T:
        filepath = self._key_to_filename(key)
        if not filepath.exists():
            return default

        with open(filepath, "r", encoding="utf-8") as f:
            return deserialize(f.read())

    def get_or_set(self, key: Any, value: T) -> tuple[T, bool]:
        """
        Get cached value or cache the provided value.

        Returns:
            Tuple of (value, was_set) where was_set is True if we stored the
            value (cache miss), False if it was already cached.
        """
        # Fast path: already cached
        cached = self.get(key)
        if cached is not None:
            return cached, False

        # Try to write atomically
        filepath = self._key_to_filename(key)
        if self._atomic_write(filepath, serialize(value)):
            return value, True
        # Another process wrote first - use their value
        return self.get(key, value), False

    def get_or_compute_set(
        self, key: Any, compute_fn: Callable[[], T], *, assert_cached: bool = False
    ) -> tuple[T, bool]:
        """
        Get cached value or compute and cache it. Safe for concurrent access.

        Returns:
            Tuple of (value, computed) where computed is True if compute_fn was
            called, False if value came from cache. Note: even if computed=True,
            value may be from cache if there was a concurrent write.
        """
        # Fast path: already cached
        cached = self.get(key)
        if cached is not None:
            return cached, False

        if assert_cached:
            raise RuntimeError(f"assert_cached=True but cache miss for key: {key}")

        # Compute the value
        value = compute_fn()

        # Re-check before write (saves serialization if another process wrote)
        cached = self.get(key)
        if cached is not None:
            return cached, True  # We computed, but use cached value

        # Try to write atomically
        filepath = self._key_to_filename(key)
        if self._atomic_write(filepath, serialize(value)):
            return value, True
        # Another process wrote first - use their value
        return self.get(key, value), True

    def exists(self, key: Any) -> bool:
        return self._key_to_filename(key).exists()

    async def aget(self, key: Any, default: T = None) -> Any | T:
        """Requires: pip install aiofiles"""
        if aiofiles is None:
            raise ImportError("aiofiles is required for async operations: pip install aiofiles")

        filepath = self._key_to_filename(key)
        if not filepath.exists():
            return default

        async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
            content = await f.read()
            return deserialize(content)

    async def aget_or_set(self, key: Any, value: T) -> tuple[T, bool]:
        """
        Async version of get_or_set().

        Returns:
            Tuple of (value, was_set) where was_set is True if we stored the
            value (cache miss), False if it was already cached.

        Requires: pip install aiofiles
        """
        if aiofiles is None:
            raise ImportError("aiofiles is required for async operations: pip install aiofiles")

        # Fast path: already cached
        cached = await self.aget(key)
        if cached is not None:
            return cached, False

        # Try to write atomically
        filepath = self._key_to_filename(key)
        if await self._async_atomic_write(filepath, serialize(value)):
            return value, True
        # Another process wrote first - use their value
        return await self.aget(key, value), False

    async def aget_or_compute_set(
        self, key: Any, compute_fn: Callable[[], Awaitable[T]], *, assert_cached: bool = False
    ) -> tuple[T, bool]:
        """
        Async version of get_or_compute_set().

        Returns:
            Tuple of (value, computed) where computed is True if compute_fn was
            called, False if value came from cache. Note: even if computed=True,
            value may be from cache if there was a concurrent write.

        Requires: pip install aiofiles
        """
        if aiofiles is None:
            raise ImportError("aiofiles is required for async operations: pip install aiofiles")

        # Fast path: already cached
        cached = await self.aget(key)
        if cached is not None:
            return cached, False

        if assert_cached:
            raise RuntimeError(f"assert_cached=True but cache miss for key: {key}")

        # Compute the value
        value = await compute_fn()

        # Re-check before write (saves serialization if another process wrote)
        cached = await self.aget(key)
        if cached is not None:
            return cached, True  # We computed, but use cached value

        # Try to write atomically
        filepath = self._key_to_filename(key)
        if await self._async_atomic_write(filepath, serialize(value)):
            return value, True
        # Another process wrote first - use their value
        return await self.aget(key, value), True

    async def aexists(self, key: Any) -> bool:
        # use sync check as fast enough
        return self.exists(key)
