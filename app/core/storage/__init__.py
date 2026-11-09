"""Storage backend package for temporary dataset file staging."""

from app.core.config import Settings, get_settings
from app.core.storage.base import StorageBackend
from app.core.storage.local import LocalFileStorage
from app.core.storage.memory import MemoryFileStorage

_cached_storage_backend: StorageBackend | None = None


def get_storage_backend(settings: Settings | None = None) -> StorageBackend:
    """Retrieve storage backend based on application configuration."""
    global _cached_storage_backend
    if settings is not None:
        if settings.STORAGE_BACKEND == "memory":
            return MemoryFileStorage()
        return LocalFileStorage(base_dir=settings.STORAGE_LOCAL_DIR)

    if _cached_storage_backend is None:
        cfg = get_settings()
        if cfg.STORAGE_BACKEND == "memory":
            _cached_storage_backend = MemoryFileStorage()
        else:
            _cached_storage_backend = LocalFileStorage(base_dir=cfg.STORAGE_LOCAL_DIR)

    return _cached_storage_backend


def clear_storage_backend_cache() -> None:
    """Clear cached storage backend singleton (useful in tests)."""
    global _cached_storage_backend
    _cached_storage_backend = None


__all__ = [
    "StorageBackend",
    "LocalFileStorage",
    "MemoryFileStorage",
    "get_storage_backend",
    "clear_storage_backend_cache",
]
