"""Queue broker package for asynchronous background job execution."""

from app.core.config import Settings, get_settings
from app.core.queue.arq_broker import ARQQueueBroker
from app.core.queue.base import QueueBroker
from app.core.queue.disabled_broker import DisabledQueueBroker
from app.core.queue.memory_broker import MemoryQueueBroker
from app.core.queue.sync_broker import SyncQueueBroker

_cached_queue_broker: QueueBroker | None = None


def get_queue_broker(settings: Settings | None = None) -> QueueBroker:
    """Retrieve background queue broker based on configuration.

    - "arq": Production Redis-backed ARQ worker queue.
    - "memory": In-process asynchronous task scheduler (default for dev/testing/simplified).
    - "sync": Direct synchronous execution for single-step deterministic runs.
    - "disabled": Explicitly rejects background async submissions.
    """
    global _cached_queue_broker

    if settings is not None:
        if settings.is_production and settings.DEPLOYMENT_TIER == "enterprise":
            if settings.QUEUE_MODE != "arq":
                raise ValueError(
                    f"Production environment requires QUEUE_MODE='arq' backed by Redis, but found '{settings.QUEUE_MODE}'. "
                    "In-memory or synchronous queue fallback is strictly prohibited in production."
                )
            return ARQQueueBroker(redis_url=settings.REDIS_URL)

        if settings.QUEUE_MODE == "arq":
            return ARQQueueBroker(redis_url=settings.REDIS_URL)
        if settings.QUEUE_MODE == "sync":
            return SyncQueueBroker()
        if settings.QUEUE_MODE == "disabled":
            return DisabledQueueBroker()
        return MemoryQueueBroker()

    if _cached_queue_broker is None:
        cfg = get_settings()
        if cfg.is_production and cfg.DEPLOYMENT_TIER == "enterprise":
            if cfg.QUEUE_MODE != "arq":
                raise ValueError(
                    f"Production environment requires QUEUE_MODE='arq' backed by Redis, but found '{cfg.QUEUE_MODE}'. "
                    "In-memory or synchronous queue fallback is strictly prohibited in production."
                )
            _cached_queue_broker = ARQQueueBroker(redis_url=cfg.REDIS_URL)
        elif cfg.QUEUE_MODE == "arq":
            _cached_queue_broker = ARQQueueBroker(redis_url=cfg.REDIS_URL)
        elif cfg.QUEUE_MODE == "sync":
            _cached_queue_broker = SyncQueueBroker()
        elif cfg.QUEUE_MODE == "disabled":
            _cached_queue_broker = DisabledQueueBroker()
        else:
            _cached_queue_broker = MemoryQueueBroker()

    return _cached_queue_broker


def clear_queue_broker_cache() -> None:
    """Clear cached broker singleton (useful in tests)."""
    global _cached_queue_broker
    _cached_queue_broker = None


__all__ = [
    "QueueBroker",
    "MemoryQueueBroker",
    "SyncQueueBroker",
    "ARQQueueBroker",
    "DisabledQueueBroker",
    "get_queue_broker",
    "clear_queue_broker_cache",
]
