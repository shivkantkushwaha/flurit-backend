"""ARQ Redis Queue Broker for Production Asynchronous Job Processing."""

import arq
from arq.connections import ArqRedis, RedisSettings

from app.core.logging import get_logger
from app.core.queue.base import QueueBroker

logger = get_logger("flurit.queue.arq")


class ARQQueueBroker(QueueBroker):
    """Production Redis-backed queue broker using ARQ.

    Enqueues background dataset processing jobs to a shared Redis instance
    for consumption by dedicated ARQ worker processes.
    """

    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url
        self._pool: ArqRedis | None = None

    async def get_pool(self) -> ArqRedis:
        """Lazily initialize and return the Redis connection pool."""
        if self._pool is None:
            settings = RedisSettings.from_dsn(self.redis_url)
            self._pool = await arq.create_pool(settings)
        return self._pool

    async def enqueue_processing_job(
        self,
        job_id: str,
        workspace_id: str,
    ) -> str:
        """Enqueue a dataset processing job to the ARQ Redis queue.

        Args:
            job_id: Unique processing job identifier.
            workspace_id: Tenant workspace identifier.

        Returns:
            Dispatched job identifier.
        """
        logger.info(
            "Enqueueing job to ARQ Redis",
            extra={"job_id": job_id, "workspace_id": workspace_id},
        )
        pool = await self.get_pool()
        arq_job = await pool.enqueue_job(
            "process_dataset_job",
            job_id=job_id,
            workspace_id=workspace_id,
            _job_id=f"flurit:{workspace_id}:{job_id}",
        )
        if arq_job is None:
            logger.warning(
                "ARQ duplicate job suppressed",
                extra={"job_id": job_id, "workspace_id": workspace_id},
            )
        return job_id

    async def close(self) -> None:
        """Close the underlying connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
