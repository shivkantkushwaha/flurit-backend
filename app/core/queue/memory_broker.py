"""In-memory async queue broker for local development and test execution."""

import asyncio
from typing import Any

from app.core.logging import get_logger
from app.core.queue.base import QueueBroker
from app.worker import execute_processing_job

logger = get_logger("flurit.queue.memory")


class MemoryQueueBroker(QueueBroker):
    """In-memory queue broker executing background processing tasks asynchronously.

    Runs jobs in background asyncio tasks without requiring external message brokers
    like Redis or RabbitMQ. Ideal for unit tests, local development, and CI environments.
    """

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[Any]] = set()

    async def enqueue_processing_job(
        self,
        job_id: str,
        workspace_id: str,
    ) -> str:
        """Schedule a dataset processing job in the background via asyncio.create_task.

        Args:
            job_id: Processing job identifier.
            workspace_id: Tenant workspace identifier.

        Returns:
            Dispatched job identifier.
        """
        logger.info(
            "Scheduling job in MemoryQueueBroker",
            extra={"job_id": job_id, "workspace_id": workspace_id},
        )

        task = asyncio.create_task(
            execute_processing_job(job_id=job_id, workspace_id=workspace_id),
            name=f"processing-job-{job_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

        return job_id

    async def wait_all(self, timeout: float | None = None) -> None:
        """Wait for all pending in-memory background tasks to complete (useful in tests)."""
        if self._tasks:
            await asyncio.wait(self._tasks, timeout=timeout)
