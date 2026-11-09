"""Synchronous queue broker executing jobs immediately in the same call."""

from app.core.logging import get_logger
from app.core.queue.base import QueueBroker
from app.worker import execute_processing_job

logger = get_logger("flurit.queue.sync")


class SyncQueueBroker(QueueBroker):
    """Synchronous queue broker awaiting job execution directly before returning.

    Mainly used for deterministic testing or single-process debugging.
    """

    async def enqueue_processing_job(
        self,
        job_id: str,
        workspace_id: str,
    ) -> str:
        """Execute the job immediately before returning.

        Args:
            job_id: Processing job identifier.
            workspace_id: Tenant workspace identifier.

        Returns:
            Dispatched job identifier.
        """
        logger.info(
            "Executing job synchronously in SyncQueueBroker",
            extra={"job_id": job_id, "workspace_id": workspace_id},
        )
        await execute_processing_job(job_id=job_id, workspace_id=workspace_id)
        return job_id
