"""Disabled Queue Broker for simplified single-process deployments."""

from app.core.queue.base import QueueBroker


class DisabledQueueBroker(QueueBroker):
    """Queue broker that rejects asynchronous background jobs when queues are disabled."""

    async def enqueue_processing_job(
        self,
        job_id: str,
        workspace_id: str,
    ) -> str:
        """Reject asynchronous job enqueueing."""
        raise RuntimeError(
            "Asynchronous background processing is disabled in this deployment. "
            "Please use the synchronous processing endpoint POST /api/v1/onboarding/process, "
            "or configure QUEUE_MODE='arq' backed by Redis for background queue execution."
        )
