"""Abstract Queue Broker Protocol for Background Job Dispatching."""

from abc import ABC, abstractmethod


class QueueBroker(ABC):
    """Abstract interface defining background job dispatching operations."""

    @abstractmethod
    async def enqueue_processing_job(
        self,
        job_id: str,
        workspace_id: str,
    ) -> str:
        """Dispatch a dataset processing job to the background queue.

        Args:
            job_id: Unique processing job identifier.
            workspace_id: Tenant workspace identifier.

        Returns:
            Dispatched job identifier.
        """
        ...
