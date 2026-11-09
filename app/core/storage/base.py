"""Abstract Storage Protocol for Uploaded Dataset Persistence."""

from abc import ABC, abstractmethod


class StorageBackend(ABC):
    """Abstract interface defining operations for temporary dataset storage."""

    @abstractmethod
    async def save_file(
        self,
        workspace_id: str,
        job_id: str,
        filename: str,
        content: bytes,
    ) -> str:
        """Persist uploaded file content and return a storage reference key.

        Args:
            workspace_id: Authorized tenant workspace identifier.
            job_id: Unique processing job identifier.
            filename: Target file name.
            content: Raw byte content of the file.

        Returns:
            Unique storage file key / path.
        """
        ...

    @abstractmethod
    async def read_file(self, file_key: str) -> bytes:
        """Retrieve stored file bytes by storage key.

        Args:
            file_key: Storage reference key returned by save_file.

        Returns:
            Raw file bytes.

        Raises:
            FileNotFoundError: If the file key does not exist.
        """
        ...

    @abstractmethod
    async def delete_file(self, file_key: str) -> bool:
        """Delete an individual stored file.

        Args:
            file_key: Storage reference key.

        Returns:
            True if deleted, False if not found.
        """
        ...

    @abstractmethod
    async def delete_job_files(self, workspace_id: str, job_id: str) -> int:
        """Delete all temporary files associated with a specific job.

        Args:
            workspace_id: Tenant workspace identifier.
            job_id: Processing job identifier.

        Returns:
            Count of deleted files.
        """
        ...
