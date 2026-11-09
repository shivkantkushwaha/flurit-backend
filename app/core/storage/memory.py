"""In-memory storage backend for fast isolated testing without disk I/O."""

from app.core.storage.base import StorageBackend


class MemoryFileStorage(StorageBackend):
    """Ephemeral in-memory storage backend storing bytes in a dictionary."""

    def __init__(self) -> None:
        self._storage: dict[str, bytes] = {}

    async def save_file(
        self,
        workspace_id: str,
        job_id: str,
        filename: str,
        content: bytes,
    ) -> str:
        """Store bytes in memory with a virtual path key."""
        file_key = f"mem://{workspace_id}/{job_id}/{filename}"
        self._storage[file_key] = content
        return file_key

    async def read_file(self, file_key: str) -> bytes:
        """Retrieve stored bytes from memory."""
        if file_key not in self._storage:
            raise FileNotFoundError(f"In-memory file not found: {file_key}")
        return self._storage[file_key]

    async def delete_file(self, file_key: str) -> bool:
        """Delete an in-memory file."""
        return self._storage.pop(file_key, None) is not None

    async def delete_job_files(self, workspace_id: str, job_id: str) -> int:
        """Delete all in-memory files for a job."""
        prefix = f"mem://{workspace_id}/{job_id}/"
        keys_to_del = [k for k in self._storage if k.startswith(prefix)]
        for k in keys_to_del:
            del self._storage[k]
        return len(keys_to_del)
