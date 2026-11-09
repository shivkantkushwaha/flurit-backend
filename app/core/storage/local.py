"""Local filesystem storage backend for temporary dataset file staging."""

import asyncio
import shutil
from pathlib import Path

from app.core.logging import get_logger
from app.core.storage.base import StorageBackend
from app.onboarding.profiler import sanitize_filename

logger = get_logger("flurit.storage.local")


class LocalFileStorage(StorageBackend):
    """Local filesystem storage backend storing files under a configurable root directory."""

    def __init__(self, base_dir: str | Path = "data/uploads") -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_target_path(self, workspace_id: str, job_id: str, filename: str) -> Path:
        """Resolve safe target path with traversal protection."""
        if (
            ".." in filename
            or ".." in workspace_id
            or ".." in job_id
            or "/" in filename
            or "\\" in filename
        ):
            raise ValueError(f"Directory traversal detected in storage path resolution: {filename}")
        clean_file = sanitize_filename(filename)
        clean_ws = sanitize_filename(workspace_id)
        clean_job = sanitize_filename(job_id)

        target = (self.base_dir / clean_ws / clean_job / clean_file).resolve()
        if not str(target).startswith(str(self.base_dir)):
            raise ValueError(f"Path traversal detected in storage path resolution: {filename}")
        return target

    def _resolve_key_path(self, file_key: str) -> Path:
        """Resolve an existing file_key and verify it resides within base_dir."""
        target = Path(file_key).resolve()
        if not str(target).startswith(str(self.base_dir)):
            raise ValueError(f"Path traversal detected for file key: {file_key}")
        return target

    async def save_file(
        self,
        workspace_id: str,
        job_id: str,
        filename: str,
        content: bytes,
    ) -> str:
        """Save file content to local disk under workspace_id/job_id."""
        target_path = self._get_target_path(workspace_id, job_id, filename)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        def _write() -> None:
            with open(target_path, "wb") as f:
                f.write(content)

        await asyncio.to_thread(_write)
        logger.debug(
            "Saved temporary file to local storage",
            extra={"workspace_id": workspace_id, "job_id": job_id, "path": str(target_path)},
        )
        return str(target_path)

    async def read_file(self, file_key: str) -> bytes:
        """Read bytes from local disk."""
        target_path = self._resolve_key_path(file_key)
        if not target_path.exists() or not target_path.is_file():
            raise FileNotFoundError(f"Storage file not found: {file_key}")

        def _read() -> bytes:
            with open(target_path, "rb") as f:
                return f.read()

        return await asyncio.to_thread(_read)

    async def delete_file(self, file_key: str) -> bool:
        """Delete file from local disk."""
        try:
            target_path = self._resolve_key_path(file_key)
            if target_path.exists() and target_path.is_file():
                target_path.unlink()
                return True
            return False
        except Exception as exc:
            logger.warning("Failed to delete file", extra={"file_key": file_key, "error": str(exc)})
            return False

    async def delete_job_files(self, workspace_id: str, job_id: str) -> int:
        """Delete entire job staging directory."""
        clean_ws = sanitize_filename(workspace_id)
        clean_job = sanitize_filename(job_id)
        job_dir = (self.base_dir / clean_ws / clean_job).resolve()

        if not str(job_dir).startswith(str(self.base_dir)):
            raise ValueError(f"Path traversal detected in job directory deletion: {job_dir}")

        if not job_dir.exists():
            return 0

        def _cleanup() -> int:
            count = sum(1 for p in job_dir.glob("*") if p.is_file())
            shutil.rmtree(job_dir, ignore_errors=True)
            return count

        deleted = await asyncio.to_thread(_cleanup)
        logger.info(
            "Cleaned up temporary job files",
            extra={"workspace_id": workspace_id, "job_id": job_id, "deleted_count": deleted},
        )
        return deleted
