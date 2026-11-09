"""Background Worker Implementation for Asynchronous Dataset Processing.

Executes point-in-time snapshot generation, canonical 45-feature extraction,
ML scoring, SHAP explainability, and value-risk portfolio prioritization.
Integrates with ARQ for production Redis-backed job consumption and
provides standalone async execution for in-process brokers.
"""

import io
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from arq.connections import RedisSettings
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.storage import get_storage_backend
from app.db.session import dispose_db_engine, get_shared_engine
from app.models.job import ProcessingJob
from app.onboarding.processor import DatasetProcessingEngine
from app.onboarding.profiler import CSVProfiler
from app.onboarding.schemas import ProcessingOptions, SchemaMappingRequest
from app.onboarding.transformer import SchemaTransformer

logger = get_logger("flurit.worker")


def _sanitize_error_message(exc: Exception) -> str:
    """Sanitize error messages to prevent exposing stack traces, DB credentials, or file paths."""
    msg = str(exc).strip()
    if not msg:
        return "An unknown error occurred during dataset processing."

    # Prevent internal directory exposure
    if "C:\\" in msg or "c:\\" in msg or "/home/" in msg or "/var/" in msg:
        return "A file system or path error occurred during dataset processing."

    # Prevent database credentials/URLs leakage
    low = msg.lower()
    if "password" in low or "postgresql" in low or "select " in low or "insert " in low:
        return "A database operation failed during dataset processing."

    # Safe validation messages (truncated if too long)
    if len(msg) > 300:
        return msg[:297] + "..."
    return msg


def recover_stale_jobs(db: Session, max_age_seconds: int = 600) -> int:
    """Find jobs stuck in 'running' state past the timeout threshold and mark them as failed.

    Prevents orphaned jobs when worker processes crash, restart, or encounter OOM kills.

    Args:
        db: Active database session.
        max_age_seconds: Threshold in seconds beyond which running jobs are considered stale.

    Returns:
        Count of recovered stale jobs.
    """
    cutoff = datetime.now(UTC) - timedelta(seconds=max_age_seconds)
    stale_jobs = (
        db.execute(
            select(ProcessingJob).where(
                ProcessingJob.status == "running",
                ProcessingJob.started_at < cutoff,
            )
        )
        .scalars()
        .all()
    )

    count = 0
    now_dt = datetime.now(UTC)
    for job in stale_jobs:
        job.status = "failed"
        job.completed_at = now_dt
        job.stage_message = "Job execution timed out or was interrupted by worker restart."
        job.error_message = (
            "Processing was interrupted before completion. Please re-submit the job."
        )
        count += 1

    if count > 0:
        db.commit()
        logger.warning(
            "Recovered stale interrupted jobs on worker startup",
            extra={"recovered_count": count, "cutoff": cutoff.isoformat()},
        )

    return count


async def execute_processing_job(
    job_id: str,
    workspace_id: str,
) -> None:
    """Execute asynchronous dataset processing job with state transitions, progress, and cleanup.

    Guarantees:
      - Cancellation check at each lifecycle boundary: a job marked 'cancelled' is never
        overwritten as 'succeeded'.
      - Sanitization of error messages to exclude paths, credentials, and SQL statements.
      - Automated temporary file cleanup after processing.

    Args:
        job_id: Unique processing job identifier.
        workspace_id: Tenant workspace identifier.
    """
    settings = get_settings()
    engine = get_shared_engine()
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    storage = get_storage_backend()

    with SessionLocal() as db:
        job = db.get(ProcessingJob, job_id)
        if not job or job.workspace_id != workspace_id:
            logger.error(
                "Worker received non-existent or unauthorized job",
                extra={"job_id": job_id, "workspace_id": workspace_id},
            )
            return

        if job.status == "cancelled":
            logger.info("Skipping cancelled job", extra={"job_id": job_id})
            return

        # 1. Transition to running
        now_dt = datetime.now(UTC)
        job.status = "running"
        job.started_at = now_dt
        job.progress_pct = 10
        job.stage_message = "Retrieving uploaded dataset files"
        db.commit()

        try:
            # 2. Retrieve files from storage
            file_keys = job.file_keys or []
            if not file_keys:
                raise ValueError("No uploaded dataset files found for this processing job.")

            profiler = CSVProfiler()
            raw_dfs: dict[str, pd.DataFrame] = {}

            for fkey in file_keys:
                file_bytes = await storage.read_file(fkey)
                filename = Path(fkey).name
                stem = Path(filename).stem.lower()

                text_content = profiler._decode_bytes(file_bytes, filename)
                df = pd.read_csv(
                    io.StringIO(text_content),
                    low_memory=False,
                    encoding_errors="replace",
                )
                raw_dfs[filename] = df
                raw_dfs[stem] = df

            # Cancellation check before transform
            db.refresh(job)
            if job.status == "cancelled":
                logger.info(
                    "Job cancelled during file retrieval; terminating worker run",
                    extra={"job_id": job_id},
                )
                if settings.STORAGE_CLEANUP_AFTER_PROCESSING:
                    await storage.delete_job_files(workspace_id=workspace_id, job_id=job_id)
                return

            # 3. Intermediate transformation
            job.progress_pct = 35
            job.stage_message = "Transforming into canonical intermediate relational schema"
            db.commit()

            transformer = SchemaTransformer()
            mapping = SchemaMappingRequest.model_validate(job.mapping_payload)
            canonical_dataset = transformer.transform(
                raw_dfs=raw_dfs,
                mapping=mapping,
                workspace_id=workspace_id,
            )

            # Cancellation check before snapshotting
            db.refresh(job)
            if job.status == "cancelled":
                logger.info(
                    "Job cancelled before snapshotting; terminating worker run",
                    extra={"job_id": job_id},
                )
                if settings.STORAGE_CLEANUP_AFTER_PROCESSING:
                    await storage.delete_job_files(workspace_id=workspace_id, job_id=job_id)
                return

            # 4. Snapshot generation & Feature engineering
            job.progress_pct = 65
            job.stage_message = "Generating customer snapshots and canonical 45-feature matrix"
            db.commit()

            opts = (
                ProcessingOptions.model_validate(job.options_payload)
                if job.options_payload
                else None
            )
            processor = DatasetProcessingEngine()

            # Cancellation check before model scoring
            db.refresh(job)
            if job.status == "cancelled":
                logger.info(
                    "Job cancelled before model scoring; terminating worker run",
                    extra={"job_id": job_id},
                )
                if settings.STORAGE_CLEANUP_AFTER_PROCESSING:
                    await storage.delete_job_files(workspace_id=workspace_id, job_id=job_id)
                return

            # 5. Model scoring, explainability & prioritization
            job.progress_pct = 85
            job.stage_message = "Scoring customer risk, attributing SHAP factors, and segmenting"
            db.commit()

            analysis_resp = processor.process(dataset=canonical_dataset, options=opts)

            # 6. Evaluation and result assembly - VERIFY NOT CANCELLED BEFORE SUCCEEDED
            db.refresh(job)
            if job.status == "cancelled":
                logger.info(
                    "Job was cancelled while computation was running; discarding results and preserving cancelled status",
                    extra={"job_id": job_id, "workspace_id": workspace_id},
                )
                if settings.STORAGE_CLEANUP_AFTER_PROCESSING:
                    await storage.delete_job_files(workspace_id=workspace_id, job_id=job_id)
                return

            end_dt = datetime.now(UTC)
            job.completed_at = end_dt

            if analysis_resp.processing_status == "FAILED":
                job.status = "failed"
                job.stage_message = "Processing failed: schema or relational blockers detected"
                job.error_message = "; ".join(
                    analysis_resp.scoring_blockers or ["Dataset processing validation failed."]
                )
                job.result_summary = analysis_resp.model_dump(mode="json")
            else:
                job.status = "succeeded"
                job.progress_pct = 100
                job.stage_message = "Analysis completed successfully"
                job.result_summary = analysis_resp.model_dump(mode="json")
                job.error_message = None

            db.commit()
            logger.info(
                "Completed processing job execution",
                extra={
                    "job_id": job_id,
                    "workspace_id": workspace_id,
                    "status": job.status,
                },
            )

            # 7. Safe cleanup of temporary files
            if settings.STORAGE_CLEANUP_AFTER_PROCESSING:
                await storage.delete_job_files(workspace_id=workspace_id, job_id=job_id)

        except Exception as exc:
            logger.exception(
                "Uncaught exception during worker job execution",
                extra={"job_id": job_id, "workspace_id": workspace_id},
            )

            db.refresh(job)
            # If job was cancelled concurrently, do not overwrite status with failed
            if job.status != "cancelled":
                job.status = "failed"
                job.completed_at = datetime.now(UTC)
                job.stage_message = "Job execution failed"
                job.error_message = _sanitize_error_message(exc)
                db.commit()

            if settings.STORAGE_CLEANUP_AFTER_PROCESSING:
                await storage.delete_job_files(workspace_id=workspace_id, job_id=job_id)


async def process_dataset_job(ctx: dict[str, Any], job_id: str, workspace_id: str) -> None:
    """Entry point for ARQ Redis worker."""
    await execute_processing_job(job_id=job_id, workspace_id=workspace_id)


class WorkerSettings:
    """ARQ Worker configuration for production Redis queue processing."""

    functions = [process_dataset_job]
    max_jobs = 10
    job_timeout = get_settings().QUEUE_JOB_TIMEOUT_SECONDS
    max_tries = get_settings().QUEUE_MAX_RETRIES
    redis_settings = RedisSettings.from_dsn(get_settings().REDIS_URL)

    @staticmethod
    async def on_startup(ctx: dict[str, Any]) -> None:
        """Worker startup hook: recover stale interrupted jobs and log readiness."""
        settings = get_settings()
        engine = get_shared_engine()
        SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
        with SessionLocal() as db:
            recover_stale_jobs(db, max_age_seconds=settings.QUEUE_JOB_TIMEOUT_SECONDS)
        logger.info("ARQ background worker initialized and ready to consume jobs")

    @staticmethod
    async def on_shutdown(ctx: dict[str, Any]) -> None:
        """Worker shutdown hook: clean up worker resources and dispose database engine."""
        dispose_db_engine()
        logger.info("ARQ background worker shutting down gracefully")
