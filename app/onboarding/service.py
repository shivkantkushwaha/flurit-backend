"""Onboarding Service Orchestrator for Flurit.

Coordinates secure file upload handling, statistical CSV profiling,
concept schema mapping, canonical intermediate transformation, and
end-to-end ML analysis processing.
"""

import io
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import UploadFile
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.core.logging import get_logger
from app.onboarding.mapper import SchemaMapper
from app.onboarding.processor import DatasetProcessingEngine
from app.onboarding.profiler import CSVProfiler, sanitize_filename
from app.onboarding.schemas import (
    DatasetProfile,
    DatasetProfileResponse,
    EndToEndAnalysisResponse,
    ProcessingOptions,
    ProcessingPreviewResponse,
    ReadinessAssessmentResponse,
    SchemaMappingRequest,
)
from app.onboarding.transformer import SchemaTransformer
from app.services.analysis_persistence import AnalysisPersistenceService

logger = get_logger("flurit.onboarding.service")

CHUNK_SIZE = 64 * 1024  # 64 KB read buffer


class OnboardingService:
    """Service handling dataset upload, profiling, schema mapping, and end-to-end processing."""

    def __init__(
        self,
        profiler: CSVProfiler | None = None,
        mapper: SchemaMapper | None = None,
        transformer: SchemaTransformer | None = None,
        processor: DatasetProcessingEngine | None = None,
        persistence: AnalysisPersistenceService | None = None,
    ) -> None:
        self.settings = get_settings()
        self.profiler = profiler or CSVProfiler()
        self.mapper = mapper or SchemaMapper()
        self.transformer = transformer or SchemaTransformer()
        self.processor = processor or DatasetProcessingEngine()
        self.persistence = persistence or AnalysisPersistenceService()

    async def profile_uploaded_files(
        self,
        files: Sequence[UploadFile],
    ) -> DatasetProfileResponse:
        """Stream uploaded files with memory bounds, validate format, and profile statistics."""
        if not files:
            raise ValueError("No files provided for upload and profiling.")

        max_size = self.settings.ONBOARDING_MAX_FILE_SIZE_BYTES
        profiles: list[DatasetProfile] = []

        for upload in files:
            orig_name = upload.filename or "uploaded_dataset.csv"
            safe_name = sanitize_filename(orig_name)
            raw_bytes = await self._stream_file_bytes(upload, safe_name, max_size)

            profile = await run_in_threadpool(
                self.profiler.profile_bytes, raw_bytes=raw_bytes, original_filename=safe_name
            )
            profiles.append(profile)

        message = (
            f"Successfully profiled {len(profiles)} dataset file(s)."
            if profiles
            else "No files processed."
        )

        return DatasetProfileResponse(
            success=True,
            message=message,
            datasets=profiles,
        )

    def assess_readiness(
        self,
        request: SchemaMappingRequest,
    ) -> ReadinessAssessmentResponse:
        """Evaluate concept mapping against Flurit entities and canonical feature requirements."""
        return self.mapper.assess_readiness(request)

    async def preview_processing(
        self,
        files: Sequence[UploadFile],
        mapping: SchemaMappingRequest,
        options: ProcessingOptions | None = None,
        workspace_id: str = "ws_default",
    ) -> ProcessingPreviewResponse:
        """Dry-run transformation and readiness verification without running heavy ML."""
        raw_dfs = await self._read_files_into_dataframes(files)
        canonical_dataset = self.transformer.transform(
            raw_dfs=raw_dfs, mapping=mapping, workspace_id=workspace_id
        )
        return self.processor.preview(dataset=canonical_dataset, options=options)

    async def process_dataset(
        self,
        files: Sequence[UploadFile],
        mapping: SchemaMappingRequest,
        options: ProcessingOptions | None = None,
        workspace_id: str = "ws_default",
        session: Session | None = None,
    ) -> EndToEndAnalysisResponse:
        """Execute full end-to-end dataset transformation, snapshotting, and ML analysis."""
        raw_dfs = await self._read_files_into_dataframes(files)
        canonical_dataset = self.transformer.transform(
            raw_dfs=raw_dfs, mapping=mapping, workspace_id=workspace_id
        )
        execution = await run_in_threadpool(
            self.processor.process_with_artifacts, dataset=canonical_dataset, options=options
        )
        readiness = self.mapper.assess_readiness(mapping)
        execution.response.feature_coverage = readiness.feature_coverage
        execution.response.sentinel_scoring_used = bool(
            options and options.allow_sentinel_scoring and not execution.response.existing_model_scoring_ready
        )

        if session is not None:
            try:
                execution.response.persisted_summary = await run_in_threadpool(
                    self.persistence.persist,
                    session=session, dataset=canonical_dataset, execution=execution
                )
            except Exception:
                session.rollback()
                raise
        return execution.response

    async def _read_files_into_dataframes(
        self,
        files: Sequence[UploadFile],
    ) -> dict[str, pd.DataFrame]:
        """Read uploaded files into dictionary of Pandas DataFrames keyed by filename and stem."""
        if not files:
            raise ValueError("At least one CSV file must be provided for dataset processing.")

        max_size = self.settings.ONBOARDING_MAX_FILE_SIZE_BYTES
        dfs: dict[str, pd.DataFrame] = {}

        for upload in files:
            orig_name = upload.filename or "dataset.csv"
            safe_name = sanitize_filename(orig_name)
            stem = Path(safe_name).stem.lower()
            raw_bytes = await self._stream_file_bytes(upload, safe_name, max_size)

            text_content = self.profiler._decode_bytes(raw_bytes, safe_name)
            try:
                df = pd.read_csv(
                    io.StringIO(text_content),
                    low_memory=False,
                    encoding_errors="replace",
                )
            except Exception as exc:
                raise ValueError(
                    f"Unable to parse tabular CSV content in '{safe_name}': {exc}"
                ) from exc

            # Key by full filename and stem for easy resolution
            dfs[safe_name] = df
            dfs[stem] = df

        return dfs

    async def _stream_file_bytes(
        self,
        upload: UploadFile,
        safe_name: str,
        max_size: int,
    ) -> bytes:
        """Stream read an UploadFile with strict memory bounding."""
        total_read = 0
        chunks: list[bytes] = []

        while True:
            chunk = await upload.read(CHUNK_SIZE)
            if not chunk:
                break
            total_read += len(chunk)
            if total_read > max_size:
                max_mb = max_size / (1024 * 1024)
                raise ValueError(
                    f"File '{safe_name}' exceeds maximum allowed upload size ({max_mb:.0f} MB)."
                )
            chunks.append(chunk)

        return b"".join(chunks)


@lru_cache
def get_onboarding_service() -> OnboardingService:
    """Retrieve cached singleton instance of OnboardingService."""
    return OnboardingService()
