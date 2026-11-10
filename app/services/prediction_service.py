"""Batch Prediction and Persistence Service Orchestrator.

Coordinates high-throughput batch scoring, SHAP explainability generation,
unsupervised customer segmentation, and ACID-compliant SQLAlchemy database persistence.
"""

from typing import Any

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.ml.inference.batch import (
    BatchInferencePipeline,
    BatchInputRecord,
    BatchPipelineSummary,
)
from app.repositories.prediction_repository import PredictionRepository

logger = get_logger("flurit.services.prediction")


class BatchPredictionService:
    """Service orchestrating batch inference pipeline execution and database persistence."""

    def __init__(
        self,
        pipeline: BatchInferencePipeline | None = None,
        repository: PredictionRepository | None = None,
    ) -> None:
        """Initialize BatchPredictionService.

        Args:
            pipeline: BatchInferencePipeline instance. If None, initialized with defaults.
            repository: PredictionRepository instance. If None, initialized with defaults.
        """
        self.pipeline = pipeline or BatchInferencePipeline()
        self.repository = repository or PredictionRepository()

    def run_and_persist_batch(
        self,
        session: Session,
        records: list[BatchInputRecord] | list[dict[str, Any]],
        batch_size: int | None = None,
        workspace_id: str = "ws_default",
    ) -> BatchPipelineSummary:
        """Execute batch inference across customer records and persist outputs to database.

        Manages strict transaction boundaries: all predictions, explanations, and segment
        assignments are committed atomically upon success. Any database failure triggers an
        immediate rollback to prevent partial or inconsistent state.

        Args:
            session: Active SQLAlchemy database session.
            records: Collection of point-in-time customer feature records.
            batch_size: Optional override for batch processing chunk size.

        Returns:
            BatchPipelineSummary with execution counts, predictions, and any validation errors.

        Raises:
            Exception: If database persistence or pipeline execution fails.
        """
        if batch_size is not None:
            self.pipeline.batch_size = batch_size

        logger.info(
            "Executing batch prediction and persistence workflow",
            extra={"record_count": len(records), "batch_size": self.pipeline.batch_size},
        )

        # 1. Run batch inference pipeline
        summary = self.pipeline.process_records(records)

        # 2. Persist successful predictions to database
        if summary.predictions:
            try:
                # Ensure active ModelRun is registered
                meta = self.pipeline.service.metadata or {}
                model_run = self.repository.get_or_create_model_run(
                    session=session,
                    model_name=summary.model_name,
                    version=summary.model_version,
                    metadata=meta,
                )

                # Persist batch predictions, explanations, and segments idempotently
                persisted_count = self.repository.save_prediction_batch(
                    session=session,
                    model_run_id=model_run.model_run_id,
                    predictions=summary.predictions,
                    workspace_id=workspace_id,
                )

                # Commit full transaction
                session.commit()
                logger.info(
                    "Successfully committed batch predictions to database",
                    extra={
                        "model_run_id": model_run.model_run_id,
                        "persisted_predictions": persisted_count,
                    },
                )

            except Exception as exc:
                session.rollback()
                logger.error(
                    "Database persistence failed. Transaction rolled back cleanly.",
                    extra={"error": str(exc)},
                )
                raise

        return summary
