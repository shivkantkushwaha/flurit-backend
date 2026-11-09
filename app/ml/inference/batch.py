"""Batch Inference Pipeline for Flurit Customer Snapshots.

Executes deterministic point-in-time model scoring, SHAP explainability factor attributions,
and K=3 customer segmentation assignments across configurable batches of validated customer records.
"""

import time
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Literal

import pandas as pd
from pydantic import ValidationError

from app.core.logging import get_logger
from app.ml.features.schema import FEATURE_NAMES
from app.ml.inference.schemas import CanonicalFeaturesPayload
from app.ml.inference.service import (
    SCORE_INTERPRETATION_NOTICE,
    ModelInferenceService,
    get_inference_service,
)

logger = get_logger("flurit.ml.batch")


@dataclass(frozen=True)
class BatchInputRecord:
    """Individual customer point-in-time record for batch inference."""

    account_id: str
    snapshot_date: str | date
    features: CanonicalFeaturesPayload | dict[str, Any]
    snapshot_id: int | None = None

    def to_features_payload(self) -> CanonicalFeaturesPayload:
        """Coerce and validate features into CanonicalFeaturesPayload."""
        if isinstance(self.features, CanonicalFeaturesPayload):
            return self.features
        return CanonicalFeaturesPayload(**self.features)


@dataclass(frozen=True)
class BatchRecordError:
    """Diagnostic detail for an individual record validation failure."""

    index: int
    account_id: str
    error_type: str
    error_message: str


@dataclass(frozen=True)
class BatchPredictionOutput:
    """Complete prediction, explanation, and segmentation payload for a processed customer record."""

    account_id: str
    snapshot_date: str
    snapshot_id: int | None
    model_name: str
    model_version: str
    model_score: float
    predicted_risk_label: int
    risk_tier: Literal["High Risk", "Moderate Risk", "Low Risk"]
    cluster_id: int
    cluster_name: str
    value_tier: str
    top_risk_increasing_factors: list[dict[str, Any]]
    top_risk_decreasing_factors: list[dict[str, Any]]
    shap_values: dict[str, float]
    plain_language_summary: str
    feature_payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert prediction output to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class BatchPipelineSummary:
    """Aggregate execution summary for a batch inference run."""

    total_records: int
    successful_records: int
    failed_records: int
    batch_count: int
    duration_seconds: float
    model_name: str
    model_version: str
    predictions: list[BatchPredictionOutput]
    errors: list[BatchRecordError]
    risk_metric_interpretation: str

    def to_dict(self) -> dict[str, Any]:
        """Convert execution summary to dictionary."""
        return {
            "total_records": self.total_records,
            "successful_records": self.successful_records,
            "failed_records": self.failed_records,
            "batch_count": self.batch_count,
            "duration_seconds": self.duration_seconds,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "predictions_count": len(self.predictions),
            "errors_count": len(self.errors),
            "risk_metric_interpretation": self.risk_metric_interpretation,
        }


def _determine_value_tier(
    mrr: float,
    high_threshold: float = 8000.0,
    mid_threshold: float = 4000.0,
) -> str:
    """Categorize commercial value tier based on contract MRR."""
    if mrr >= high_threshold:
        return "High Value"
    if mrr >= mid_threshold:
        return "Mid Value"
    return "Core Value"


class BatchInferencePipeline:
    """Deterministic batch inference engine for Flurit customer retention platform."""

    def __init__(
        self,
        service: ModelInferenceService | None = None,
        batch_size: int = 100,
        fail_fast: bool = False,
    ) -> None:
        """Initialize the batch inference pipeline.

        Args:
            service: ModelInferenceService instance. If None, uses global singleton.
            batch_size: Maximum number of customer records processed per chunk.
            fail_fast: If True, raises ValidationError on the first invalid record;
                if False, aggregates errors in the summary without silent skipping.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")

        self.service = service or get_inference_service()
        self.batch_size = batch_size
        self.fail_fast = fail_fast

    def process_records(
        self,
        records: list[BatchInputRecord] | list[dict[str, Any]],
    ) -> BatchPipelineSummary:
        """Process a collection of customer records through the inference pipeline.

        Args:
            records: List of BatchInputRecord objects or raw dictionaries with
                'account_id', 'snapshot_date', and canonical feature fields.

        Returns:
            BatchPipelineSummary containing predictions, errors, counts, and metadata.

        Raises:
            ValueError: If fail_fast is enabled and a record fails validation.
            RuntimeError: If the inference service artifacts are not loaded.
        """
        self.service._ensure_loaded()
        assert self.service.pipeline is not None
        assert self.service.explainer is not None
        assert self.service.segmenter is not None

        start_time = time.time()
        total_records = len(records)
        predictions: list[BatchPredictionOutput] = []
        errors: list[BatchRecordError] = []

        logger.info(
            "Starting batch inference execution",
            extra={
                "total_records": total_records,
                "batch_size": self.batch_size,
                "model_version": self.service._model_version,
            },
        )

        # 1. Normalize and coerce inputs into BatchInputRecord objects
        normalized_records: list[BatchInputRecord] = []
        for idx, item in enumerate(records):
            if isinstance(item, BatchInputRecord):
                normalized_records.append(item)
            elif isinstance(item, dict):
                account_id = str(item.get("account_id", "")).strip()
                snap_date = item.get("snapshot_date", "latest")
                snap_id = item.get("snapshot_id")

                # If features are nested or flat in item
                if "features" in item and isinstance(
                    item["features"], (dict, CanonicalFeaturesPayload)
                ):
                    feat_data = item["features"]
                else:
                    feat_data = {
                        k: v
                        for k, v in item.items()
                        if k not in ("account_id", "snapshot_date", "snapshot_id")
                    }

                normalized_records.append(
                    BatchInputRecord(
                        account_id=account_id,
                        snapshot_date=snap_date,
                        snapshot_id=snap_id,
                        features=feat_data,
                    )
                )
            else:
                err = BatchRecordError(
                    index=idx,
                    account_id="unknown",
                    error_type="TypeError",
                    error_message=f"Expected BatchInputRecord or dict, got {type(item).__name__}",
                )
                if self.fail_fast:
                    raise ValueError(err.error_message)
                errors.append(err)

        # 2. Process records in configured batch sizes
        batch_count = 0
        for chunk_start in range(0, len(normalized_records), self.batch_size):
            chunk = normalized_records[chunk_start : chunk_start + self.batch_size]
            batch_count += 1

            valid_chunk_records: list[BatchInputRecord] = []
            valid_chunk_payloads: list[CanonicalFeaturesPayload] = []

            # Validate each record in chunk strictly
            for idx_offset, record in enumerate(chunk):
                global_idx = chunk_start + idx_offset

                if not record.account_id:
                    err = BatchRecordError(
                        index=global_idx,
                        account_id="missing",
                        error_type="MissingAccountId",
                        error_message="account_id must be a non-empty string",
                    )
                    if self.fail_fast:
                        raise ValueError(err.error_message)
                    errors.append(err)
                    continue

                try:
                    features_payload = record.to_features_payload()
                    valid_chunk_records.append(record)
                    valid_chunk_payloads.append(features_payload)
                except (ValidationError, ValueError, TypeError) as exc:
                    err = BatchRecordError(
                        index=global_idx,
                        account_id=record.account_id,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                    if self.fail_fast:
                        raise ValueError(
                            f"Validation error on record {global_idx} (account '{record.account_id}'): {exc}"
                        ) from exc
                    errors.append(err)

            if not valid_chunk_payloads:
                continue

            # 3. Batch Scoring
            feature_dicts = [p.to_canonical_dict() for p in valid_chunk_payloads]
            chunk_df = pd.DataFrame(feature_dicts)[FEATURE_NAMES]

            # Vectorized scoring
            probabilities = self.service.pipeline.predict_proba(chunk_df)[:, 1]
            labels = (probabilities >= self.service._decision_threshold).astype(int)

            # Vectorized segmentation
            cluster_assignments = self.service.segmenter.predict_segments(chunk_df)

            # 4. Generate local explanations and assemble outputs
            for i, rec in enumerate(valid_chunk_records):
                prob = float(probabilities[i])
                label = int(labels[i])
                cluster_id = int(cluster_assignments[i])
                feat_dict = feature_dicts[i]

                # Risk tier
                if prob >= 0.50:
                    risk_tier: Literal["High Risk", "Moderate Risk", "Low Risk"] = "High Risk"
                elif prob >= 0.35:
                    risk_tier = "Moderate Risk"
                else:
                    risk_tier = "Low Risk"

                # Cluster profile lookup
                matching_profile = next(
                    (
                        p
                        for p in self.service.segmenter.cluster_profiles
                        if p.cluster_id == cluster_id
                    ),
                    None,
                )
                cluster_name = (
                    matching_profile.cluster_name if matching_profile else f"Cluster {cluster_id}"
                )

                # Value tier
                mrr = float(feat_dict.get("sub_current_mrr", 0.0))
                value_tier = _determine_value_tier(mrr)

                # Date formatting
                snap_date_str = (
                    rec.snapshot_date.isoformat()
                    if isinstance(rec.snapshot_date, (date, datetime))
                    else str(rec.snapshot_date)
                )

                # SHAP local explanation
                exp = self.service.explainer.explain_instance(
                    account_id=rec.account_id,
                    snapshot_date=snap_date_str,
                    feature_row=feat_dict,
                    top_k=5,
                )

                predictions.append(
                    BatchPredictionOutput(
                        account_id=rec.account_id,
                        snapshot_date=snap_date_str,
                        snapshot_id=rec.snapshot_id,
                        model_name=self.service._model_name,
                        model_version=self.service._model_version,
                        model_score=round(prob, 4),
                        predicted_risk_label=label,
                        risk_tier=risk_tier,
                        cluster_id=cluster_id,
                        cluster_name=cluster_name,
                        value_tier=value_tier,
                        top_risk_increasing_factors=exp.top_risk_increasing_factors,
                        top_risk_decreasing_factors=exp.top_risk_decreasing_factors,
                        shap_values=exp.feature_attributions,
                        plain_language_summary=exp.plain_language_summary,
                        feature_payload=feat_dict,
                    )
                )

            logger.info(
                f"Processed batch chunk {batch_count}",
                extra={
                    "chunk_size": len(valid_chunk_records),
                    "accumulated_predictions": len(predictions),
                },
            )

        duration = round(time.time() - start_time, 4)
        logger.info(
            "Batch inference execution completed",
            extra={
                "total_records": total_records,
                "successful_records": len(predictions),
                "failed_records": len(errors),
                "duration_seconds": duration,
            },
        )

        return BatchPipelineSummary(
            total_records=total_records,
            successful_records=len(predictions),
            failed_records=len(errors),
            batch_count=batch_count,
            duration_seconds=duration,
            model_name=self.service._model_name,
            model_version=self.service._model_version,
            predictions=predictions,
            errors=errors,
            risk_metric_interpretation=SCORE_INTERPRETATION_NOTICE,
        )
