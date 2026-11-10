"""Custom Dataset Processing & End-to-End Analysis Engine for Flurit.

Orchestrates the complete execution pipeline for customer-onboarded datasets:
  1. Relational normalization & referential validation.
  2. Point-in-time customer snapshotting & leak-free 60d target labeling.
  3. Canonical 45-feature extraction across all 5 domains.
  4. Model scoring, SHAP explainability, segmentation, and prioritization.
Enforces honest compatibility boundaries without fabricating customer data.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from app.core.logging import get_logger
from app.ml.features.pipeline import FeaturePipeline
from app.ml.inference.batch import BatchInferencePipeline, BatchInputRecord, BatchPipelineSummary
from app.ml.inference.schemas import PrioritizationQuadrantsRequest, ScoredCustomerRecord
from app.ml.inference.service import SCORE_INTERPRETATION_NOTICE, get_inference_service
from app.ml.snapshots.engine import SnapshotEngine
from app.onboarding.schemas import (
    EndToEndAnalysisResponse,
    PrioritizationQuadrantSummary,
    ProcessingOptions,
    ProcessingPreviewResponse,
    ScoringSummary,
    SnapshotGenerationSummary,
)
from app.onboarding.transformer import CanonicalIntermediateDataset

logger = get_logger("flurit.onboarding.processor")


@dataclass
class ProcessingExecution:
    """Internal artifacts from one processing run used for durable persistence.

    This object is deliberately not an API schema. The public response remains an
    aggregate report while the service layer can atomically persist the underlying
    canonical records and successful model outputs.
    """

    response: EndToEndAnalysisResponse
    snapshots_df: pd.DataFrame | None
    feature_matrix: pd.DataFrame | None
    batch_summary: BatchPipelineSummary | None
    model_metadata: dict[str, object] | None = None


class DatasetProcessingEngine:
    """Executes validated end-to-end processing from canonical datasets to ML insights."""

    def __init__(self) -> None:
        self.inference_service = get_inference_service()

    def preview(
        self,
        dataset: CanonicalIntermediateDataset,
        options: ProcessingOptions | None = None,
    ) -> ProcessingPreviewResponse:
        """Dry-run execution preview assessing feasibility without running heavy ML."""
        opts = options or ProcessingOptions()
        analysis_ready, label_ready, scoring_ready, blockers, warnings, next_steps = (
            self._evaluate_readiness_boundaries(dataset, opts)
        )

        total_input_rows = (
            len(dataset.accounts_df)
            + len(dataset.subscriptions_df)
            + len(dataset.usage_df)
            + len(dataset.tickets_df)
            + len(dataset.churn_events_df)
            + dataset.rejection_summary.total_rejected
        )

        potential_obs_dates: list[str] = []
        if label_ready:
            dates = self._determine_observation_dates(dataset, opts)
            potential_obs_dates = [d.isoformat() for d in dates]

        estimated_eligible = len(dataset.accounts_df)

        return ProcessingPreviewResponse(
            analysis_ready=analysis_ready,
            label_construction_ready=label_ready,
            existing_model_scoring_ready=scoring_ready,
            total_input_rows=total_input_rows,
            rejection_summary=dataset.rejection_summary,
            scoring_blockers=blockers,
            estimated_eligible_accounts=estimated_eligible,
            potential_observation_dates=potential_obs_dates,
            warnings=dataset.warnings + warnings,
            next_steps=next_steps,
        )

    def process(
        self,
        dataset: CanonicalIntermediateDataset,
        options: ProcessingOptions | None = None,
    ) -> EndToEndAnalysisResponse:
        """Run full end-to-end dataset transformation, snapshotting, and model analysis."""
        return self.process_with_artifacts(dataset, options).response

    def process_with_artifacts(
        self,
        dataset: CanonicalIntermediateDataset,
        options: ProcessingOptions | None = None,
    ) -> ProcessingExecution:
        """Run processing and retain internal artifacts for persistence by the service layer."""
        opts = options or ProcessingOptions()

        analysis_ready, label_ready, scoring_ready, blockers, warnings, next_steps = (
            self._evaluate_readiness_boundaries(dataset, opts)
        )
        all_warnings = list(dataset.warnings) + warnings

        if not analysis_ready:
            return ProcessingExecution(
                response=EndToEndAnalysisResponse(
                    processing_status="FAILED",
                    analysis_ready=False,
                    label_construction_ready=False,
                    existing_model_scoring_ready=False,
                    scoring_blockers=blockers,
                    rejection_summary=dataset.rejection_summary,
                    snapshot_summary=None,
                    scoring_summary=None,
                    warnings=all_warnings,
                    next_steps=next_steps,
                ),
                snapshots_df=None,
                feature_matrix=None,
                batch_summary=None,
            )

        # ----------------------------------------------------------------------
        # 1. Point-in-time Snapshot Generation & Target Labeling
        # ----------------------------------------------------------------------
        snapshot_summary: SnapshotGenerationSummary | None = None
        snapshots_df: pd.DataFrame | None = None

        if label_ready:
            obs_dates = self._determine_observation_dates(dataset, opts)
            if obs_dates:
                engine = SnapshotEngine(
                    accounts_df=dataset.accounts_df,
                    subscriptions_df=dataset.subscriptions_df,
                    churn_events_df=dataset.churn_events_df,
                    prediction_window_days=opts.prediction_window_days,
                    eligibility_policy="subscription_active",
                    global_max_date=dataset.max_date,
                )
                snapshots_df = engine.generate_monthly_snapshots(observation_dates=obs_dates)

                if not snapshots_df.empty:
                    snapshots_df["workspace_id"] = dataset.workspace_id
                    pos = int((snapshots_df["target_churn_60d"] == 1).sum())
                    neg = int((snapshots_df["target_churn_60d"] == 0).sum())
                    censored = int(snapshots_df["target_churn_60d"].isnull().sum())
                    uncensored = pos + neg
                    rate = round((pos / uncensored * 100.0), 2) if uncensored > 0 else None

                    snapshot_summary = SnapshotGenerationSummary(
                        total_snapshots=len(snapshots_df),
                        observation_dates_count=len(obs_dates),
                        observation_dates=[d.isoformat() for d in obs_dates],
                        eligible_accounts_count=int(snapshots_df["account_id"].nunique()),
                        positive_targets_count=pos,
                        negative_targets_count=neg,
                        censored_targets_count=censored,
                        churn_rate_pct=rate,
                    )
        else:
            all_warnings.append(
                "Historical target labeling bypassed: insufficient temporal observation span or missing churn ground truth."
            )

        # If snapshots could not be generated from monthly cohorts, fallback to scoring current active accounts at latest date
        if snapshots_df is None or snapshots_df.empty:
            if dataset.max_date:
                active_subs = dataset.subscriptions_df[
                    dataset.subscriptions_df["end_date"].isnull()
                    | (
                        pd.to_datetime(dataset.subscriptions_df["end_date"])
                        > pd.to_datetime(dataset.max_date)
                    )
                ]
                active_accs = sorted(active_subs["account_id"].unique())
                if active_accs:
                    snapshots_df = pd.DataFrame(
                        {
                            "workspace_id": dataset.workspace_id,
                            "account_id": active_accs,
                            "snapshot_date": dataset.max_date,
                            "is_active": True,
                            "target_churn_60d": None,
                        }
                    )

        # ----------------------------------------------------------------------
        # 2. Canonical Feature Matrix Construction
        # ----------------------------------------------------------------------
        feature_matrix: pd.DataFrame | None = None
        if snapshots_df is not None and not snapshots_df.empty:
            pipeline = FeaturePipeline(
                accounts_df=dataset.accounts_df,
                subscriptions_df=dataset.subscriptions_df,
                usage_df=dataset.usage_df,
                tickets_df=dataset.tickets_df,
                churn_events_df=dataset.churn_events_df,
            )
            feature_matrix = pipeline.build_feature_matrix(snapshots_df, include_metadata=True)

        # ----------------------------------------------------------------------
        # 3. Model Scoring, Explainability, and Portfolio Prioritization
        # ----------------------------------------------------------------------
        scoring_summary: ScoringSummary | None = None
        batch_summary: BatchPipelineSummary | None = None
        can_score = (scoring_ready or opts.allow_sentinel_scoring) and (
            feature_matrix is not None and not feature_matrix.empty
        )

        if can_score and feature_matrix is not None:
            try:
                scoring_summary, batch_summary = self._execute_scoring_and_prioritization(
                    feature_matrix=feature_matrix,
                    opts=opts,
                )
                status = "SUCCESS" if scoring_ready else "PARTIAL_SUCCESS"
            except Exception:
                logger.exception("Model inference failed after successful dataset transformation")
                all_warnings.append(
                    "Model inference failed. Customer source records and feature artifacts were persisted, but no predictions were written."
                )
                blockers.append("Model inference could not complete for this run.")
                next_steps.append("Retry processing after the model inference service is available.")
                status = "INFERENCE_FAILED"
        elif not scoring_ready:
            # Transformation/snapshotting completed, but model scoring was correctly withheld.
            # Preserve PARTIAL_SUCCESS for callers that received usable artifacts; use NOT_READY
            # only where label construction also could not produce a completed analysis.
            status = "PARTIAL_SUCCESS" if label_ready else "NOT_READY"
        else:
            status = "FAILED"

        return ProcessingExecution(
            response=EndToEndAnalysisResponse(
                processing_status=status,
                analysis_ready=analysis_ready,
                label_construction_ready=label_ready,
                existing_model_scoring_ready=scoring_ready,
                scoring_blockers=blockers,
                rejection_summary=dataset.rejection_summary,
                snapshot_summary=snapshot_summary,
                scoring_summary=scoring_summary,
                warnings=all_warnings,
                next_steps=next_steps,
            ),
            snapshots_df=snapshots_df,
            feature_matrix=feature_matrix,
            batch_summary=batch_summary,
            model_metadata=self.inference_service.metadata if batch_summary else None,
        )

    def _evaluate_readiness_boundaries(
        self,
        dataset: CanonicalIntermediateDataset,
        opts: ProcessingOptions,
    ) -> tuple[bool, bool, bool, list[str], list[str], list[str]]:
        """Evaluate analysis_ready, label_construction_ready, and existing_model_scoring_ready."""
        blockers: list[str] = []
        warnings: list[str] = []
        next_steps: list[str] = []

        # 1. Analysis Ready
        has_accounts = not dataset.accounts_df.empty
        has_subscriptions = not dataset.subscriptions_df.empty
        analysis_ready = has_accounts and has_subscriptions

        if not analysis_ready:
            blockers.append("Mandatory accounts or subscriptions dataset is missing or empty.")
            next_steps.append("Upload valid accounts and subscriptions CSV files.")
            return False, False, False, blockers, warnings, next_steps

        # 2. Label Construction Ready
        has_churn_truth = (
            not dataset.churn_events_df.empty or dataset.derived_churn_from_subscriptions
        )
        has_sufficient_span = dataset.span_days >= opts.prediction_window_days

        label_ready = has_churn_truth and has_sufficient_span

        if not has_churn_truth:
            blockers.append(
                "No churn ground truth available. Cannot label 60d future churn targets without churn events or subscription end dates."
            )
        elif not has_sufficient_span:
            blockers.append(
                f"Dataset temporal span ({dataset.span_days} days) is less than required prediction window ({opts.prediction_window_days} days)."
            )

        # 3. Existing Model Scoring Ready (Semantic 45-feature compatibility)
        has_usage = not dataset.usage_df.empty
        has_tickets = not dataset.tickets_df.empty
        has_churn = not dataset.churn_events_df.empty

        # Model requires real feature signals from all domains to score without distribution shift
        missing_domains: list[str] = []
        if not has_usage:
            missing_domains.append("feature_usage (14 product telemetry features)")
        if not has_tickets:
            missing_domains.append("support_tickets (11 customer service features)")
        if not has_churn:
            missing_domains.append("churn_events (4 historical churn recency/frequency features)")

        if missing_domains:
            scoring_ready = False
            blockers.append(
                f"Frozen champion model requires all 45 canonical features. Missing domain telemetry: {'; '.join(missing_domains)}. "
                "Direct inference scoring is blocked to prevent uncalibrated predictions from feature distribution shift."
            )
            if not opts.allow_sentinel_scoring:
                next_steps.append(
                    "Upload product usage and support ticket logs to unlock model scoring, or enable 'allow_sentinel_scoring' to force scoring with default sentinels."
                )
            else:
                warnings.append(
                    "Scoring with frozen model enabled via 'allow_sentinel_scoring' override. Predictions will exhibit distribution shift due to imputed zero sentinels."
                )
        else:
            scoring_ready = True

        if scoring_ready and label_ready:
            next_steps.append(
                "Review portfolio prioritization quadrants and intervention recommendations."
            )

        return analysis_ready, label_ready, scoring_ready, blockers, warnings, next_steps

    def _determine_observation_dates(
        self,
        dataset: CanonicalIntermediateDataset,
        opts: ProcessingOptions,
    ) -> list[date]:
        """Compute valid historical observation dates strictly bounded by the cutoff date."""
        if opts.observation_dates:
            return [
                datetime.fromisoformat(d).date() if isinstance(d, str) else d
                for d in opts.observation_dates
            ]

        if not dataset.min_date or not dataset.max_date:
            return []

        # Observation cutoff: T_obs <= T_max - prediction_window_days
        cutoff_date = dataset.max_date - timedelta(days=opts.prediction_window_days)
        if cutoff_date < dataset.min_date:
            return []

        # Generate monthly dates starting from min_date + 30 days
        first_obs = dataset.min_date + timedelta(days=30)
        # Snap first_obs to start of next month
        first_obs_month = (
            date(first_obs.year, first_obs.month + 1, 1)
            if first_obs.month < 12
            else date(first_obs.year + 1, 1, 1)
        )

        if first_obs_month > cutoff_date:
            return [cutoff_date]

        ts_range = pd.date_range(start=first_obs_month, end=cutoff_date, freq="MS")
        dates = [ts.date() for ts in ts_range]
        if not dates:
            dates = [cutoff_date]

        return dates

    def _execute_scoring_and_prioritization(
        self,
        feature_matrix: pd.DataFrame,
        opts: ProcessingOptions,
    ) -> tuple[ScoringSummary, BatchPipelineSummary]:
        """Run batch scoring, SHAP explainability, and portfolio prioritization."""
        self.inference_service._ensure_loaded()
        assert self.inference_service.segmenter is not None

        # Filter to latest cohort if requested
        if opts.score_latest_only and "snapshot_date" in feature_matrix.columns:
            latest_dt = feature_matrix["snapshot_date"].max()
            records_df = feature_matrix[feature_matrix["snapshot_date"] == latest_dt].copy()
        else:
            records_df = feature_matrix.copy()

        batch_pipeline = BatchInferencePipeline(
            service=self.inference_service,
            batch_size=500,
            fail_fast=False,
        )

        # Convert feature matrix rows into batch inputs
        batch_inputs: list[BatchInputRecord] = []
        for _, row in records_df.iterrows():
            acc_id = str(row["account_id"])
            snap_dt = row.get("snapshot_date", "latest")
            features_dict = {
                k: v
                for k, v in row.items()
                if k not in ("account_id", "snapshot_date", "target_churn_60d", "workspace_id")
            }
            batch_inputs.append(
                BatchInputRecord(
                    account_id=acc_id,
                    snapshot_date=snap_dt,
                    features=features_dict,
                )
            )

        batch_summary = batch_pipeline.process_records(batch_inputs)
        predictions = batch_summary.predictions

        if not predictions:
            raise RuntimeError("Batch scoring produced zero predictions.")

        scores = [p.model_score for p in predictions]
        avg_score = float(round(np.mean(scores), 4))

        # Risk tier counts
        high_count = sum(1 for p in predictions if p.risk_tier == "High Risk")
        mod_count = sum(1 for p in predictions if p.risk_tier == "Moderate Risk")
        low_count = sum(1 for p in predictions if p.risk_tier == "Low Risk")

        # Cluster distribution
        cluster_counts: dict[str, int] = {}
        for p in predictions:
            cluster_counts[p.cluster_name] = cluster_counts.get(p.cluster_name, 0) + 1

        # Global risk factors (aggregate SHAP values from top increasing factors)
        factor_impacts: dict[str, float] = {}
        for p in predictions:
            for f in p.top_risk_increasing_factors:
                feat = f["feature_name"]
                factor_impacts[feat] = factor_impacts.get(feat, 0.0) + abs(f["attribution"])

        top_increasing = [
            {"feature_name": k, "mean_absolute_shap": round(v / len(predictions), 4)}
            for k, v in sorted(factor_impacts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

        # Prioritization quadrants
        prioritize_records: list[ScoredCustomerRecord] = []
        latest_by_account = {}
        for p in predictions:
            previous = latest_by_account.get(p.account_id)
            if previous is None or p.snapshot_date > previous.snapshot_date:
                latest_by_account[p.account_id] = p
        for p in latest_by_account.values():
            prioritize_records.append(
                ScoredCustomerRecord(
                    account_id=p.account_id,
                    sub_current_mrr=float(p.feature_payload["sub_current_mrr"]),
                    model_score=p.model_score,
                )
            )

        quad_resp = self.inference_service.prioritize_quadrants(
            PrioritizationQuadrantsRequest(records=prioritize_records)
        )

        quadrants_summary = [
            PrioritizationQuadrantSummary(
                quadrant_name=q.quadrant_name,
                priority_level=q.priority_level,
                account_count=q.account_count,
                account_percentage=q.account_percentage,
                total_mrr=q.total_mrr,
                estimated_revenue_at_risk_mrr=q.estimated_revenue_at_risk_mrr,
                action_recommendation=q.action_recommendation,
            )
            for q in quad_resp.quadrants
        ]

        return ScoringSummary(
            scored_snapshots_count=len(predictions),
            model_name=self.inference_service._model_name,
            model_version=self.inference_service._model_version,
            average_model_score=avg_score,
            high_risk_count=high_count,
            moderate_risk_count=mod_count,
            low_risk_count=low_count,
            segments_distribution=cluster_counts,
            prioritization_quadrants=quadrants_summary,
            top_global_risk_increasing_factors=top_increasing,
            score_interpretation_notice=SCORE_INTERPRETATION_NOTICE,
        ), batch_summary
