"""SQLAlchemy Repository for Machine Learning Predictions, Explanations, and Customer Segments.

Provides transactional, idempotent persistence for model runs, inference outputs,
SHAP factor attributions, and unsupervised cluster assignments.
"""

from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.ml.inference.batch import BatchPredictionOutput
from app.models.account import Account
from app.models.ml import (
    CustomerExplanation,
    CustomerPrediction,
    CustomerSegment,
    ModelRun,
)

logger = get_logger("flurit.repositories.prediction")


class PredictionRepository:
    """Repository handling database operations for ML predictions, explanations, and segments."""

    def ensure_accounts_exist(
        self,
        session: Session,
        account_ids: list[str] | set[str],
        workspace_id: str = "ws_default",
        default_signup_date: date | None = None,
    ) -> int:
        """Ensure all required account records exist in accounts table to satisfy foreign keys.

        Args:
            session: Active SQLAlchemy database session.
            account_ids: Collection of account IDs to verify/create.
            workspace_id: Tenant workspace identifier.
            default_signup_date: Signup date to assign to newly created stub accounts.

        Returns:
            Count of newly created accounts.
        """
        unique_ids = {str(aid).strip() for aid in account_ids if str(aid).strip()}
        if not unique_ids:
            return 0

        existing_stmt = select(Account.account_id).where(
            Account.workspace_id == workspace_id,
            Account.account_id.in_(unique_ids),
        )
        existing_ids = set(session.scalars(existing_stmt).all())
        missing_ids = unique_ids - existing_ids

        if not missing_ids:
            return 0

        created_count = 0
        ref_date = default_signup_date or date.today()
        for aid in sorted(missing_ids):
            stub_account = Account(
                workspace_id=workspace_id,
                account_id=aid,
                account_name=f"Account {aid}",
                industry="unknown",
                country="UNKNOWN",
                signup_date=ref_date,
                referral_source="unknown",
                plan_tier="standard",
                seats=1,
                is_trial=False,
                churn_flag=False,
            )
            session.add(stub_account)
            created_count += 1

        session.flush()
        logger.info(
            "Created stub account records to satisfy foreign key integrity",
            extra={"created_count": created_count, "workspace_id": workspace_id},
        )
        return created_count

    def get_or_create_model_run(
        self,
        session: Session,
        model_name: str,
        version: str,
        metadata: dict[str, Any] | None = None,
    ) -> ModelRun:
        """Retrieve existing ModelRun by name and version, or insert a new record.

        Args:
            session: Active database session.
            model_name: Canonical model name (e.g. 'flurit_churn_predictor').
            version: Model version identifier (e.g. '1.0.0').
            metadata: Optional dictionary with hyperparameters, metrics, and threshold.

        Returns:
            Persisted ModelRun instance.
        """
        stmt = (
            select(ModelRun)
            .where(ModelRun.model_name == model_name, ModelRun.version == version)
            .order_by(desc(ModelRun.trained_at))
        )
        existing = session.scalars(stmt).first()
        if existing is not None:
            return existing

        meta = metadata or {}
        model_run = ModelRun(
            model_name=model_name,
            version=version,
            is_active=True,
            decision_threshold=float(meta.get("decision_threshold", 0.50)),
            hyperparameters=meta.get("hyperparameters"),
            metrics=meta.get("metrics"),
            feature_names=meta.get("feature_names"),
            artifact_path=str(meta.get("artifact_path", "")),
            status="completed",
        )
        session.add(model_run)
        session.flush()
        logger.info(
            "Registered new ModelRun in database",
            extra={
                "model_run_id": model_run.model_run_id,
                "model_name": model_name,
                "version": version,
            },
        )
        return model_run

    def get_active_model_run(self, session: Session) -> ModelRun | None:
        """Retrieve the latest active ModelRun."""
        stmt = (
            select(ModelRun).where(ModelRun.is_active.is_(True)).order_by(desc(ModelRun.trained_at))
        )
        return session.scalars(stmt).first()

    def save_prediction_batch(
        self,
        session: Session,
        model_run_id: int,
        predictions: list[BatchPredictionOutput],
        workspace_id: str = "ws_default",
    ) -> int:
        """Persist a batch of predictions, explanations, and segments idempotently scoped to a workspace.

        Uses savepoint nested transactions to ensure atomic execution. If any record
        fails, the batch rolls back completely to maintain database consistency.

        Args:
            session: Active SQLAlchemy database session.
            model_run_id: Foreign key ID of the active ModelRun.
            predictions: List of BatchPredictionOutput items to persist.
            workspace_id: Tenant workspace identifier.

        Returns:
            Number of prediction records saved/updated.

        Raises:
            Exception: If any database error occurs during batch persistence.
        """
        if not predictions:
            return 0

        # Ensure all referenced accounts exist in this workspace before inserting predictions
        account_ids = [p.account_id for p in predictions]
        self.ensure_accounts_exist(session, account_ids, workspace_id=workspace_id)

        saved_count = 0
        now_dt = datetime.now(UTC).replace(tzinfo=None)

        try:
            with session.begin_nested():
                scope = (
                    CustomerPrediction.workspace_id == workspace_id,
                    CustomerPrediction.model_run_id == model_run_id,
                    CustomerPrediction.account_id.in_(set(account_ids)),
                )
                stored = {
                    (row.account_id, row.snapshot_id): row
                    for row in session.scalars(select(CustomerPrediction).where(*scope))
                }
                explanations = {
                    row.prediction_id: row
                    for row in session.scalars(
                        select(CustomerExplanation)
                        .join(CustomerPrediction, CustomerExplanation.prediction_id == CustomerPrediction.prediction_id)
                        .where(CustomerExplanation.workspace_id == workspace_id, *scope)
                    )
                }
                segments = {
                    row.account_id: row
                    for row in session.scalars(
                        select(CustomerSegment).where(
                            CustomerSegment.workspace_id == workspace_id,
                            CustomerSegment.model_run_id == model_run_id,
                            CustomerSegment.account_id.in_(set(account_ids)),
                        )
                    )
                }

                for output in predictions:
                    key = (output.account_id, output.snapshot_id)
                    record = stored.get(key)
                    if record is None:
                        record = CustomerPrediction(
                            workspace_id=workspace_id,
                            model_run_id=model_run_id,
                            account_id=output.account_id,
                            snapshot_id=output.snapshot_id,
                        )
                        session.add(record)
                        stored[key] = record
                    record.churn_probability = output.model_score
                    record.risk_level = output.risk_tier
                    record.predicted_at = now_dt

                # Allocate IDs together before writing dependent explanations.
                session.flush()
                for output in predictions:
                    record = stored[(output.account_id, output.snapshot_id)]
                    explanation = explanations.get(record.prediction_id)
                    if explanation is None:
                        explanation = CustomerExplanation(
                            workspace_id=workspace_id,
                            prediction_id=record.prediction_id,
                            account_id=output.account_id,
                        )
                        session.add(explanation)
                        explanations[record.prediction_id] = explanation
                    explanation.top_risk_factors = output.top_risk_increasing_factors
                    explanation.shap_values = output.shap_values
                    explanation.summary_text = output.plain_language_summary
                    explanation.computed_at = now_dt

                    segment = segments.get(output.account_id)
                    if segment is None:
                        segment = CustomerSegment(
                            workspace_id=workspace_id,
                            model_run_id=model_run_id,
                            account_id=output.account_id,
                        )
                        session.add(segment)
                        segments[output.account_id] = segment
                    segment.cluster_id = output.cluster_id
                    segment.cluster_name = output.cluster_name
                    segment.risk_tier = output.risk_tier
                    segment.value_tier = output.value_tier
                    segment.created_at = now_dt
                    saved_count += 1

                session.flush()

            logger.info(
                "Batch predictions, explanations, and segments persisted successfully",
                extra={
                    "saved_count": saved_count,
                    "model_run_id": model_run_id,
                    "workspace_id": workspace_id,
                },
            )
            return saved_count

        except Exception as exc:
            logger.error(
                "Failed to persist prediction batch",
                extra={"error": str(exc), "workspace_id": workspace_id},
            )
            raise

    def get_latest_prediction_by_account(
        self,
        session: Session,
        account_id: str,
        workspace_id: str = "ws_default",
    ) -> CustomerPrediction | None:
        """Retrieve the most recent prediction for a given account within a workspace."""
        stmt = (
            select(CustomerPrediction)
            .where(
                CustomerPrediction.workspace_id == workspace_id,
                CustomerPrediction.account_id == account_id,
            )
            .order_by(desc(CustomerPrediction.predicted_at))
        )
        return session.scalars(stmt).first()

    def get_prediction_with_explanation(
        self,
        session: Session,
        prediction_id: int,
        workspace_id: str | None = None,
    ) -> CustomerPrediction | None:
        """Retrieve prediction along with its linked SHAP explanation."""
        stmt = select(CustomerPrediction).where(CustomerPrediction.prediction_id == prediction_id)
        if workspace_id is not None:
            stmt = stmt.where(CustomerPrediction.workspace_id == workspace_id)
        return session.scalars(stmt).first()

    def get_customer_segment(
        self,
        session: Session,
        account_id: str,
        model_run_id: int | None = None,
        workspace_id: str = "ws_default",
    ) -> CustomerSegment | None:
        """Retrieve customer segment assignment for an account within a workspace."""
        stmt = select(CustomerSegment).where(
            CustomerSegment.workspace_id == workspace_id,
            CustomerSegment.account_id == account_id,
        )
        if model_run_id is not None:
            stmt = stmt.where(CustomerSegment.model_run_id == model_run_id)
        stmt = stmt.order_by(desc(CustomerSegment.created_at))
        return session.scalars(stmt).first()

    def get_predictions_by_model_run(
        self,
        session: Session,
        model_run_id: int,
        workspace_id: str = "ws_default",
        limit: int = 100,
        offset: int = 0,
    ) -> list[CustomerPrediction]:
        """Query predictions generated by a specific model run within a workspace."""
        stmt = (
            select(CustomerPrediction)
            .where(
                CustomerPrediction.workspace_id == workspace_id,
                CustomerPrediction.model_run_id == model_run_id,
            )
            .order_by(desc(CustomerPrediction.predicted_at))
            .limit(limit)
            .offset(offset)
        )
        return list(session.scalars(stmt).all())
