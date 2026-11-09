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
                for output in predictions:
                    # 1. Check for existing prediction by (workspace_id, model_run_id, account_id, snapshot_id)
                    pred_stmt = select(CustomerPrediction).where(
                        CustomerPrediction.workspace_id == workspace_id,
                        CustomerPrediction.model_run_id == model_run_id,
                        CustomerPrediction.account_id == output.account_id,
                    )
                    if output.snapshot_id is not None:
                        pred_stmt = pred_stmt.where(
                            CustomerPrediction.snapshot_id == output.snapshot_id
                        )
                    else:
                        pred_stmt = pred_stmt.where(CustomerPrediction.snapshot_id.is_(None))

                    existing_pred = session.scalars(pred_stmt).first()

                    if existing_pred is not None:
                        # Idempotent update
                        existing_pred.churn_probability = output.model_score
                        existing_pred.risk_level = output.risk_tier
                        existing_pred.predicted_at = now_dt
                        pred_record = existing_pred
                    else:
                        # Insert new prediction
                        pred_record = CustomerPrediction(
                            workspace_id=workspace_id,
                            model_run_id=model_run_id,
                            account_id=output.account_id,
                            snapshot_id=output.snapshot_id,
                            churn_probability=output.model_score,
                            risk_level=output.risk_tier,
                            predicted_at=now_dt,
                        )
                        session.add(pred_record)
                        session.flush()

                    # 2. Upsert CustomerExplanation linked to prediction
                    exp_stmt = select(CustomerExplanation).where(
                        CustomerExplanation.workspace_id == workspace_id,
                        CustomerExplanation.prediction_id == pred_record.prediction_id,
                    )
                    existing_exp = session.scalars(exp_stmt).first()

                    if existing_exp is not None:
                        existing_exp.top_risk_factors = output.top_risk_increasing_factors
                        existing_exp.shap_values = output.shap_values
                        existing_exp.summary_text = output.plain_language_summary
                        existing_exp.computed_at = now_dt
                    else:
                        exp_record = CustomerExplanation(
                            workspace_id=workspace_id,
                            prediction_id=pred_record.prediction_id,
                            account_id=output.account_id,
                            top_risk_factors=output.top_risk_increasing_factors,
                            shap_values=output.shap_values,
                            summary_text=output.plain_language_summary,
                            computed_at=now_dt,
                        )
                        session.add(exp_record)

                    # 3. Upsert CustomerSegment for (workspace_id, model_run_id, account_id)
                    seg_stmt = select(CustomerSegment).where(
                        CustomerSegment.workspace_id == workspace_id,
                        CustomerSegment.model_run_id == model_run_id,
                        CustomerSegment.account_id == output.account_id,
                    )
                    existing_seg = session.scalars(seg_stmt).first()

                    if existing_seg is not None:
                        existing_seg.cluster_id = output.cluster_id
                        existing_seg.cluster_name = output.cluster_name
                        existing_seg.risk_tier = output.risk_tier
                        existing_seg.value_tier = output.value_tier
                        existing_seg.created_at = now_dt
                    else:
                        seg_record = CustomerSegment(
                            workspace_id=workspace_id,
                            model_run_id=model_run_id,
                            account_id=output.account_id,
                            cluster_id=output.cluster_id,
                            cluster_name=output.cluster_name,
                            risk_tier=output.risk_tier,
                            value_tier=output.value_tier,
                            created_at=now_dt,
                        )
                        session.add(seg_record)

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
