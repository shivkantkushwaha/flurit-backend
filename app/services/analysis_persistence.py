"""Persist completed onboarding analysis artifacts under the authorized workspace."""

from dataclasses import replace
from datetime import date, datetime
from typing import Any

import pandas as pd
from sqlalchemy import delete, insert, select
from sqlalchemy.orm import Session

from app.models.account import Account
from app.models.churn import ChurnEvent
from app.models.ml import CustomerFeature, CustomerSnapshot
from app.models.subscription import Subscription
from app.models.ticket import SupportTicket
from app.models.usage import FeatureUsage
from app.onboarding.processor import ProcessingExecution
from app.onboarding.transformer import CanonicalIntermediateDataset
from app.repositories.prediction_repository import PredictionRepository


def _date(value: Any) -> date:
    """Normalize pandas/Python dates without persisting time-of-day information."""
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def _datetime(value: Any) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).to_pydatetime().replace(tzinfo=None)


def _json_value(value: Any) -> Any:
    """Convert dataframe scalars to JSON-safe values without changing their meaning."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value.item() if hasattr(value, "item") else value


class AnalysisPersistenceService:
    """Upsert source, snapshot, feature, and inference artifacts for one workspace.

    Processing is intentionally separate from persistence: expensive feature and model
    work runs first, then this service commits the internally consistent result in one
    short database transaction. A failure rolls back all writes for that processing run.
    """

    def __init__(self, repository: PredictionRepository | None = None) -> None:
        self.repository = repository or PredictionRepository()

    def persist(
        self,
        session: Session,
        dataset: CanonicalIntermediateDataset,
        execution: ProcessingExecution,
    ) -> dict[str, int | str | None]:
        workspace_id = dataset.workspace_id
        self._upsert_accounts(session, dataset, workspace_id)
        self._upsert_subscriptions(session, dataset, workspace_id)
        self._replace_usage(session, dataset, workspace_id)
        self._upsert_tickets(session, dataset, workspace_id)
        self._upsert_churn_events(session, dataset, workspace_id)
        snapshot_ids = self._upsert_snapshots_and_features(
            session, execution, workspace_id
        )

        model_run_id: int | None = None
        persisted_predictions = 0
        summary = execution.batch_summary
        if summary and summary.predictions:
            model_run = self.repository.get_or_create_model_run(
                session=session,
                model_name=summary.model_name,
                version=summary.model_version,
                metadata=execution.model_metadata,
            )
            model_run_id = model_run.model_run_id
            predictions = []
            for prediction in summary.predictions:
                key = (prediction.account_id, _date(prediction.snapshot_date))
                snapshot_id = snapshot_ids.get(key)
                if snapshot_id is None:
                    raise ValueError(
                        f"Missing persisted snapshot for prediction {prediction.account_id} at {prediction.snapshot_date}."
                    )
                predictions.append(replace(prediction, snapshot_id=snapshot_id))
            persisted_predictions = self.repository.save_prediction_batch(
                session=session,
                model_run_id=model_run_id,
                predictions=predictions,
                workspace_id=workspace_id,
            )

        session.commit()
        return {
            "persisted_accounts": len(dataset.accounts_df),
            "persisted_subscriptions": len(dataset.subscriptions_df),
            "persisted_usage_events": len(dataset.usage_df),
            "persisted_support_tickets": len(dataset.tickets_df),
            "persisted_churn_events": len(dataset.churn_events_df),
            "persisted_snapshots": len(snapshot_ids),
            "persisted_predictions": persisted_predictions,
            "model_run_id": model_run_id,
        }

    def _upsert_accounts(
        self, session: Session, dataset: CanonicalIntermediateDataset, workspace_id: str
    ) -> None:
        ids = [str(item) for item in dataset.accounts_df["account_id"].tolist()]
        existing = {
            account.account_id: account
            for account in session.scalars(
                select(Account).where(
                    Account.workspace_id == workspace_id, Account.account_id.in_(ids)
                )
            ).all()
        }
        for _, row in dataset.accounts_df.iterrows():
            account_id = str(row["account_id"])
            values = {
                "account_name": account_id,
                "industry": str(row.get("industry", "unknown")),
                "country": str(row.get("country", "unknown")),
                "signup_date": _date(row["signup_date"]),
                "referral_source": str(row.get("referral_source", "unknown")),
                # The mapping deliberately has no plan/name field. Do not invent one.
                "plan_tier": "unknown",
                "seats": max(1, int(row.get("seats", 1))),
                "is_trial": bool(row.get("is_trial", False)),
                "churn_flag": False,
            }
            account = existing.get(account_id)
            if account:
                for key, value in values.items():
                    setattr(account, key, value)
            else:
                session.add(Account(workspace_id=workspace_id, account_id=account_id, **values))
        session.flush()

    def _upsert_subscriptions(
        self, session: Session, dataset: CanonicalIntermediateDataset, workspace_id: str
    ) -> None:
        ids = [str(item) for item in dataset.subscriptions_df["subscription_id"].tolist()]
        existing = {
            subscription.subscription_id: subscription
            for subscription in session.scalars(
                select(Subscription).where(
                    Subscription.workspace_id == workspace_id,
                    Subscription.subscription_id.in_(ids),
                )
            ).all()
        }
        for _, row in dataset.subscriptions_df.iterrows():
            sub_id = str(row["subscription_id"])
            values = {
                "account_id": str(row["account_id"]),
                "start_date": _date(row["start_date"]),
                "end_date": None if pd.isna(row.get("end_date")) else _date(row["end_date"]),
                "plan_tier": "unknown",
                "seats": max(1, int(row.get("seats", 1))),
                "mrr_amount": float(row.get("mrr_amount", 0.0)),
                "arr_amount": float(row.get("arr_amount", 0.0)),
                "is_trial": bool(row.get("is_trial", False)),
                "upgrade_flag": bool(row.get("upgrade_flag", False)),
                "downgrade_flag": bool(row.get("downgrade_flag", False)),
                "churn_flag": False,
                "billing_frequency": str(row.get("billing_frequency", "monthly")),
                "auto_renew_flag": bool(row.get("auto_renew_flag", True)),
            }
            subscription = existing.get(sub_id)
            if subscription:
                for key, value in values.items():
                    setattr(subscription, key, value)
            else:
                session.add(Subscription(workspace_id=workspace_id, subscription_id=sub_id, **values))
        session.flush()

    def _replace_usage(
        self, session: Session, dataset: CanonicalIntermediateDataset, workspace_id: str
    ) -> None:
        """Replace telemetry for submitted subscriptions so duplicate source IDs remain idempotent."""
        if dataset.usage_df.empty:
            return
        subscription_ids = [str(item) for item in dataset.usage_df["subscription_id"].unique()]
        session.execute(
            delete(FeatureUsage).where(
                FeatureUsage.workspace_id == workspace_id,
                FeatureUsage.subscription_id.in_(subscription_ids),
            )
        )
        for offset in range(0, len(dataset.usage_df), 1000):
            rows = dataset.usage_df.iloc[offset : offset + 1000].to_dict("records")
            session.execute(
                insert(FeatureUsage),
                [
                    {
                        "workspace_id": workspace_id,
                        "usage_id": str(row["usage_id"]),
                        "subscription_id": str(row["subscription_id"]),
                        "usage_date": _date(row["usage_date"]),
                        "feature_name": str(row.get("feature_name", "unknown")),
                        "usage_count": int(row.get("usage_count", 1)),
                        "usage_duration_secs": int(row.get("usage_duration_secs", 0)),
                        "error_count": int(row.get("error_count", 0)),
                        "is_beta_feature": bool(row.get("is_beta_feature", False)),
                    }
                    for row in rows
                ],
            )
        session.flush()

    def _upsert_tickets(
        self, session: Session, dataset: CanonicalIntermediateDataset, workspace_id: str
    ) -> None:
        if dataset.tickets_df.empty:
            return
        ids = [str(item) for item in dataset.tickets_df["ticket_id"].tolist()]
        existing = {
            item.ticket_id: item
            for item in session.scalars(
                select(SupportTicket).where(
                    SupportTicket.workspace_id == workspace_id,
                    SupportTicket.ticket_id.in_(ids),
                )
            ).all()
        }
        for _, row in dataset.tickets_df.iterrows():
            values = {
                "account_id": str(row["account_id"]),
                "submitted_at": _datetime(row["submitted_at"]),
                "closed_at": _datetime(row.get("closed_at")),
                "resolution_time_hours": None,
                "priority": str(row.get("priority", "unknown")),
                "first_response_time_minutes": None,
                "satisfaction_score": None if pd.isna(row.get("satisfaction_score")) else float(row["satisfaction_score"]),
                "escalation_flag": bool(row.get("escalation_flag", False)),
            }
            record = existing.get(str(row["ticket_id"]))
            if record:
                for field, value in values.items():
                    setattr(record, field, value)
            else:
                session.add(SupportTicket(workspace_id=workspace_id, ticket_id=str(row["ticket_id"]), **values))
        session.flush()

    def _upsert_churn_events(
        self, session: Session, dataset: CanonicalIntermediateDataset, workspace_id: str
    ) -> None:
        if dataset.churn_events_df.empty:
            return
        ids = [str(item) for item in dataset.churn_events_df["churn_event_id"].tolist()]
        existing = {
            item.churn_event_id: item
            for item in session.scalars(
                select(ChurnEvent).where(
                    ChurnEvent.workspace_id == workspace_id,
                    ChurnEvent.churn_event_id.in_(ids),
                )
            ).all()
        }
        for _, row in dataset.churn_events_df.iterrows():
            values = {
                "account_id": str(row["account_id"]),
                "churn_date": _date(row["churn_date"]),
                "reason_code": "unknown",
                "refund_amount_usd": float(row.get("refund_amount_usd", 0.0)),
                "preceding_upgrade_flag": False,
                "preceding_downgrade_flag": False,
                "is_reactivation": False,
                "feedback_text": None,
            }
            record = existing.get(str(row["churn_event_id"]))
            if record:
                for field, value in values.items():
                    setattr(record, field, value)
            else:
                session.add(ChurnEvent(workspace_id=workspace_id, churn_event_id=str(row["churn_event_id"]), **values))
        session.flush()

    def _upsert_snapshots_and_features(
        self, session: Session, execution: ProcessingExecution, workspace_id: str
    ) -> dict[tuple[str, date], int]:
        if execution.snapshots_df is None or execution.snapshots_df.empty:
            return {}
        snapshot_keys = [
            (str(row["account_id"]), _date(row["snapshot_date"]))
            for _, row in execution.snapshots_df.iterrows()
        ]
        account_ids = {account_id for account_id, _ in snapshot_keys}
        existing = {
            (item.account_id, item.snapshot_date): item
            for item in session.scalars(
                select(CustomerSnapshot).where(
                    CustomerSnapshot.workspace_id == workspace_id,
                    CustomerSnapshot.account_id.in_(account_ids),
                )
            ).all()
        }
        for _, row in execution.snapshots_df.iterrows():
            key = (str(row["account_id"]), _date(row["snapshot_date"]))
            snapshot = existing.get(key)
            target = row.get("target_churn_60d")
            values = {
                "is_active": bool(row.get("is_active", True)),
                "target_churn_60d": None if pd.isna(target) else int(target),
            }
            if snapshot:
                for field, value in values.items():
                    setattr(snapshot, field, value)
            else:
                snapshot = CustomerSnapshot(
                    workspace_id=workspace_id,
                    account_id=key[0],
                    snapshot_date=key[1],
                    **values,
                )
                session.add(snapshot)
                existing[key] = snapshot
        session.flush()

        feature_rows = execution.feature_matrix
        if feature_rows is not None and not feature_rows.empty:
            snapshot_id_values = [item.snapshot_id for item in existing.values()]
            features_by_snapshot = {
                item.snapshot_id: item
                for item in session.scalars(
                    select(CustomerFeature).where(CustomerFeature.snapshot_id.in_(snapshot_id_values))
                ).all()
            }
            for _, row in feature_rows.iterrows():
                key = (str(row["account_id"]), _date(row["snapshot_date"]))
                snapshot = existing[key]
                payload = {
                    key: _json_value(value)
                    for key, value in row.items()
                    if key not in {"account_id", "snapshot_date", "target_churn_60d", "workspace_id"}
                }
                values = {
                    "workspace_id": workspace_id,
                    "account_id": key[0],
                    "snapshot_date": key[1],
                    "current_mrr": float(payload.get("sub_current_mrr", 0.0)),
                    "current_arr": float(payload.get("sub_current_arr", 0.0)),
                    "tenure_days": int(payload.get("sub_tenure_days", 0)),
                    "usage_count_30d": int(payload.get("usage_count_30d", 0)),
                    "usage_duration_30d": int(payload.get("usage_duration_secs_30d", 0)),
                    "usage_change_pct": float(payload.get("usage_trend_pct_30d", 0.0)),
                    "ticket_count_30d": int(payload.get("tickets_count_30d", 0)),
                    "avg_satisfaction": float(payload.get("tickets_avg_satisfaction_90d", -1.0)),
                    "has_unrated_tickets": bool(payload.get("tickets_has_unrated_90d", 0)),
                    "feature_payload": payload,
                }
                feature = features_by_snapshot.get(snapshot.snapshot_id)
                if feature:
                    for field, value in values.items():
                        setattr(feature, field, value)
                else:
                    session.add(CustomerFeature(snapshot_id=snapshot.snapshot_id, **values))
        session.flush()
        return {key: item.snapshot_id for key, item in existing.items()}
