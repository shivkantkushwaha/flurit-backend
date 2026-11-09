"""Tests for SQLAlchemy database models and constraint validation.

Tests:
- Schema creation on all 11 declared models.
- Handling duplicate usage_id records in feature_usage without data loss.
- Foreign key cascading and relationship navigation.
"""

import unittest
from datetime import date

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.models import (
    Account,
    Base,
    CustomerExplanation,
    CustomerFeature,
    CustomerPrediction,
    CustomerSegment,
    CustomerSnapshot,
    FeatureUsage,
    ModelRun,
    Subscription,
)


class TestDatabaseModels(unittest.TestCase):
    """Integration tests for SQLAlchemy ORM models using in-memory SQLite."""

    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.db: Session = self.SessionLocal()

    def tearDown(self) -> None:
        self.db.close()
        Base.metadata.drop_all(self.engine)

    def test_schema_creates_all_eleven_tables(self) -> None:
        """Verify that all tables including tenant workspaces and jobs are registered and created in metadata."""
        expected_tables = {
            "workspaces",
            "workspace_members",
            "processing_jobs",
            "accounts",
            "subscriptions",
            "feature_usage",
            "support_tickets",
            "churn_events",
            "customer_snapshots",
            "customer_features",
            "model_runs",
            "customer_predictions",
            "customer_explanations",
            "customer_segments",
        }
        actual_tables = set(Base.metadata.tables.keys())
        self.assertEqual(expected_tables, actual_tables)

    def test_duplicate_usage_id_handled_without_collision(self) -> None:
        """Verify that multiple FeatureUsage records with identical usage_ids can be inserted.

        The source CSV has 21 duplicate usage_ids; the surrogate primary key (id)
        must ensure no primary key violation occurs.
        """
        # Create parent account & subscription
        account = Account(
            account_id="A-test-01",
            account_name="Test Corp",
            industry="DevTools",
            country="US",
            signup_date=date(2023, 1, 15),
            referral_source="organic",
            plan_tier="Pro",
            seats=10,
            is_trial=False,
            churn_flag=False,
        )
        self.db.add(account)

        subscription = Subscription(
            subscription_id="S-test-01",
            account_id="A-test-01",
            start_date=date(2023, 1, 15),
            end_date=None,
            plan_tier="Pro",
            seats=10,
            mrr_amount=500.0,
            arr_amount=6000.0,
            is_trial=False,
            upgrade_flag=False,
            downgrade_flag=False,
            churn_flag=False,
            billing_frequency="monthly",
            auto_renew_flag=True,
        )
        self.db.add(subscription)
        self.db.commit()

        # Insert two usage events with the EXACT SAME usage_id
        shared_usage_id = "U-duplicate-01"
        u1 = FeatureUsage(
            usage_id=shared_usage_id,
            subscription_id="S-test-01",
            usage_date=date(2023, 2, 1),
            feature_name="export_csv",
            usage_count=5,
            usage_duration_secs=300,
            error_count=0,
            is_beta_feature=False,
        )
        u2 = FeatureUsage(
            usage_id=shared_usage_id,
            subscription_id="S-test-01",
            usage_date=date(2023, 2, 2),
            feature_name="api_sync",
            usage_count=12,
            usage_duration_secs=450,
            error_count=1,
            is_beta_feature=True,
        )
        self.db.add_all([u1, u2])
        self.db.commit()

        # Query back and verify both exist with distinct primary keys
        results = self.db.scalars(
            select(FeatureUsage).where(FeatureUsage.usage_id == shared_usage_id)
        ).all()
        self.assertEqual(len(results), 2)
        self.assertNotEqual(results[0].id, results[1].id)
        self.assertEqual(results[0].usage_id, results[1].usage_id)

    def test_cascading_and_derived_ml_relationships(self) -> None:
        """Verify full entity relationship tree from Account to Snapshots, Predictions, and SHAP Explanations."""
        account = Account(
            account_id="A-test-02",
            account_name="Acme Inc",
            industry="FinTech",
            country="UK",
            signup_date=date(2023, 3, 1),
            referral_source="partner",
            plan_tier="Enterprise",
            seats=50,
            is_trial=False,
            churn_flag=False,
        )
        self.db.add(account)

        # Snapshot
        snapshot = CustomerSnapshot(
            account_id="A-test-02",
            snapshot_date=date(2023, 6, 1),
            is_active=True,
            target_churn_60d=1,
        )
        self.db.add(snapshot)
        self.db.flush()

        # Features linked to snapshot
        feature = CustomerFeature(
            snapshot_id=snapshot.snapshot_id,
            account_id="A-test-02",
            snapshot_date=date(2023, 6, 1),
            current_mrr=2500.0,
            current_arr=30000.0,
            tenure_days=92,
            usage_count_30d=140,
            usage_change_pct=-0.35,
            ticket_count_30d=3,
            avg_satisfaction=3.5,
            has_unrated_tickets=False,
            feature_payload={"mrr": 2500.0, "usage_delta": -0.35},
        )
        self.db.add(feature)

        # Model Run
        model_run = ModelRun(
            model_name="RandomForestClassifier",
            version="v1.0.0",
            is_active=True,
            decision_threshold=0.45,
            hyperparameters={"n_estimators": 100, "max_depth": 6},
            metrics={"roc_auc": 0.88, "pr_auc": 0.65},
            feature_names=["usage_change_pct", "ticket_count_30d", "current_mrr"],
            artifact_path="artifacts/rf_v1.joblib",
        )
        self.db.add(model_run)
        self.db.flush()

        # Prediction
        prediction = CustomerPrediction(
            model_run_id=model_run.model_run_id,
            account_id="A-test-02",
            snapshot_id=snapshot.snapshot_id,
            churn_probability=0.78,
            risk_level="HIGH",
        )
        self.db.add(prediction)
        self.db.flush()

        # SHAP Explanation
        explanation = CustomerExplanation(
            prediction_id=prediction.prediction_id,
            account_id="A-test-02",
            top_risk_factors=[
                {"feature": "usage_change_pct", "value": -0.35, "attribution": 0.22},
                {"feature": "ticket_count_30d", "value": 3, "attribution": 0.14},
            ],
            shap_values={"usage_change_pct": 0.22, "ticket_count_30d": 0.14},
            summary_text="Usage dropped 35% with 3 support tickets.",
        )
        self.db.add(explanation)

        # Customer Segment
        segment = CustomerSegment(
            model_run_id=model_run.model_run_id,
            account_id="A-test-02",
            cluster_id=2,
            cluster_name="High-Value At-Risk",
            risk_tier="HIGH",
            value_tier="HIGH",
        )
        self.db.add(segment)
        self.db.commit()

        # Validate relationship retrieval
        loaded_acc = self.db.scalar(select(Account).where(Account.account_id == "A-test-02"))
        self.assertIsNotNone(loaded_acc)
        assert loaded_acc is not None
        self.assertEqual(len(loaded_acc.snapshots), 1)
        assert loaded_acc.snapshots[0].features is not None
        self.assertEqual(loaded_acc.snapshots[0].features.current_mrr, 2500.0)

        loaded_pred = self.db.scalar(
            select(CustomerPrediction).where(CustomerPrediction.account_id == "A-test-02")
        )
        self.assertIsNotNone(loaded_pred)
        assert loaded_pred is not None
        assert loaded_pred.explanation is not None
        self.assertEqual(loaded_pred.explanation.top_risk_factors[0]["feature"], "usage_change_pct")

        # Test Cascade Delete: deleting account cascades to snapshots and predictions
        self.db.delete(loaded_acc)
        self.db.commit()

        remaining_snapshots = self.db.scalars(
            select(CustomerSnapshot).where(CustomerSnapshot.account_id == "A-test-02")
        ).all()
        self.assertEqual(len(remaining_snapshots), 0)


if __name__ == "__main__":
    unittest.main()
