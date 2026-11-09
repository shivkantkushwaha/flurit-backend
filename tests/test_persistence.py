"""Unit and Integration Tests for Flurit ML Persistence Layer and Repositories.

Tests SQLAlchemy repository operations, idempotent batch writes, duplicate handling,
transaction rollback on failure, and high-level BatchPredictionService orchestration
using an isolated in-memory SQLite database.
"""

import unittest
from typing import Any
from unittest.mock import patch

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.ml.inference.batch import (
    BatchInferencePipeline,
    BatchInputRecord,
    BatchPredictionOutput,
)
from app.ml.inference.service import get_inference_service
from app.models import (
    Base,
    CustomerExplanation,
    CustomerPrediction,
    CustomerSegment,
    ModelRun,
)
from app.repositories.prediction_repository import PredictionRepository
from app.services.prediction_service import BatchPredictionService


def _get_sample_features(multiplier: float = 1.0) -> dict[str, Any]:
    """Helper returning valid 45 canonical features."""
    return {
        "account_age_days": int(150 * multiplier),
        "account_industry": "Tech",
        "account_country": "US",
        "account_referral_source": "organic",
        "account_initial_seats": 20,
        "account_is_trial": 0,
        "sub_active_count": 2,
        "sub_current_mrr": round(6500.0 * multiplier, 2),
        "sub_current_arr": round(78000.0 * multiplier, 2),
        "sub_total_active_seats": 35,
        "sub_tenure_days": 210.0,
        "sub_billing_frequency": "monthly",
        "sub_has_auto_renew": 1,
        "sub_has_active_trial": 0,
        "sub_historical_upgrade_count": 1,
        "sub_historical_downgrade_count": 0,
        "usage_count_30d": 120,
        "usage_count_60d": 240,
        "usage_count_90d": 350,
        "usage_duration_secs_30d": 15000.0,
        "usage_duration_secs_90d": 42000.0,
        "usage_error_count_30d": 1,
        "usage_error_rate_30d": 0.008,
        "usage_active_days_30d": 22,
        "usage_active_days_90d": 65,
        "usage_distinct_features_30d": 7,
        "usage_beta_feature_count_30d": 1,
        "usage_days_since_last": 1.2,
        "usage_has_history": 1,
        "usage_trend_pct_30d": 5.4,
        "tickets_count_30d": 1,
        "tickets_count_90d": 2,
        "tickets_count_lifetime": 4,
        "tickets_unresolved_count": 0,
        "tickets_has_unresolved": 0,
        "tickets_urgent_count_90d": 0,
        "tickets_high_or_urgent_count_90d": 1,
        "tickets_escalated_count_90d": 0,
        "tickets_avg_satisfaction_90d": 4.8,
        "tickets_has_unrated_90d": 0,
        "tickets_avg_first_response_minutes_90d": 35.0,
        "churn_prior_count": 0,
        "churn_has_prior": 0,
        "churn_days_since_last": -1.0,
        "churn_prior_refund_total": 0.0,
    }


def _create_mock_prediction_output(
    account_id: str,
    score: float = 0.65,
    snapshot_id: int | None = None,
) -> BatchPredictionOutput:
    """Helper creating a complete BatchPredictionOutput item for repository testing."""
    return BatchPredictionOutput(
        account_id=account_id,
        snapshot_date="2024-11-01",
        snapshot_id=snapshot_id,
        model_name="flurit_churn_predictor",
        model_version="1.0.0",
        model_score=score,
        predicted_risk_label=1 if score >= 0.50 else 0,
        risk_tier="High Risk" if score >= 0.50 else "Low Risk",
        cluster_id=1,
        cluster_name="Cluster 1: High-Value Enterprise Accounts",
        value_tier="High Value",
        top_risk_increasing_factors=[{"feature_name": "sub_active_count", "attribution": 0.5}],
        top_risk_decreasing_factors=[{"feature_name": "sub_tenure_days", "attribution": -0.3}],
        shap_values={"sub_active_count": 0.5, "sub_tenure_days": -0.3},
        plain_language_summary=f"Score is {score:.2f}",
        feature_payload=_get_sample_features(),
    )


class TestPredictionRepository(unittest.TestCase):
    """Test suite for PredictionRepository in an isolated in-memory SQLite database."""

    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.db: Session = self.SessionLocal()
        self.repo = PredictionRepository()

    def tearDown(self) -> None:
        self.db.close()
        Base.metadata.drop_all(self.engine)

    def test_get_or_create_model_run_idempotent(self) -> None:
        """Calling get_or_create_model_run repeatedly returns the same record without duplicates."""
        run1 = self.repo.get_or_create_model_run(
            session=self.db,
            model_name="flurit_churn_predictor",
            version="1.0.0",
            metadata={"decision_threshold": 0.50},
        )
        self.assertIsNotNone(run1.model_run_id)
        first_id = run1.model_run_id

        # Second call with same name and version
        run2 = self.repo.get_or_create_model_run(
            session=self.db,
            model_name="flurit_churn_predictor",
            version="1.0.0",
        )
        self.assertEqual(run2.model_run_id, first_id)

        # Count total model runs in DB
        total_runs = self.db.scalar(select(func.count(ModelRun.model_run_id)))
        self.assertEqual(total_runs, 1)

    def test_save_prediction_batch_inserts_records(self) -> None:
        """Batch persistence correctly writes predictions, explanations, and segments."""
        model_run = self.repo.get_or_create_model_run(
            session=self.db,
            model_name="flurit_churn_predictor",
            version="1.0.0",
        )

        outputs = [
            _create_mock_prediction_output(account_id="A-PERSIST-01", score=0.72, snapshot_id=101),
            _create_mock_prediction_output(account_id="A-PERSIST-02", score=0.28, snapshot_id=102),
        ]

        saved_count = self.repo.save_prediction_batch(
            session=self.db,
            model_run_id=model_run.model_run_id,
            predictions=outputs,
        )
        self.db.commit()
        self.assertEqual(saved_count, 2)

        # Verify predictions
        preds = self.db.scalars(select(CustomerPrediction)).all()
        self.assertEqual(len(preds), 2)
        p1 = next(p for p in preds if p.account_id == "A-PERSIST-01")
        self.assertEqual(p1.churn_probability, 0.72)
        self.assertEqual(p1.risk_level, "High Risk")

        # Verify linked explanation
        self.assertIsNotNone(p1.explanation)
        assert p1.explanation is not None
        self.assertEqual(p1.explanation.account_id, "A-PERSIST-01")
        self.assertIn("Score is 0.72", str(p1.explanation.summary_text))

        # Verify segment assignment
        segments = self.db.scalars(select(CustomerSegment)).all()
        self.assertEqual(len(segments), 2)
        seg1 = next(s for s in segments if s.account_id == "A-PERSIST-01")
        self.assertEqual(seg1.cluster_id, 1)
        self.assertEqual(seg1.cluster_name, "Cluster 1: High-Value Enterprise Accounts")

    def test_idempotent_repeat_runs_updates_records(self) -> None:
        """Re-saving the same batch updates probabilities and does not duplicate rows."""
        model_run = self.repo.get_or_create_model_run(
            session=self.db,
            model_name="flurit_churn_predictor",
            version="1.0.0",
        )

        outputs_initial = [
            _create_mock_prediction_output(account_id="A-IDEM-01", score=0.55, snapshot_id=201),
        ]
        self.repo.save_prediction_batch(self.db, model_run.model_run_id, outputs_initial)
        self.db.commit()

        initial_count = self.db.scalar(select(func.count(CustomerPrediction.prediction_id)))
        self.assertEqual(initial_count, 1)

        # Re-save with updated score
        outputs_updated = [
            _create_mock_prediction_output(account_id="A-IDEM-01", score=0.88, snapshot_id=201),
        ]
        self.repo.save_prediction_batch(self.db, model_run.model_run_id, outputs_updated)
        self.db.commit()

        # Row count remains 1, value updated
        final_count = self.db.scalar(select(func.count(CustomerPrediction.prediction_id)))
        self.assertEqual(final_count, 1)

        pred = self.db.scalars(select(CustomerPrediction)).first()
        self.assertIsNotNone(pred)
        assert pred is not None
        self.assertEqual(pred.churn_probability, 0.88)
        self.assertEqual(pred.risk_level, "High Risk")

    def test_transaction_rollback_on_failure(self) -> None:
        """Database error inside save_prediction_batch rolls back changes completely."""
        model_run = self.repo.get_or_create_model_run(
            session=self.db,
            model_name="flurit_churn_predictor",
            version="1.0.0",
        )

        # Pre-seed one record
        initial = [_create_mock_prediction_output(account_id="A-SEED-01", score=0.40)]
        self.repo.save_prediction_batch(self.db, model_run.model_run_id, initial)
        self.db.commit()

        initial_count = self.db.scalar(select(func.count(CustomerPrediction.prediction_id)))
        self.assertEqual(initial_count, 1)

        # Simulate exception during batch persistence
        bad_output = _create_mock_prediction_output(account_id="A-FAIL-01", score=0.90)

        with patch.object(
            self.db, "flush", side_effect=RuntimeError("Simulated DB connection error")
        ):
            with self.assertRaises(RuntimeError):
                self.repo.save_prediction_batch(
                    session=self.db,
                    model_run_id=model_run.model_run_id,
                    predictions=[bad_output],
                )

        # Verify rollback preserved initial state exactly
        post_failure_count = self.db.scalar(select(func.count(CustomerPrediction.prediction_id)))
        self.assertEqual(post_failure_count, initial_count)

    def test_query_methods(self) -> None:
        """Query helper methods retrieve latest prediction, explanation, and segment."""
        model_run = self.repo.get_or_create_model_run(
            session=self.db,
            model_name="flurit_churn_predictor",
            version="1.0.0",
        )
        outputs = [
            _create_mock_prediction_output(account_id="A-QUERY-01", score=0.62),
        ]
        self.repo.save_prediction_batch(self.db, model_run.model_run_id, outputs)
        self.db.commit()

        latest = self.repo.get_latest_prediction_by_account(self.db, "A-QUERY-01")
        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertEqual(latest.churn_probability, 0.62)

        pred_with_exp = self.repo.get_prediction_with_explanation(self.db, latest.prediction_id)
        self.assertIsNotNone(pred_with_exp)
        assert pred_with_exp is not None
        self.assertIsNotNone(pred_with_exp.explanation)

        seg = self.repo.get_customer_segment(self.db, "A-QUERY-01")
        self.assertIsNotNone(seg)
        assert seg is not None
        self.assertEqual(seg.cluster_id, 1)

        run_preds = self.repo.get_predictions_by_model_run(self.db, model_run.model_run_id)
        self.assertEqual(len(run_preds), 1)


class TestBatchPredictionService(unittest.TestCase):
    """End-to-end integration tests for BatchPredictionService orchestration."""

    def setUp(self) -> None:
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.db: Session = self.SessionLocal()

        self.inference_service = get_inference_service()
        self.pipeline = BatchInferencePipeline(service=self.inference_service, batch_size=2)
        self.repo = PredictionRepository()
        self.service = BatchPredictionService(pipeline=self.pipeline, repository=self.repo)

    def tearDown(self) -> None:
        self.db.close()
        Base.metadata.drop_all(self.engine)

    def test_run_and_persist_batch_end_to_end(self) -> None:
        """Full batch run scores records and commits predictions to the database."""
        records = [
            BatchInputRecord(
                account_id=f"A-E2E-{i:02d}",
                snapshot_date="2024-11-01",
                snapshot_id=300 + i,
                features=_get_sample_features(multiplier=0.9 + 0.1 * i),
            )
            for i in range(4)
        ]

        summary = self.service.run_and_persist_batch(session=self.db, records=records)
        self.assertEqual(summary.total_records, 4)
        self.assertEqual(summary.successful_records, 4)
        self.assertEqual(summary.failed_records, 0)

        # Verify DB has 4 predictions, 4 explanations, 4 segments, and 1 model run
        total_preds = self.db.scalar(select(func.count(CustomerPrediction.prediction_id)))
        total_exps = self.db.scalar(select(func.count(CustomerExplanation.explanation_id)))
        total_segs = self.db.scalar(select(func.count(CustomerSegment.segment_id)))
        total_runs = self.db.scalar(select(func.count(ModelRun.model_run_id)))

        self.assertEqual(total_preds, 4)
        self.assertEqual(total_exps, 4)
        self.assertEqual(total_segs, 4)
        self.assertEqual(total_runs, 1)

    def test_run_and_persist_batch_rollback_on_database_error(self) -> None:
        """Failure during database commit rolls back session and leaves DB clean."""
        records = [
            BatchInputRecord(
                account_id="A-FAIL-01",
                snapshot_date="2024-11-01",
                features=_get_sample_features(),
            )
        ]

        with patch.object(
            self.repo, "save_prediction_batch", side_effect=RuntimeError("Simulated write error")
        ):
            with self.assertRaises(RuntimeError):
                self.service.run_and_persist_batch(session=self.db, records=records)

        # Database remains empty
        total_preds = self.db.scalar(select(func.count(CustomerPrediction.prediction_id)))
        self.assertEqual(total_preds, 0)


if __name__ == "__main__":
    unittest.main()
