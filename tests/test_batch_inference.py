"""Unit tests for Flurit Batch Inference Pipeline.

Tests batch execution, configurable chunk sizes, deterministic scoring,
validation error handling (fail-fast vs aggregation), and output integrity.
"""

import unittest
from datetime import date
from typing import Any

from app.ml.inference.batch import (
    BatchInferencePipeline,
    BatchInputRecord,
    BatchPipelineSummary,
)
from app.ml.inference.service import get_inference_service


def _get_sample_features(multiplier: float = 1.0) -> dict[str, Any]:
    """Helper returning valid 45 canonical features with adjustable values."""
    return {
        "account_age_days": int(150 * multiplier),
        "account_industry": "Tech" if multiplier >= 1.0 else "Finance",
        "account_country": "US",
        "account_referral_source": "organic",
        "account_initial_seats": int(20 * multiplier),
        "account_is_trial": 0,
        "sub_active_count": int(2 * multiplier),
        "sub_current_mrr": round(6500.0 * multiplier, 2),
        "sub_current_arr": round(78000.0 * multiplier, 2),
        "sub_total_active_seats": int(35 * multiplier),
        "sub_tenure_days": round(210.0 * multiplier, 1),
        "sub_billing_frequency": "monthly",
        "sub_has_auto_renew": 1,
        "sub_has_active_trial": 0,
        "sub_historical_upgrade_count": 1,
        "sub_historical_downgrade_count": 0,
        "usage_count_30d": int(120 * multiplier),
        "usage_count_60d": int(240 * multiplier),
        "usage_count_90d": int(350 * multiplier),
        "usage_duration_secs_30d": round(15000.0 * multiplier, 1),
        "usage_duration_secs_90d": round(42000.0 * multiplier, 1),
        "usage_error_count_30d": 1,
        "usage_error_rate_30d": 0.008,
        "usage_active_days_30d": min(30, int(22 * multiplier)),
        "usage_active_days_90d": min(90, int(65 * multiplier)),
        "usage_distinct_features_30d": min(10, int(7 * multiplier)),
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


class TestBatchInferencePipeline(unittest.TestCase):
    """Test suite verifying batch inference pipeline execution and edge cases."""

    def setUp(self) -> None:
        self.service = get_inference_service()
        self.pipeline = BatchInferencePipeline(service=self.service, batch_size=2)

    def test_invalid_batch_size_rejected(self) -> None:
        """Non-positive batch sizes must raise ValueError."""
        with self.assertRaises(ValueError):
            BatchInferencePipeline(service=self.service, batch_size=0)
        with self.assertRaises(ValueError):
            BatchInferencePipeline(service=self.service, batch_size=-5)

    def test_batch_inference_execution_and_chunking(self) -> None:
        """Batch pipeline correctly chunks records, computes scores, and structures outputs."""
        records = [
            BatchInputRecord(
                account_id=f"A-{i:03d}",
                snapshot_date="2024-11-01",
                snapshot_id=100 + i,
                features=_get_sample_features(multiplier=0.8 + 0.2 * i),
            )
            for i in range(5)
        ]

        summary = self.pipeline.process_records(records)
        self.assertIsInstance(summary, BatchPipelineSummary)
        self.assertEqual(summary.total_records, 5)
        self.assertEqual(summary.successful_records, 5)
        self.assertEqual(summary.failed_records, 0)
        self.assertEqual(summary.batch_count, 3)  # 5 records in chunks of 2 -> 3 chunks
        self.assertEqual(len(summary.predictions), 5)
        self.assertEqual(len(summary.errors), 0)

        for p in summary.predictions:
            self.assertTrue(p.account_id.startswith("A-"))
            self.assertEqual(p.snapshot_date, "2024-11-01")
            self.assertGreaterEqual(p.model_score, 0.0)
            self.assertLessEqual(p.model_score, 1.0)
            self.assertIn(p.risk_tier, ["High Risk", "Moderate Risk", "Low Risk"])
            self.assertIn(p.cluster_id, [0, 1, 2])
            self.assertIn(p.value_tier, ["High Value", "Mid Value", "Core Value"])
            self.assertGreater(len(p.top_risk_increasing_factors), 0)
            self.assertGreater(len(p.top_risk_decreasing_factors), 0)
            self.assertEqual(len(p.shap_values), 45)

    def test_batch_inference_deterministic_outputs(self) -> None:
        """Repeated runs on the same input records produce identical scores and explanations."""
        records = [
            BatchInputRecord(
                account_id="A-DET-01",
                snapshot_date=date(2024, 11, 1),
                features=_get_sample_features(multiplier=1.0),
            ),
            BatchInputRecord(
                account_id="A-DET-02",
                snapshot_date=date(2024, 11, 1),
                features=_get_sample_features(multiplier=1.5),
            ),
        ]

        run1 = self.pipeline.process_records(records)
        run2 = self.pipeline.process_records(records)

        self.assertEqual(len(run1.predictions), len(run2.predictions))
        for p1, p2 in zip(run1.predictions, run2.predictions, strict=True):
            self.assertEqual(p1.account_id, p2.account_id)
            self.assertEqual(p1.model_score, p2.model_score)
            self.assertEqual(p1.risk_tier, p2.risk_tier)
            self.assertEqual(p1.cluster_id, p2.cluster_id)
            self.assertEqual(p1.shap_values, p2.shap_values)

    def test_validation_error_aggregation_fail_fast_false(self) -> None:
        """When fail_fast=False, invalid records are recorded as errors without crashing."""
        pipeline = BatchInferencePipeline(service=self.service, batch_size=5, fail_fast=False)

        bad_feat_missing = _get_sample_features()
        del bad_feat_missing["sub_current_mrr"]

        bad_feat_extra = _get_sample_features()
        bad_feat_extra["target_churn_60d"] = 1  # Prohibited target field

        records = [
            BatchInputRecord(
                account_id="A-GOOD-01", snapshot_date="2024-11-01", features=_get_sample_features()
            ),
            BatchInputRecord(
                account_id="A-BAD-01", snapshot_date="2024-11-01", features=bad_feat_missing
            ),
            BatchInputRecord(
                account_id="A-GOOD-02",
                snapshot_date="2024-11-01",
                features=_get_sample_features(multiplier=1.2),
            ),
            BatchInputRecord(
                account_id="A-BAD-02", snapshot_date="2024-11-01", features=bad_feat_extra
            ),
            BatchInputRecord(
                account_id="", snapshot_date="2024-11-01", features=_get_sample_features()
            ),  # Empty account_id
        ]

        summary = pipeline.process_records(records)
        self.assertEqual(summary.total_records, 5)
        self.assertEqual(summary.successful_records, 2)
        self.assertEqual(summary.failed_records, 3)
        self.assertEqual(len(summary.errors), 3)

        error_accounts = {e.account_id for e in summary.errors}
        self.assertIn("A-BAD-01", error_accounts)
        self.assertIn("A-BAD-02", error_accounts)
        self.assertIn("missing", error_accounts)

    def test_validation_fail_fast_true_raises(self) -> None:
        """When fail_fast=True, an invalid record immediately raises ValueError."""
        pipeline = BatchInferencePipeline(service=self.service, batch_size=5, fail_fast=True)
        bad_feat = _get_sample_features()
        del bad_feat["sub_current_mrr"]

        records = [
            BatchInputRecord(
                account_id="A-GOOD-01", snapshot_date="2024-11-01", features=_get_sample_features()
            ),
            BatchInputRecord(account_id="A-BAD-01", snapshot_date="2024-11-01", features=bad_feat),
        ]

        with self.assertRaises(ValueError) as ctx:
            pipeline.process_records(records)
        self.assertIn("Validation error", str(ctx.exception))

    def test_empty_records_returns_clean_summary(self) -> None:
        """Processing an empty record collection returns zeroed summary without error."""
        summary = self.pipeline.process_records([])
        self.assertEqual(summary.total_records, 0)
        self.assertEqual(summary.successful_records, 0)
        self.assertEqual(summary.failed_records, 0)
        self.assertEqual(summary.batch_count, 0)
        self.assertEqual(len(summary.predictions), 0)


if __name__ == "__main__":
    unittest.main()
