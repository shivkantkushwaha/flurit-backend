"""Unit and integration tests for point-in-time feature engineering pipeline.

Verifies:
- Lookback boundaries (30d, 60d, 90d).
- Events occurring exactly on the snapshot date (T_obs).
- Strict exclusion of future events (T_obs + 1d).
- Future perturbation invariance (mutating future data does not alter historical features).
- Missing values and sparse customer history handling.
- Repeat churners and prior churn history.
- Anti-leakage guarantees (prohibited fields strictly absent).
"""

import unittest
from datetime import date

import pandas as pd

from app.ml.features.account_features import AccountFeatureExtractor
from app.ml.features.churn_features import ChurnFeatureExtractor
from app.ml.features.pipeline import FeaturePipeline
from app.ml.features.schema import (
    FEATURE_DEFINITIONS,
    FEATURE_NAMES,
    STRICTLY_EXCLUDED_FIELDS,
)
from app.ml.features.subscription_features import SubscriptionFeatureExtractor
from app.ml.features.support_features import SupportFeatureExtractor
from app.ml.features.usage_features import UsageFeatureExtractor


class TestFeatureSchema(unittest.TestCase):
    """Tests for schema integrity and prohibited fields."""

    def test_feature_count_and_uniqueness(self) -> None:
        """Verify feature count matches definitions and names are unique."""
        self.assertEqual(len(FEATURE_DEFINITIONS), len(FEATURE_NAMES))
        self.assertEqual(len(FEATURE_NAMES), len(set(FEATURE_NAMES)))
        self.assertEqual(len(FEATURE_NAMES), 45)

    def test_prohibited_fields_not_in_features(self) -> None:
        """Verify none of the prohibited leakage fields exist in feature schema."""
        feature_set = {f.lower() for f in FEATURE_NAMES}
        for prohibited in STRICTLY_EXCLUDED_FIELDS:
            self.assertNotIn(prohibited.lower(), feature_set)


class TestAccountFeatures(unittest.TestCase):
    """Unit tests for AccountFeatureExtractor."""

    def setUp(self) -> None:
        self.accounts_df = pd.DataFrame(
            [
                {
                    "account_id": "A1",
                    "account_name": "Alpha Corp",
                    "industry": "DevTools",
                    "country": "US",
                    "signup_date": "2023-01-15",
                    "referral_source": "organic",
                    "plan_tier": "Pro",
                    "seats": 10,
                    "is_trial": False,
                    "churn_flag": True,  # Must be ignored
                }
            ]
        )
        self.extractor = AccountFeatureExtractor(self.accounts_df)

    def test_account_age_and_attributes(self) -> None:
        """Verify age calculation and attribute mapping at snapshot date."""
        snap = date(2023, 3, 1)
        feat = self.extractor.extract_features("A1", snap)

        expected_age = (snap - date(2023, 1, 15)).days
        self.assertEqual(feat["account_age_days"], expected_age)
        self.assertEqual(feat["account_industry"], "DevTools")
        self.assertEqual(feat["account_country"], "US")
        self.assertEqual(feat["account_referral_source"], "organic")
        self.assertEqual(feat["account_initial_seats"], 10)
        self.assertEqual(feat["account_is_trial"], 0)
        self.assertNotIn("churn_flag", feat)

    def test_missing_account_fallback(self) -> None:
        """Verify nonexistent account returns safe defaults."""
        feat = self.extractor.extract_features("NONEXISTENT", date(2023, 3, 1))
        self.assertEqual(feat["account_age_days"], 0)
        self.assertEqual(feat["account_industry"], "unknown")
        self.assertEqual(feat["account_initial_seats"], 1)


class TestSubscriptionFeatures(unittest.TestCase):
    """Unit tests for SubscriptionFeatureExtractor."""

    def setUp(self) -> None:
        self.subs_df = pd.DataFrame(
            [
                {
                    "subscription_id": "S1",
                    "account_id": "A1",
                    "start_date": "2023-01-20",
                    "end_date": "2023-05-01",  # Ended before snapshot
                    "plan_tier": "Basic",
                    "seats": 5,
                    "mrr_amount": 100.0,
                    "arr_amount": 1200.0,
                    "is_trial": False,
                    "upgrade_flag": False,
                    "downgrade_flag": False,
                    "churn_flag": False,
                    "billing_frequency": "monthly",
                    "auto_renew_flag": True,
                },
                {
                    "subscription_id": "S2",
                    "account_id": "A1",
                    "start_date": "2023-05-01",
                    "end_date": None,  # Active open-ended
                    "plan_tier": "Pro",
                    "seats": 15,
                    "mrr_amount": 500.0,
                    "arr_amount": 6000.0,
                    "is_trial": False,
                    "upgrade_flag": True,  # Historical upgrade
                    "downgrade_flag": False,
                    "churn_flag": False,
                    "billing_frequency": "monthly",
                    "auto_renew_flag": True,
                },
                {
                    "subscription_id": "S3",
                    "account_id": "A1",
                    "start_date": "2023-08-01",  # Future subscription
                    "end_date": None,
                    "plan_tier": "Enterprise",
                    "seats": 50,
                    "mrr_amount": 2000.0,
                    "arr_amount": 24000.0,
                    "is_trial": False,
                    "upgrade_flag": True,
                    "downgrade_flag": False,
                    "churn_flag": False,
                    "billing_frequency": "annual",
                    "auto_renew_flag": True,
                },
            ]
        )
        self.extractor = SubscriptionFeatureExtractor(self.subs_df)

    def test_active_and_historical_subscriptions(self) -> None:
        """Verify active filtering excludes ended and future contracts."""
        snap = date(2023, 6, 1)
        feat = self.extractor.extract_features("A1", snap)

        # On 2023-06-01:
        # S1 has ended (2023-05-01 <= 2023-06-01) -> not active
        # S2 is active (start 2023-05-01, end None) -> active
        # S3 has not started yet (2023-08-01 > 2023-06-01) -> not started
        self.assertEqual(feat["sub_active_count"], 1)
        self.assertEqual(feat["sub_current_mrr"], 500.0)
        self.assertEqual(feat["sub_current_arr"], 6000.0)
        self.assertEqual(feat["sub_total_active_seats"], 15)
        self.assertEqual(feat["sub_billing_frequency"], "monthly")
        self.assertEqual(feat["sub_has_auto_renew"], 1)
        # S2 had upgrade_flag=True started <= snap; S3 upgrade must NOT be counted yet
        self.assertEqual(feat["sub_historical_upgrade_count"], 1)
        self.assertEqual(feat["sub_historical_downgrade_count"], 0)
        # Commercial tenure from S1 start (2023-01-20)
        self.assertEqual(feat["sub_tenure_days"], (snap - date(2023, 1, 20)).days)


class TestUsageFeatures(unittest.TestCase):
    """Unit tests for UsageFeatureExtractor and temporal lookback boundaries."""

    def setUp(self) -> None:
        self.subs_df = pd.DataFrame([{"subscription_id": "S1", "account_id": "A1"}])
        self.usage_df = pd.DataFrame(
            [
                # Record 1: 10 days before snapshot (in 30d, 60d, 90d)
                {
                    "usage_id": "U1",
                    "subscription_id": "S1",
                    "usage_date": "2023-05-22",
                    "feature_name": "dashboard",
                    "usage_count": 10,
                    "usage_duration_secs": 600.0,
                    "error_count": 1,
                    "is_beta_feature": False,
                },
                # Record 2: Exactly on snapshot date (in 30d, 60d, 90d)
                {
                    "usage_id": "U2",
                    "subscription_id": "S1",
                    "usage_date": "2023-06-01",
                    "feature_name": "reports",
                    "usage_count": 5,
                    "usage_duration_secs": 300.0,
                    "error_count": 0,
                    "is_beta_feature": True,
                },
                # Record 3: 40 days before snapshot (in prev_30d: (60d, 30d], in 60d and 90d)
                {
                    "usage_id": "U3",
                    "subscription_id": "S1",
                    "usage_date": "2023-04-22",
                    "feature_name": "dashboard",
                    "usage_count": 20,
                    "usage_duration_secs": 1200.0,
                    "error_count": 2,
                    "is_beta_feature": False,
                },
                # Record 4: Future event (day after snapshot - must be EXCLUDED)
                {
                    "usage_id": "U4",
                    "subscription_id": "S1",
                    "usage_date": "2023-06-02",
                    "feature_name": "dashboard",
                    "usage_count": 100,
                    "usage_duration_secs": 5000.0,
                    "error_count": 5,
                    "is_beta_feature": False,
                },
            ]
        )
        self.extractor = UsageFeatureExtractor(self.usage_df, self.subs_df)

    def test_window_boundaries_and_future_exclusion(self) -> None:
        """Verify 30d, 60d, 90d windows and strict future exclusion."""
        snap = date(2023, 6, 1)
        feat = self.extractor.extract_features("A1", snap)

        # 30d window: (2023-05-02, 2023-06-01] -> U1 (10) + U2 (5) = 15 (U4 future excluded)
        self.assertEqual(feat["usage_count_30d"], 15)
        self.assertEqual(feat["usage_duration_secs_30d"], 900.0)
        self.assertEqual(feat["usage_error_count_30d"], 1)
        self.assertAlmostEqual(feat["usage_error_rate_30d"], 1 / 15)
        self.assertEqual(feat["usage_active_days_30d"], 2)
        self.assertEqual(feat["usage_distinct_features_30d"], 2)  # dashboard, reports
        self.assertEqual(feat["usage_beta_feature_count_30d"], 5)  # U2 is beta

        # 60d window: U1 (10) + U2 (5) + U3 (20) = 35
        self.assertEqual(feat["usage_count_60d"], 35)

        # Recency: latest event on or before snap is 2023-06-01 (snap itself) -> 0.0 days
        self.assertEqual(feat["usage_days_since_last"], 0.0)
        self.assertEqual(feat["usage_has_history"], 1)

        # Velocity: curr_30d = 15, prev_30d = 20 -> (15 - 20) / 20 = -0.25 (-25%)
        self.assertAlmostEqual(feat["usage_trend_pct_30d"], -0.25)


class TestSupportFeatures(unittest.TestCase):
    """Unit tests for SupportFeatureExtractor and ticket resolution state."""

    def setUp(self) -> None:
        self.tickets_df = pd.DataFrame(
            [
                # Ticket 1: 15 days ago, closed before snapshot
                {
                    "ticket_id": "T1",
                    "account_id": "A1",
                    "submitted_at": "2023-05-17",
                    "closed_at": "2023-05-18 10:00:00",
                    "resolution_time_hours": 24.0,
                    "priority": "medium",
                    "first_response_time_minutes": 30.0,
                    "satisfaction_score": 4.0,
                    "escalation_flag": False,
                },
                # Ticket 2: 5 days ago, unresolved at snapshot (closed after snapshot)
                {
                    "ticket_id": "T2",
                    "account_id": "A1",
                    "submitted_at": "2023-05-27",
                    "closed_at": "2023-06-05 12:00:00",  # Closes in future
                    "resolution_time_hours": 200.0,
                    "priority": "urgent",
                    "first_response_time_minutes": 15.0,
                    "satisfaction_score": None,  # Unrated
                    "escalation_flag": True,
                },
                # Ticket 3: Future ticket (submitted after snapshot - must be EXCLUDED)
                {
                    "ticket_id": "T3",
                    "account_id": "A1",
                    "submitted_at": "2023-06-05",
                    "closed_at": "2023-06-06",
                    "resolution_time_hours": 24.0,
                    "priority": "low",
                    "first_response_time_minutes": 60.0,
                    "satisfaction_score": 1.0,
                    "escalation_flag": False,
                },
            ]
        )
        self.extractor = SupportFeatureExtractor(self.tickets_df)

    def test_ticket_windows_and_unresolved_status(self) -> None:
        """Verify ticket counts, unrated indicator, and point-in-time unresolved logic."""
        snap = date(2023, 6, 1)
        feat = self.extractor.extract_features("A1", snap)

        # T1 and T2 are in 30d/90d window; T3 future excluded
        self.assertEqual(feat["tickets_count_30d"], 2)
        self.assertEqual(feat["tickets_count_90d"], 2)
        self.assertEqual(feat["tickets_count_lifetime"], 2)

        # Unresolved at snapshot: T2 is open at snap (closes 2023-06-05)
        self.assertEqual(feat["tickets_unresolved_count"], 1)
        self.assertEqual(feat["tickets_has_unresolved"], 1)

        # Priority and escalations
        self.assertEqual(feat["tickets_urgent_count_90d"], 1)
        self.assertEqual(feat["tickets_high_or_urgent_count_90d"], 1)
        self.assertEqual(feat["tickets_escalated_count_90d"], 1)

        # Satisfaction: T1 rated (4.0), T2 unrated (None)
        self.assertEqual(feat["tickets_avg_satisfaction_90d"], 4.0)
        self.assertEqual(feat["tickets_has_unrated_90d"], 1)


class TestChurnHistoryFeatures(unittest.TestCase):
    """Unit tests for ChurnFeatureExtractor."""

    def setUp(self) -> None:
        self.churn_df = pd.DataFrame(
            [
                # Event 1: Past churn
                {
                    "churn_event_id": "C1",
                    "account_id": "A1",
                    "churn_date": "2023-03-01",
                    "refund_amount_usd": 150.0,
                },
                # Event 2: Exactly on snapshot date
                {
                    "churn_event_id": "C2",
                    "account_id": "A1",
                    "churn_date": "2023-06-01",
                    "refund_amount_usd": 50.0,
                },
                # Event 3: Future churn event (must be EXCLUDED)
                {
                    "churn_event_id": "C3",
                    "account_id": "A1",
                    "churn_date": "2023-07-15",
                    "refund_amount_usd": 500.0,
                },
            ]
        )
        self.extractor = ChurnFeatureExtractor(self.churn_df)

    def test_historical_churn_metrics(self) -> None:
        """Verify prior churn events include events on T_obs and exclude future churn."""
        snap = date(2023, 6, 1)
        feat = self.extractor.extract_features("A1", snap)

        # C1 and C2 occurred on or before 2023-06-01; C3 is future
        self.assertEqual(feat["churn_prior_count"], 2)
        self.assertEqual(feat["churn_has_prior"], 1)
        self.assertEqual(feat["churn_days_since_last"], 0.0)  # C2 occurred on snap date
        self.assertEqual(feat["churn_prior_refund_total"], 200.0)  # 150 + 50 (C3 excluded)


class TestFuturePerturbationInvariance(unittest.TestCase):
    """Test proving mutating future data leaves historical features 100% invariant."""

    def test_future_records_do_not_alter_historical_features(self) -> None:
        """Assert feature vector at T_obs is identical before and after injecting future events."""
        snap = date(2023, 6, 1)

        acc_df = pd.DataFrame(
            [
                {
                    "account_id": "A1",
                    "account_name": "Test",
                    "industry": "FinTech",
                    "country": "US",
                    "signup_date": "2023-01-01",
                    "referral_source": "organic",
                    "plan_tier": "Basic",
                    "seats": 5,
                    "is_trial": False,
                    "churn_flag": False,
                }
            ]
        )
        sub_df = pd.DataFrame(
            [
                {
                    "subscription_id": "S1",
                    "account_id": "A1",
                    "start_date": "2023-01-01",
                    "end_date": None,
                    "plan_tier": "Basic",
                    "seats": 5,
                    "mrr_amount": 100.0,
                    "arr_amount": 1200.0,
                    "is_trial": False,
                    "upgrade_flag": False,
                    "downgrade_flag": False,
                    "churn_flag": False,
                    "billing_frequency": "monthly",
                    "auto_renew_flag": True,
                }
            ]
        )
        usage_df = pd.DataFrame(
            [
                {
                    "usage_id": "U1",
                    "subscription_id": "S1",
                    "usage_date": "2023-05-15",
                    "feature_name": "core",
                    "usage_count": 50,
                    "usage_duration_secs": 1000.0,
                    "error_count": 1,
                    "is_beta_feature": False,
                }
            ]
        )
        tickets_df = pd.DataFrame(
            [
                {
                    "ticket_id": "T1",
                    "account_id": "A1",
                    "submitted_at": "2023-05-20",
                    "closed_at": "2023-05-21",
                    "resolution_time_hours": 24.0,
                    "priority": "low",
                    "first_response_time_minutes": 10.0,
                    "satisfaction_score": 5.0,
                    "escalation_flag": False,
                }
            ]
        )
        churn_df = pd.DataFrame(
            [
                {
                    "churn_event_id": "C1",
                    "account_id": "A1",
                    "churn_date": "2023-04-01",
                    "refund_amount_usd": 0.0,
                }
            ]
        )

        pipeline1 = FeaturePipeline(acc_df, sub_df, usage_df, tickets_df, churn_df)
        feat1 = pipeline1.extract_snapshot_features("A1", snap)

        # Inject future events into all 4 dynamic tables at T_obs + 10d
        sub_df_future = pd.concat(
            [
                sub_df,
                pd.DataFrame(
                    [
                        {
                            "subscription_id": "S2",
                            "account_id": "A1",
                            "start_date": "2023-07-01",
                            "end_date": None,
                            "plan_tier": "Enterprise",
                            "seats": 100,
                            "mrr_amount": 5000.0,
                            "arr_amount": 60000.0,
                            "is_trial": False,
                            "upgrade_flag": True,
                            "downgrade_flag": False,
                            "churn_flag": False,
                            "billing_frequency": "annual",
                            "auto_renew_flag": True,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        usage_df_future = pd.concat(
            [
                usage_df,
                pd.DataFrame(
                    [
                        {
                            "usage_id": "U2",
                            "subscription_id": "S1",
                            "usage_date": "2023-06-15",
                            "feature_name": "core",
                            "usage_count": 9999,
                            "usage_duration_secs": 99999.0,
                            "error_count": 50,
                            "is_beta_feature": True,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        tickets_df_future = pd.concat(
            [
                tickets_df,
                pd.DataFrame(
                    [
                        {
                            "ticket_id": "T2",
                            "account_id": "A1",
                            "submitted_at": "2023-06-20",
                            "closed_at": "2023-06-21",
                            "resolution_time_hours": 100.0,
                            "priority": "urgent",
                            "first_response_time_minutes": 500.0,
                            "satisfaction_score": 1.0,
                            "escalation_flag": True,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        churn_df_future = pd.concat(
            [
                churn_df,
                pd.DataFrame(
                    [
                        {
                            "churn_event_id": "C2",
                            "account_id": "A1",
                            "churn_date": "2023-07-01",
                            "refund_amount_usd": 1000.0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        pipeline2 = FeaturePipeline(
            acc_df, sub_df_future, usage_df_future, tickets_df_future, churn_df_future
        )
        feat2 = pipeline2.extract_snapshot_features("A1", snap)

        # Assert 100% equality
        self.assertEqual(feat1, feat2)


if __name__ == "__main__":
    unittest.main()
