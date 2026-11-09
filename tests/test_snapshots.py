"""Unit and integration tests for snapshot engine and target constructor.

Tests:
- Point-in-time eligibility (accounts signed up after T_obs excluded).
- Target labeling window boundaries (T_obs < churn_date <= T_obs + 60d).
- Events occurring on T_obs itself are not counted as future churn (anti-leakage).
- Right-censoring beyond cutoff date (target is None).
- Deterministic repeatability and uniqueness constraints.
- Repeat churn and reactivation scenarios.
"""

import unittest
from datetime import date, timedelta

import pandas as pd

from app.ml.snapshots.engine import SnapshotEngine
from app.ml.snapshots.targets import TargetConstructor


class TestTargetConstructor(unittest.TestCase):
    """Unit tests for TargetConstructor logic and temporal window boundaries."""

    def setUp(self) -> None:
        self.global_max = date(2024, 12, 31)
        self.constructor = TargetConstructor(
            prediction_window_days=60,
            global_max_date=self.global_max,
        )

    def test_cutoff_date_calculation(self) -> None:
        """Verify cutoff date is exactly global_max - 60 days."""
        expected_cutoff = self.global_max - timedelta(days=60)
        self.assertEqual(self.constructor.cutoff_date, expected_cutoff)
        self.assertEqual(expected_cutoff, date(2024, 11, 1))

    def test_right_censoring_after_cutoff(self) -> None:
        """Verify snapshots after cutoff date evaluate to None (right-censored)."""
        post_cutoff = date(2024, 11, 2)
        self.assertTrue(self.constructor.is_right_censored(post_cutoff))

        churns_df = pd.DataFrame(
            [{"account_id": "A1", "parsed_churn_date": pd.to_datetime("2024-12-15")}]
        )
        target = self.constructor.compute_target("A1", post_cutoff, churns_df)
        self.assertIsNone(target)

    def test_churn_event_strictly_after_snapshot(self) -> None:
        """Verify churn on T_obs itself is NOT future churn (leakage prevention)."""
        snap = date(2024, 6, 1)
        # Churn happened ON snapshot date
        churns_df = pd.DataFrame(
            [{"account_id": "A1", "parsed_churn_date": pd.to_datetime("2024-06-01")}]
        )
        target = self.constructor.compute_target("A1", snap, churns_df)
        # Must be 0 because event occurred at or before T_obs, not strictly in (T_obs, T_obs + 60d]
        self.assertEqual(target, 0)

    def test_churn_event_within_window(self) -> None:
        """Verify churn event within 60 days results in target = 1."""
        snap = date(2024, 6, 1)
        # Churn happened 30 days later
        churns_df = pd.DataFrame(
            [{"account_id": "A1", "parsed_churn_date": pd.to_datetime("2024-07-01")}]
        )
        target = self.constructor.compute_target("A1", snap, churns_df)
        self.assertEqual(target, 1)

    def test_churn_event_at_exact_window_boundary(self) -> None:
        """Verify churn event at exactly T_obs + 60d is included (target = 1)."""
        snap = date(2024, 6, 1)
        exact_60d = snap + timedelta(days=60)
        churns_df = pd.DataFrame(
            [{"account_id": "A1", "parsed_churn_date": pd.to_datetime(exact_60d)}]
        )
        target = self.constructor.compute_target("A1", snap, churns_df)
        self.assertEqual(target, 1)

    def test_churn_event_after_window_boundary(self) -> None:
        """Verify churn event beyond T_obs + 60d is NOT included (target = 0)."""
        snap = date(2024, 6, 1)
        after_60d = snap + timedelta(days=61)
        churns_df = pd.DataFrame(
            [{"account_id": "A1", "parsed_churn_date": pd.to_datetime(after_60d)}]
        )
        target = self.constructor.compute_target("A1", snap, churns_df)
        self.assertEqual(target, 0)


class TestSnapshotEngine(unittest.TestCase):
    """Unit and synthetic tests for SnapshotEngine cohort generation."""

    def setUp(self) -> None:
        self.accounts_df = pd.DataFrame(
            [
                {
                    "account_id": "A1",
                    "account_name": "Co1",
                    "industry": "DevTools",
                    "country": "US",
                    "signup_date": "2023-01-10",
                    "referral_source": "organic",
                    "plan_tier": "Pro",
                    "seats": 5,
                    "is_trial": False,
                    "churn_flag": True,  # Source retrospective flag
                },
                {
                    "account_id": "A2",
                    "account_name": "Co2",
                    "industry": "FinTech",
                    "country": "UK",
                    "signup_date": "2023-05-15",
                    "referral_source": "partner",
                    "plan_tier": "Basic",
                    "seats": 2,
                    "is_trial": False,
                    "churn_flag": False,
                },
            ]
        )

        self.subscriptions_df = pd.DataFrame(
            [
                {
                    "subscription_id": "S1",
                    "account_id": "A1",
                    "start_date": "2023-01-15",
                    "end_date": None,
                    "plan_tier": "Pro",
                    "seats": 5,
                    "mrr_amount": 500,
                    "arr_amount": 6000,
                    "is_trial": False,
                    "upgrade_flag": False,
                    "downgrade_flag": False,
                    "churn_flag": False,
                    "billing_frequency": "monthly",
                    "auto_renew_flag": True,
                },
                {
                    "subscription_id": "S2",
                    "account_id": "A2",
                    "start_date": "2023-05-20",
                    "end_date": None,
                    "plan_tier": "Basic",
                    "seats": 2,
                    "mrr_amount": 100,
                    "arr_amount": 1200,
                    "is_trial": False,
                    "upgrade_flag": False,
                    "downgrade_flag": False,
                    "churn_flag": False,
                    "billing_frequency": "monthly",
                    "auto_renew_flag": True,
                },
            ]
        )

        self.churn_events_df = pd.DataFrame(
            [
                {
                    "churn_event_id": "C1",
                    "account_id": "A1",
                    "churn_date": "2023-07-20",
                    "reason_code": "pricing",
                    "refund_amount_usd": 0.0,
                    "preceding_upgrade_flag": False,
                    "preceding_downgrade_flag": False,
                    "is_reactivation": False,
                    "feedback_text": "too expensive",
                }
            ]
        )

    def test_account_not_signed_up_is_excluded(self) -> None:
        """Verify accounts signed up after snapshot date are excluded."""
        engine = SnapshotEngine(
            accounts_df=self.accounts_df,
            subscriptions_df=self.subscriptions_df,
            churn_events_df=self.churn_events_df,
            prediction_window_days=60,
        )
        # On 2023-03-01, A2 has not signed up yet (signup: 2023-05-15)
        eligible = engine.get_eligible_accounts(date(2023, 3, 1))
        self.assertIn("A1", eligible)
        self.assertNotIn("A2", eligible)

    def test_target_labeling_across_dates(self) -> None:
        """Verify target is 1 when churn occurs in (T_obs, T_obs + 60d], and 0 when not."""
        engine = SnapshotEngine(
            accounts_df=self.accounts_df,
            subscriptions_df=self.subscriptions_df,
            churn_events_df=self.churn_events_df,
            prediction_window_days=60,
            global_max_date=date(2023, 12, 31),
        )
        # On 2023-06-01: A1 churns on 2023-07-20 (within 60d -> target=1)
        # A2 does not churn (target=0)
        snapshots = engine.generate_snapshots_for_date(date(2023, 6, 1))
        snap_map = {s.account_id: s.target_churn_60d for s in snapshots}
        self.assertEqual(snap_map["A1"], 1)
        self.assertEqual(snap_map["A2"], 0)

    def test_deterministic_repeatability_and_uniqueness(self) -> None:
        """Verify snapshot generation produces identical DataFrames with no duplicate keys."""
        engine = SnapshotEngine(
            accounts_df=self.accounts_df,
            subscriptions_df=self.subscriptions_df,
            churn_events_df=self.churn_events_df,
            prediction_window_days=60,
            global_max_date=date(2023, 12, 31),
        )
        dates = [date(2023, 6, 1), date(2023, 7, 1)]
        df1 = engine.generate_monthly_snapshots(observation_dates=dates)
        df2 = engine.generate_monthly_snapshots(observation_dates=dates)
        self.assertTrue(df1.equals(df2))
        self.assertEqual(df1.duplicated(subset=["account_id", "snapshot_date"]).sum(), 0)

    def test_sub_active_excludes_pre_subscription_account(self) -> None:
        """Verify subscription_active policy excludes signed-up accounts if subscription hasn't started."""
        engine = SnapshotEngine(
            accounts_df=self.accounts_df,
            subscriptions_df=self.subscriptions_df,
            churn_events_df=self.churn_events_df,
            prediction_window_days=60,
            eligibility_policy="subscription_active",
        )
        # On 2023-05-16, A2 is signed up (2023-05-15) but sub starts 2023-05-20
        eligible = engine.get_eligible_accounts(date(2023, 5, 16))
        self.assertNotIn("A2", eligible)

    def test_churn_free_excludes_account_with_prior_churn(self) -> None:
        """Verify churn_free policy excludes accounts that churned on or before snapshot date."""
        engine = SnapshotEngine(
            accounts_df=self.accounts_df,
            subscriptions_df=self.subscriptions_df,
            churn_events_df=self.churn_events_df,
            prediction_window_days=60,
            eligibility_policy="churn_free",
        )
        # On 2023-08-01, A1 churned on 2023-07-20 -> must be excluded
        eligible = engine.get_eligible_accounts(date(2023, 8, 1))
        self.assertNotIn("A1", eligible)
        self.assertIn("A2", eligible)

    def test_repeat_churners_receive_target_one(self) -> None:
        """Verify multiple churn events in prediction window evaluate to binary target 1."""
        multi_churn_df = pd.DataFrame(
            [
                {
                    "churn_event_id": "C1",
                    "account_id": "A1",
                    "churn_date": "2023-06-15",
                    "reason_code": "pricing",
                    "refund_amount_usd": 0.0,
                    "preceding_upgrade_flag": False,
                    "preceding_downgrade_flag": False,
                    "is_reactivation": False,
                    "feedback_text": "expensive",
                },
                {
                    "churn_event_id": "C2",
                    "account_id": "A1",
                    "churn_date": "2023-07-10",
                    "reason_code": "bugs",
                    "refund_amount_usd": 0.0,
                    "preceding_upgrade_flag": False,
                    "preceding_downgrade_flag": False,
                    "is_reactivation": False,
                    "feedback_text": "buggy",
                },
            ]
        )
        engine = SnapshotEngine(
            accounts_df=self.accounts_df,
            subscriptions_df=self.subscriptions_df,
            churn_events_df=multi_churn_df,
            prediction_window_days=60,
            global_max_date=date(2023, 12, 31),
        )
        snapshots = engine.generate_snapshots_for_date(date(2023, 6, 1))
        snap_map = {s.account_id: s.target_churn_60d for s in snapshots}
        self.assertEqual(snap_map["A1"], 1)


if __name__ == "__main__":
    unittest.main()
