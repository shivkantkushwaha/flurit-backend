"""End-to-End Processing and ML Pipeline Integration Tests.

Validates:
  1. Full execution with compatible 5-concept synthetic dataset (snapshotting -> features -> scoring).
  2. Incompatible dataset (missing usage/support) honest boundary: scoring blocked without fabricated data.
  3. Insufficient historical span (<60d) target labeling boundary.
  4. Orphan record quarantine and referential integrity enforcement.
  5. Anti-leakage safeguards ensuring post-snapshot events do not contaminate features.
  6. API routes POST /api/v1/onboarding/preview and POST /api/v1/onboarding/process.
"""

import time
import unittest
from datetime import date

import pandas as pd
from starlette.testclient import TestClient

from app.core.auth import create_jwt_token
from app.core.config import get_settings
from app.main import app
from app.ml.features.pipeline import FeaturePipeline
from app.onboarding.processor import DatasetProcessingEngine
from app.onboarding.schemas import (
    AccountConceptMapping,
    ChurnEventsConceptMapping,
    FeatureUsageConceptMapping,
    ProcessingOptions,
    SchemaMappingRequest,
    SubscriptionConceptMapping,
    SupportTicketsConceptMapping,
)
from app.onboarding.service import get_onboarding_service
from app.onboarding.transformer import SchemaTransformer

TEST_SECRET = "flurit-secure-test-jwt-secret-key-32-chars-min!"


def _generate_valid_jwt(user_id: str = "pipeline-user-789") -> str:
    """Generate a valid signed Supabase JWT token for authenticated testing."""
    payload = {
        "sub": user_id,
        "email": "lead-analyst@example.com",
        "role": "authenticated",
        "aud": "authenticated",
        "exp": int(time.time()) + 3600,
        "iat": int(time.time()),
    }
    return create_jwt_token(payload, secret=TEST_SECRET)


def _build_synthetic_compatible_csvs() -> dict[str, str]:
    """Build synthetic 5-concept CSV string fixtures spanning 240 days (2024-01-01 to 2024-08-30)."""
    # 1. Accounts
    accounts_csv = (
        "cust_id,created_date,vertical,hq_country,acquisition_channel,licensed_seats,is_trial_flag\n"
        "cust_1,2024-01-01,FinTech,US,organic,10,0\n"
        "cust_2,2024-01-05,DevTools,GB,partner,25,0\n"
        "cust_3,2024-01-10,Healthcare,DE,paid_ad,5,1\n"
        "cust_4,2024-01-15,E-commerce,US,referral,50,0\n"
    )

    # 2. Subscriptions
    subs_csv = (
        "sub_key,cust_id,activation_date,cancellation_date,monthly_revenue,annual_revenue,seat_count,billing_cycle,auto_renewal,trial_active\n"
        "sub_1,cust_1,2024-01-01,,1500.0,18000.0,10,monthly,1,0\n"
        "sub_2,cust_2,2024-01-05,,3500.0,42000.0,25,annual,1,0\n"
        "sub_3,cust_3,2024-01-10,2024-06-15,800.0,9600.0,5,monthly,0,0\n"
        "sub_4,cust_4,2024-01-15,,6000.0,72000.0,50,monthly,1,0\n"
    )

    # 3. Feature Usage
    usage_rows = [
        "event_id,sub_key,cust_id,timestamp,feature,duration,is_err,beta_flag",
        "u1,sub_1,cust_1,2024-02-15 10:00:00,dashboard,120.0,0,0",
        "u2,sub_1,cust_1,2024-03-10 14:30:00,reports,350.0,0,1",
        "u3,sub_2,cust_2,2024-02-01 09:15:00,export,45.0,1,0",
        "u4,sub_3,cust_3,2024-02-20 16:45:00,api_call,15.0,0,0",
        "u5,sub_4,cust_4,2024-03-01 11:20:00,dashboard,200.0,0,0",
    ]
    usage_csv = "\n".join(usage_rows) + "\n"

    # 4. Support Tickets
    tickets_rows = [
        "ticket_code,cust_id,open_time,close_time,urgency,csat",
        "t1,cust_1,2024-02-10 10:00:00,2024-02-11 12:00:00,low,5.0",
        "t2,cust_2,2024-02-18 14:00:00,,high,",
        "t3,cust_3,2024-03-05 08:30:00,2024-03-06 09:00:00,urgent,2.0",
        "t4,cust_4,2024-03-12 11:00:00,2024-03-13 15:00:00,low,4.0",
    ]
    tickets_csv = "\n".join(tickets_rows) + "\n"

    # 5. Churn Events
    churn_csv = (
        "event_uuid,cust_id,effective_date,churn_classification\n"
        "c1,cust_3,2024-06-15,cancellation\n"
    )

    return {
        "accounts.csv": accounts_csv,
        "subscriptions.csv": subs_csv,
        "feature_usage.csv": usage_csv,
        "support_tickets.csv": tickets_csv,
        "churn_events.csv": churn_csv,
    }


def _get_compatible_mapping() -> SchemaMappingRequest:
    """Return matching SchemaMappingRequest for synthetic compatible CSVs."""
    return SchemaMappingRequest(
        accounts=AccountConceptMapping(
            account_id="cust_id",
            signup_date="created_date",
            industry="vertical",
            country="hq_country",
            referral_source="acquisition_channel",
            initial_seats="licensed_seats",
            is_trial="is_trial_flag",
        ),
        subscriptions=SubscriptionConceptMapping(
            subscription_id="sub_key",
            account_id="cust_id",
            start_date="activation_date",
            end_date="cancellation_date",
            mrr="monthly_revenue",
            arr="annual_revenue",
            seats="seat_count",
            billing_frequency="billing_cycle",
            auto_renew="auto_renewal",
            is_trial="trial_active",
        ),
        feature_usage=FeatureUsageConceptMapping(
            usage_id="event_id",
            subscription_id="sub_key",
            account_id="cust_id",
            event_time="timestamp",
            feature_name="feature",
            duration_secs="duration",
            is_error="is_err",
            is_beta="beta_flag",
        ),
        support_tickets=SupportTicketsConceptMapping(
            ticket_id="ticket_code",
            account_id="cust_id",
            created_at="open_time",
            closed_at="close_time",
            priority="urgency",
            satisfaction_score="csat",
        ),
        churn_events=ChurnEventsConceptMapping(
            churn_event_id="event_uuid",
            account_id="cust_id",
            churn_date="effective_date",
            churn_type="churn_classification",
        ),
        prediction_window_days=60,
    )


class TestDatasetProcessingExecution(unittest.TestCase):
    """Unit tests for SchemaTransformer and DatasetProcessingEngine."""

    def setUp(self) -> None:
        self.transformer = SchemaTransformer()
        self.processor = DatasetProcessingEngine()
        self.service = get_onboarding_service()

    def test_end_to_end_compatible_pipeline_success(self) -> None:
        """Verify full end-to-end execution with compatible 5-concept synthetic dataset."""
        csv_fixtures = _build_synthetic_compatible_csvs()
        raw_dfs = {k: pd.read_csv(pd.io.common.StringIO(v)) for k, v in csv_fixtures.items()}
        mapping = _get_compatible_mapping()

        # Step 1: Transform
        canonical = self.transformer.transform(raw_dfs, mapping)
        self.assertEqual(len(canonical.accounts_df), 4)
        self.assertEqual(len(canonical.subscriptions_df), 4)
        self.assertEqual(len(canonical.churn_events_df), 1)
        self.assertEqual(canonical.rejection_summary.total_rejected, 0)
        self.assertGreaterEqual(canonical.span_days, 150)

        # Step 2: Process
        options = ProcessingOptions(
            prediction_window_days=60,
            observation_dates=["2024-04-01"],
            allow_sentinel_scoring=False,
        )
        response = self.processor.process(canonical, options)

        self.assertEqual(response.processing_status, "SUCCESS")
        self.assertTrue(response.analysis_ready)
        self.assertTrue(response.label_construction_ready)
        self.assertTrue(response.existing_model_scoring_ready)
        self.assertEqual(len(response.scoring_blockers), 0)

        # Check snapshot summary
        self.assertIsNotNone(response.snapshot_summary)
        assert response.snapshot_summary is not None
        self.assertEqual(response.snapshot_summary.total_snapshots, 4)
        self.assertEqual(response.snapshot_summary.eligible_accounts_count, 4)
        # cust_3 churned on 2024-06-15, which is within (2024-04-01, 2024-04-01 + 60d = 2024-05-31]?
        # 60 days from 2024-04-01 is 2024-05-31. 2024-06-15 is 75 days out.
        # So at 2024-04-01, positive targets = 0.
        self.assertIsNotNone(response.snapshot_summary.negative_targets_count)

        # Check scoring summary
        self.assertIsNotNone(response.scoring_summary)
        assert response.scoring_summary is not None
        self.assertEqual(response.scoring_summary.scored_snapshots_count, 4)
        self.assertEqual(response.scoring_summary.model_version, "1.0.0")
        self.assertGreaterEqual(response.scoring_summary.average_model_score, 0.0)
        self.assertLessEqual(response.scoring_summary.average_model_score, 1.0)
        self.assertTrue(len(response.scoring_summary.prioritization_quadrants) > 0)
        self.assertIn(
            "Model Score Interpretation", response.scoring_summary.score_interpretation_notice
        )

    def test_incompatible_dataset_scoring_blocked_without_fabrication(self) -> None:
        """Verify that when usage & support are missing, scoring is blocked and no features are fabricated."""
        csv_fixtures = _build_synthetic_compatible_csvs()
        # Omit feature usage and support tickets
        raw_dfs = {
            "accounts": pd.read_csv(pd.io.common.StringIO(csv_fixtures["accounts.csv"])),
            "subscriptions": pd.read_csv(pd.io.common.StringIO(csv_fixtures["subscriptions.csv"])),
            "churn_events": pd.read_csv(pd.io.common.StringIO(csv_fixtures["churn_events.csv"])),
        }
        mapping = SchemaMappingRequest(
            accounts=AccountConceptMapping(account_id="cust_id", signup_date="created_date"),
            subscriptions=SubscriptionConceptMapping(
                subscription_id="sub_key",
                account_id="cust_id",
                start_date="activation_date",
                end_date="cancellation_date",
                mrr="monthly_revenue",
            ),
            churn_events=ChurnEventsConceptMapping(
                account_id="cust_id",
                churn_date="effective_date",
            ),
            feature_usage=None,
            support_tickets=None,
        )

        canonical = self.transformer.transform(raw_dfs, mapping)
        options = ProcessingOptions(
            prediction_window_days=60,
            observation_dates=["2024-04-01"],
            allow_sentinel_scoring=False,
        )
        response = self.processor.process(canonical, options)

        # Must report PARTIAL_SUCCESS (snapshots generated, but scoring blocked)
        self.assertEqual(response.processing_status, "PARTIAL_SUCCESS")
        self.assertTrue(response.analysis_ready)
        self.assertTrue(response.label_construction_ready)
        self.assertFalse(response.existing_model_scoring_ready)
        self.assertTrue(len(response.scoring_blockers) > 0)
        self.assertIn("feature_usage", response.scoring_blockers[0])
        # Scoring summary must NOT be fabricated
        self.assertIsNone(response.scoring_summary)
        # Snapshots must still be generated
        self.assertIsNotNone(response.snapshot_summary)

    def test_insufficient_historical_coverage_blocks_target_labeling(self) -> None:
        """Verify dataset with <60d span blocks target construction."""
        acc_csv = "acc_id,signup\na1,2025-01-01\na2,2025-01-05\n"
        sub_csv = "sub_id,acc_id,start,mrr\ns1,a1,2025-01-01,100.0\ns2,a2,2025-01-05,200.0\n"

        raw_dfs = {
            "accounts": pd.read_csv(pd.io.common.StringIO(acc_csv)),
            "subscriptions": pd.read_csv(pd.io.common.StringIO(sub_csv)),
        }
        mapping = SchemaMappingRequest(
            accounts=AccountConceptMapping(account_id="acc_id", signup_date="signup"),
            subscriptions=SubscriptionConceptMapping(
                subscription_id="sub_id",
                account_id="acc_id",
                start_date="start",
                mrr="mrr",
            ),
            prediction_window_days=60,
        )

        canonical = self.transformer.transform(raw_dfs, mapping)
        self.assertLess(canonical.span_days, 60)

        response = self.processor.process(canonical)
        self.assertTrue(response.analysis_ready)
        self.assertFalse(response.label_construction_ready)
        self.assertFalse(response.existing_model_scoring_ready)
        self.assertIn("No churn ground truth", response.scoring_blockers[0])

    def test_referential_integrity_and_orphan_quarantine(self) -> None:
        """Verify orphan subscriptions referencing non-existent accounts are quarantined."""
        acc_csv = "acc_id,signup\na1,2025-01-01\na2,2025-01-05\n"
        sub_csv = (
            "sub_id,acc_id,start,mrr\n"
            "s1,a1,2025-01-01,100.0\n"
            "s2,a2,2025-01-05,200.0\n"
            "s3,ghost_account_99,2025-01-10,500.0\n"  # Orphan
        )

        raw_dfs = {
            "accounts": pd.read_csv(pd.io.common.StringIO(acc_csv)),
            "subscriptions": pd.read_csv(pd.io.common.StringIO(sub_csv)),
        }
        mapping = SchemaMappingRequest(
            accounts=AccountConceptMapping(account_id="acc_id", signup_date="signup"),
            subscriptions=SubscriptionConceptMapping(
                subscription_id="sub_id",
                account_id="acc_id",
                start_date="start",
                mrr="mrr",
            ),
        )

        canonical = self.transformer.transform(raw_dfs, mapping)
        self.assertEqual(len(canonical.subscriptions_df), 2)  # Clean records
        self.assertEqual(canonical.rejection_summary.rejected_by_table["subscriptions"], 1)
        self.assertTrue(any("orphaned" in r for r in canonical.rejection_summary.reasons))

    def test_anti_leakage_safeguards_in_features(self) -> None:
        """Confirm that future events occurring after observation date do NOT leak into features."""
        acc_df = pd.DataFrame(
            {
                "account_id": ["acc_leak"],
                "signup_date": ["2024-01-01"],
                "industry": ["Tech"],
                "country": ["US"],
                "referral_source": ["organic"],
                "seats": [5],
                "is_trial": [0],
            }
        )
        sub_df = pd.DataFrame(
            {
                "subscription_id": ["sub_leak"],
                "account_id": ["acc_leak"],
                "start_date": ["2024-01-01"],
                "end_date": [None],
                "mrr_amount": [1000.0],
                "arr_amount": [12000.0],
                "seats": [5],
                "billing_frequency": ["monthly"],
                "auto_renew_flag": [True],
                "is_trial": [False],
                "upgrade_flag": [False],
                "downgrade_flag": [False],
            }
        )
        # Usage on or before snapshot date (2024-03-01) vs future usage (2024-03-15)
        usage_df = pd.DataFrame(
            {
                "usage_id": ["u1", "u2"],
                "subscription_id": ["sub_leak", "sub_leak"],
                "account_id": ["acc_leak", "acc_leak"],
                "usage_date": [
                    "2024-02-15",
                    "2024-03-15",
                ],  # u2 is in future relative to 2024-03-01
                "feature_name": ["f1", "f2"],
                "usage_count": [10, 50],
                "usage_duration_secs": [100.0, 500.0],
                "error_count": [0, 0],
                "active_users": [1, 1],
                "is_beta_feature": [False, False],
            }
        )
        tickets_df = pd.DataFrame(
            columns=[
                "ticket_id",
                "account_id",
                "submitted_at",
                "closed_at",
                "priority",
                "satisfaction_score",
                "escalation_flag",
                "first_response_time_minutes",
            ]
        )
        churn_df = pd.DataFrame(
            columns=["churn_event_id", "account_id", "churn_date", "refund_amount_usd"]
        )

        pipeline = FeaturePipeline(
            accounts_df=acc_df,
            subscriptions_df=sub_df,
            usage_df=usage_df,
            tickets_df=tickets_df,
            churn_events_df=churn_df,
        )

        obs_date = date(2024, 3, 1)
        features = pipeline.extract_snapshot_features("acc_leak", obs_date)

        # Must include u1 (count 10) but NOT u2 (count 50)
        self.assertEqual(features["usage_count_30d"], 10)


class TestProcessingEndpoints(unittest.TestCase):
    """Integration tests for versioned preview and process API endpoints."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        settings = get_settings()
        settings.SUPABASE_JWT_SECRET = TEST_SECRET
        settings.SUPABASE_JWT_ALGORITHM = "HS256"
        settings.SUPABASE_JWT_AUDIENCE = "authenticated"
        settings.SUPABASE_JWT_ISSUER = None

    def test_preview_endpoint_unauthorized(self) -> None:
        """Verify preview endpoint returns 401 when unauthenticated."""
        resp = self.client.post("/api/v1/onboarding/preview")
        self.assertEqual(resp.status_code, 401)

    def test_preview_endpoint_valid_request(self) -> None:
        """Verify preview endpoint runs dry-run transformation and returns 200."""
        token = _generate_valid_jwt()
        fixtures = _build_synthetic_compatible_csvs()
        mapping = _get_compatible_mapping()

        files = [
            ("files", ("accounts.csv", fixtures["accounts.csv"].encode("utf-8"), "text/csv")),
            (
                "files",
                ("subscriptions.csv", fixtures["subscriptions.csv"].encode("utf-8"), "text/csv"),
            ),
            (
                "files",
                ("feature_usage.csv", fixtures["feature_usage.csv"].encode("utf-8"), "text/csv"),
            ),
            (
                "files",
                (
                    "support_tickets.csv",
                    fixtures["support_tickets.csv"].encode("utf-8"),
                    "text/csv",
                ),
            ),
            (
                "files",
                ("churn_events.csv", fixtures["churn_events.csv"].encode("utf-8"), "text/csv"),
            ),
        ]
        data = {"mapping": mapping.model_dump_json()}
        headers = {"Authorization": f"Bearer {token}"}

        resp = self.client.post(
            "/api/v1/onboarding/preview", files=files, data=data, headers=headers
        )
        self.assertEqual(resp.status_code, 200)

        body = resp.json()
        self.assertTrue(body["analysis_ready"])
        self.assertTrue(body["label_construction_ready"])
        self.assertTrue(body["existing_model_scoring_ready"])
        self.assertGreater(body["total_input_rows"], 0)

    def test_process_endpoint_unauthorized(self) -> None:
        """Verify process endpoint returns 401 when unauthenticated."""
        resp = self.client.post("/api/v1/onboarding/process")
        self.assertEqual(resp.status_code, 401)

    def test_process_endpoint_compatible_success(self) -> None:
        """Verify process endpoint executes full pipeline and returns predictions and quadrants."""
        token = _generate_valid_jwt()
        fixtures = _build_synthetic_compatible_csvs()
        mapping = _get_compatible_mapping()
        opts = ProcessingOptions(
            prediction_window_days=60,
            observation_dates=["2024-04-01"],
            allow_sentinel_scoring=False,
        )

        files = [
            ("files", ("accounts.csv", fixtures["accounts.csv"].encode("utf-8"), "text/csv")),
            (
                "files",
                ("subscriptions.csv", fixtures["subscriptions.csv"].encode("utf-8"), "text/csv"),
            ),
            (
                "files",
                ("feature_usage.csv", fixtures["feature_usage.csv"].encode("utf-8"), "text/csv"),
            ),
            (
                "files",
                (
                    "support_tickets.csv",
                    fixtures["support_tickets.csv"].encode("utf-8"),
                    "text/csv",
                ),
            ),
            (
                "files",
                ("churn_events.csv", fixtures["churn_events.csv"].encode("utf-8"), "text/csv"),
            ),
        ]
        data = {
            "mapping": mapping.model_dump_json(),
            "options": opts.model_dump_json(),
        }
        headers = {"Authorization": f"Bearer {token}"}

        resp = self.client.post(
            "/api/v1/onboarding/process", files=files, data=data, headers=headers
        )
        self.assertEqual(resp.status_code, 200)

        body = resp.json()
        self.assertEqual(body["processing_status"], "SUCCESS")
        self.assertTrue(body["analysis_ready"])
        self.assertTrue(body["label_construction_ready"])
        self.assertTrue(body["existing_model_scoring_ready"])
        self.assertIsNotNone(body["snapshot_summary"])
        self.assertIsNotNone(body["scoring_summary"])
        self.assertEqual(body["scoring_summary"]["model_version"], "1.0.0")

    def test_process_endpoint_malformed_json_mapping(self) -> None:
        """Verify malformed JSON mapping returns 422 Unprocessable Entity."""
        token = _generate_valid_jwt()
        files = [
            ("files", ("accounts.csv", b"account_id,signup\na1,2025-01-01\n", "text/csv")),
        ]
        data = {"mapping": "invalid_not_json"}
        headers = {"Authorization": f"Bearer {token}"}

        resp = self.client.post(
            "/api/v1/onboarding/process", files=files, data=data, headers=headers
        )
        self.assertEqual(resp.status_code, 422)
        self.assertIn("Malformed schema mapping JSON", resp.json()["detail"])


if __name__ == "__main__":
    unittest.main()
