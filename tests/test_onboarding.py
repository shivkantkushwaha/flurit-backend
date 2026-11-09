"""Unit and Integration Tests for Flurit Dataset Onboarding and Schema Mapping.

Tests CSV profiling, memory safety, encoding fallbacks, PII redaction,
concept schema mapping, feature coverage, temporal feasibility checks,
Supabase JWT auth, and versioned onboarding endpoints.
"""

import time
import unittest
from typing import Any

import pandas as pd
from starlette.testclient import TestClient

from app.core.auth import create_jwt_token
from app.core.config import get_settings
from app.main import app
from app.onboarding.mapper import SchemaMapper
from app.onboarding.profiler import CSVProfiler, sanitize_filename
from app.onboarding.schemas import (
    AccountConceptMapping,
    ChurnEventsConceptMapping,
    ColumnProfile,
    DatasetProfile,
    DateRange,
    FeatureUsageConceptMapping,
    InferredDataType,
    ReadinessStatus,
    SchemaMappingRequest,
    SubscriptionConceptMapping,
    SupportTicketsConceptMapping,
)

TEST_SECRET = "flurit-secure-test-jwt-secret-key-32-chars-min!"


def _generate_valid_jwt(user_id: str = "onboarding-user-456") -> str:
    """Generate a valid signed Supabase JWT token for authenticated testing."""
    payload = {
        "sub": user_id,
        "email": "analyst@example.com",
        "role": "authenticated",
        "aud": "authenticated",
        "exp": int(time.time()) + 3600,
        "iat": int(time.time()),
    }
    return create_jwt_token(payload, secret=TEST_SECRET)


class TestCSVProfiler(unittest.TestCase):
    """Unit tests for CSVProfiler."""

    def setUp(self) -> None:
        self.profiler = CSVProfiler(
            max_file_size_bytes=10 * 1024 * 1024,  # 10 MB for test
            max_row_count=10_000,
            max_column_count=50,
        )

    def test_sanitize_filename(self) -> None:
        """Verify filename sanitization prevents directory traversal and injection."""
        self.assertEqual(sanitize_filename("../../etc/passwd"), "passwd.csv")
        self.assertEqual(sanitize_filename("C:\\Users\\test\\data.csv"), "data.csv")
        self.assertEqual(sanitize_filename("valid_name.csv"), "valid_name.csv")
        self.assertTrue(sanitize_filename("").endswith(".csv"))

    def test_profile_valid_csv(self) -> None:
        """Verify profiling computes accurate types, counts, nulls, and candidate roles."""
        csv_content = (
            b"account_id,signup_date,industry,mrr,is_trial,churn_status\n"
            b"acc_1,2025-01-01,FinTech,500.0,0,0\n"
            b"acc_2,2025-01-15,Healthcare,1200.5,1,1\n"
            b"acc_3,2025-02-01,DevTools,,0,0\n"
            b"acc_4,2025-02-20,FinTech,850.0,0,0\n"
        )

        profile = self.profiler.profile_bytes(csv_content, "accounts.csv")

        self.assertEqual(profile.file_name, "accounts.csv")
        self.assertEqual(profile.row_count, 4)
        self.assertEqual(profile.column_count, 6)
        self.assertEqual(profile.duplicate_row_count, 0)
        self.assertIn("account_id", profile.candidate_id_columns)
        self.assertIn("churn_status", profile.candidate_target_columns)

        col_dict = {c.column_name: c for c in profile.columns}

        # Check account_id
        acc_col = col_dict["account_id"]
        self.assertTrue(acc_col.is_candidate_id)
        self.assertEqual(acc_col.null_count, 0)
        self.assertEqual(acc_col.unique_count, 4)

        # Check signup_date
        date_col = col_dict["signup_date"]
        self.assertEqual(date_col.inferred_type, InferredDataType.DATETIME)
        self.assertIsNotNone(date_col.date_range)
        assert date_col.date_range is not None
        self.assertEqual(date_col.date_range.min_date, "2025-01-01")
        self.assertEqual(date_col.date_range.max_date, "2025-02-20")

        # Check mrr
        mrr_col = col_dict["mrr"]
        self.assertEqual(mrr_col.inferred_type, InferredDataType.FLOAT)
        self.assertEqual(mrr_col.null_count, 1)
        self.assertEqual(mrr_col.null_percentage, 25.0)

        # Check is_trial
        trial_col = col_dict["is_trial"]
        self.assertEqual(trial_col.inferred_type, InferredDataType.BOOLEAN)

        # Check global date range
        self.assertIsNotNone(profile.date_range)
        assert profile.date_range is not None
        self.assertEqual(profile.date_range.span_days, 50)

    def test_file_size_limit_exceeded(self) -> None:
        """Ensure uploads exceeding size limit are rejected with clear error."""
        profiler_small = CSVProfiler(max_file_size_bytes=50)
        data = b"col1,col2\n" + b"value1,value2\n" * 10
        with self.assertRaises(ValueError) as ctx:
            profiler_small.profile_bytes(data, "large.csv")
        self.assertIn("exceeds maximum allowed upload size", str(ctx.exception))

    def test_empty_file_rejected(self) -> None:
        """Ensure 0-byte or empty uploads are rejected."""
        with self.assertRaises(ValueError) as ctx:
            self.profiler.profile_bytes(b"", "empty.csv")
        self.assertIn("is empty", str(ctx.exception))

    def test_row_and_column_limits_enforced(self) -> None:
        """Verify row and column count boundaries are enforced."""
        profiler_strict = CSVProfiler(max_row_count=5, max_column_count=3)

        # Exceed row count
        df_rows = pd.DataFrame({"a": range(10), "b": range(10)})
        with self.assertRaises(ValueError) as ctx:
            profiler_strict.profile_dataframe(df_rows, "test.csv")
        self.assertIn("exceeding the limit of 5 rows", str(ctx.exception))

        # Exceed column count
        df_cols = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
        with self.assertRaises(ValueError) as ctx:
            profiler_strict.profile_dataframe(df_cols, "test.csv")
        self.assertIn("exceeding the limit of 3 columns", str(ctx.exception))

    def test_encoding_fallbacks(self) -> None:
        """Ensure UTF-8 with BOM and Latin-1 encodings are parsed transparently."""
        # UTF-8 with BOM
        bom_bytes = "\ufeffaccount_id,name\nacc_1,Acme Corp\n".encode("utf-8-sig")
        p1 = self.profiler.profile_bytes(bom_bytes, "bom.csv")
        self.assertEqual(p1.row_count, 1)

        # Latin-1 with accented characters
        latin1_bytes = "account_id,city\nacc_1,München\n".encode("latin-1")
        p2 = self.profiler.profile_bytes(latin1_bytes, "latin.csv")
        self.assertEqual(p2.row_count, 1)

    def test_pii_redaction_in_sample_categories(self) -> None:
        """Verify emails in categorical columns are redacted from sample categories."""
        csv_bytes = (
            b"role,user_email\n"
            b"admin,alice@example.com\n"
            b"editor,bob@example.com\n"
            b"admin,carol@example.com\n"
        )
        profile = self.profiler.profile_bytes(csv_bytes, "users.csv")
        col_dict = {c.column_name: c for c in profile.columns}

        email_col = col_dict["user_email"]
        if email_col.sample_categories:
            for val in email_col.sample_categories:
                self.assertNotIn("@example.com", val)
                self.assertEqual(val, "[REDACTED_EMAIL]")

    def test_duplicate_row_detection(self) -> None:
        """Verify duplicate rows are detected and quantified."""
        csv_bytes = b"id,val\n1,a\n2,b\n1,a\n"
        profile = self.profiler.profile_bytes(csv_bytes, "dups.csv")
        self.assertEqual(profile.duplicate_row_count, 1)


class TestSchemaMapper(unittest.TestCase):
    """Unit tests for SchemaMapper."""

    def setUp(self) -> None:
        self.mapper = SchemaMapper()

    def _create_mock_profiles(self) -> list[DatasetProfile]:
        """Helper to create sample dataset profiles with 200 days span."""
        dr = DateRange(min_date="2025-01-01", max_date="2025-08-01", span_days=212)
        cols_acc = [
            ColumnProfile(
                column_name="account_id",
                inferred_type=InferredDataType.TEXT,
                total_count=100,
                null_count=0,
                null_percentage=0.0,
                unique_count=100,
                unique_ratio=1.0,
                is_candidate_id=True,
            ),
            ColumnProfile(
                column_name="signup_date",
                inferred_type=InferredDataType.DATETIME,
                total_count=100,
                null_count=0,
                null_percentage=0.0,
                unique_count=90,
                unique_ratio=0.9,
                is_candidate_id=False,
                date_range=dr,
            ),
            ColumnProfile(
                column_name="industry",
                inferred_type=InferredDataType.CATEGORY,
                total_count=100,
                null_count=0,
                null_percentage=0.0,
                unique_count=5,
                unique_ratio=0.05,
                is_candidate_id=False,
            ),
            ColumnProfile(
                column_name="country",
                inferred_type=InferredDataType.CATEGORY,
                total_count=100,
                null_count=0,
                null_percentage=0.0,
                unique_count=10,
                unique_ratio=0.1,
                is_candidate_id=False,
            ),
        ]
        cols_sub = [
            ColumnProfile(
                column_name="subscription_id",
                inferred_type=InferredDataType.TEXT,
                total_count=120,
                null_count=0,
                null_percentage=0.0,
                unique_count=120,
                unique_ratio=1.0,
                is_candidate_id=True,
            ),
            ColumnProfile(
                column_name="account_id",
                inferred_type=InferredDataType.TEXT,
                total_count=120,
                null_count=0,
                null_percentage=0.0,
                unique_count=100,
                unique_ratio=0.83,
                is_candidate_id=False,
            ),
            ColumnProfile(
                column_name="start_date",
                inferred_type=InferredDataType.DATETIME,
                total_count=120,
                null_count=0,
                null_percentage=0.0,
                unique_count=100,
                unique_ratio=0.83,
                is_candidate_id=False,
                date_range=dr,
            ),
            ColumnProfile(
                column_name="end_date",
                inferred_type=InferredDataType.DATETIME,
                total_count=120,
                null_count=80,
                null_percentage=66.7,
                unique_count=40,
                unique_ratio=0.33,
                is_candidate_id=False,
                date_range=dr,
            ),
            ColumnProfile(
                column_name="mrr",
                inferred_type=InferredDataType.FLOAT,
                total_count=120,
                null_count=0,
                null_percentage=0.0,
                unique_count=50,
                unique_ratio=0.42,
                is_candidate_id=False,
            ),
        ]
        cols_usage = [
            ColumnProfile(
                column_name="event_time",
                inferred_type=InferredDataType.DATETIME,
                total_count=1000,
                null_count=0,
                null_percentage=0.0,
                unique_count=900,
                unique_ratio=0.9,
                is_candidate_id=False,
                date_range=dr,
            ),
            ColumnProfile(
                column_name="account_id",
                inferred_type=InferredDataType.TEXT,
                total_count=1000,
                null_count=0,
                null_percentage=0.0,
                unique_count=100,
                unique_ratio=0.1,
                is_candidate_id=False,
            ),
        ]
        cols_support = [
            ColumnProfile(
                column_name="ticket_id",
                inferred_type=InferredDataType.TEXT,
                total_count=50,
                null_count=0,
                null_percentage=0.0,
                unique_count=50,
                unique_ratio=1.0,
                is_candidate_id=True,
            ),
            ColumnProfile(
                column_name="account_id",
                inferred_type=InferredDataType.TEXT,
                total_count=50,
                null_count=0,
                null_percentage=0.0,
                unique_count=40,
                unique_ratio=0.8,
                is_candidate_id=False,
            ),
            ColumnProfile(
                column_name="created_at",
                inferred_type=InferredDataType.DATETIME,
                total_count=50,
                null_count=0,
                null_percentage=0.0,
                unique_count=50,
                unique_ratio=1.0,
                is_candidate_id=False,
                date_range=dr,
            ),
        ]
        cols_churn = [
            ColumnProfile(
                column_name="account_id",
                inferred_type=InferredDataType.TEXT,
                total_count=20,
                null_count=0,
                null_percentage=0.0,
                unique_count=20,
                unique_ratio=1.0,
                is_candidate_id=False,
            ),
            ColumnProfile(
                column_name="churn_date",
                inferred_type=InferredDataType.DATETIME,
                total_count=20,
                null_count=0,
                null_percentage=0.0,
                unique_count=20,
                unique_ratio=1.0,
                is_candidate_id=False,
                date_range=dr,
            ),
        ]

        return [
            DatasetProfile(
                file_name="accounts.csv",
                file_size_bytes=5000,
                row_count=100,
                column_count=len(cols_acc),
                duplicate_row_count=0,
                candidate_id_columns=["account_id"],
                candidate_target_columns=[],
                columns=cols_acc,
                date_range=dr,
            ),
            DatasetProfile(
                file_name="subscriptions.csv",
                file_size_bytes=8000,
                row_count=120,
                column_count=len(cols_sub),
                duplicate_row_count=0,
                candidate_id_columns=["subscription_id"],
                candidate_target_columns=[],
                columns=cols_sub,
                date_range=dr,
            ),
            DatasetProfile(
                file_name="feature_usage.csv",
                file_size_bytes=30000,
                row_count=1000,
                column_count=len(cols_usage),
                duplicate_row_count=0,
                candidate_id_columns=[],
                candidate_target_columns=[],
                columns=cols_usage,
                date_range=dr,
            ),
            DatasetProfile(
                file_name="support_tickets.csv",
                file_size_bytes=4000,
                row_count=50,
                column_count=len(cols_support),
                duplicate_row_count=0,
                candidate_id_columns=["ticket_id"],
                candidate_target_columns=[],
                columns=cols_support,
                date_range=dr,
            ),
            DatasetProfile(
                file_name="churn_events.csv",
                file_size_bytes=2000,
                row_count=20,
                column_count=len(cols_churn),
                duplicate_row_count=0,
                candidate_id_columns=[],
                candidate_target_columns=["churn_date"],
                columns=cols_churn,
                date_range=dr,
            ),
        ]

    def test_full_mapping_readiness_ready(self) -> None:
        """Verify complete 5-concept mapping with >150d span returns READY across all dimensions."""
        profiles = self._create_mock_profiles()
        req = SchemaMappingRequest(
            accounts=AccountConceptMapping(
                account_id="account_id",
                signup_date="signup_date",
                industry="industry",
                country="country",
            ),
            subscriptions=SubscriptionConceptMapping(
                subscription_id="subscription_id",
                account_id="account_id",
                start_date="start_date",
                end_date="end_date",
                mrr="mrr",
            ),
            feature_usage=FeatureUsageConceptMapping(
                event_time="event_time",
                account_id="account_id",
            ),
            support_tickets=SupportTicketsConceptMapping(
                ticket_id="ticket_id",
                account_id="account_id",
                created_at="created_at",
            ),
            churn_events=ChurnEventsConceptMapping(
                account_id="account_id",
                churn_date="churn_date",
            ),
            dataset_profiles=profiles,
        )

        resp = self.mapper.assess_readiness(req)

        self.assertEqual(resp.overall_status, ReadinessStatus.READY)
        self.assertEqual(resp.target_readiness, ReadinessStatus.READY)
        self.assertEqual(resp.scoring_readiness, ReadinessStatus.READY)
        self.assertEqual(resp.feature_coverage.covered_count, 45)
        self.assertEqual(resp.feature_coverage.coverage_pct, 100.0)
        self.assertEqual(len(resp.validation_issues), 0)
        self.assertTrue(len(resp.next_steps) > 0)

    def test_missing_mandatory_account_concept(self) -> None:
        """Ensure missing accounts concept triggers UNSUPPORTED overall and target status."""
        req = SchemaMappingRequest(
            accounts=None,
            subscriptions=SubscriptionConceptMapping(
                subscription_id="sub_id",
                account_id="acc_id",
                start_date="start_date",
            ),
        )
        resp = self.mapper.assess_readiness(req)
        self.assertEqual(resp.overall_status, ReadinessStatus.UNSUPPORTED)
        self.assertEqual(resp.target_readiness, ReadinessStatus.UNSUPPORTED)
        self.assertEqual(resp.scoring_readiness, ReadinessStatus.UNSUPPORTED)
        self.assertTrue(any(i.code == "MISSING_CONCEPT_MAPPING" for i in resp.validation_issues))

    def test_missing_churn_events_and_end_date(self) -> None:
        """Ensure missing churn events without subscription end date is UNSUPPORTED."""
        req = SchemaMappingRequest(
            accounts=AccountConceptMapping(account_id="acc_id", signup_date="signup_date"),
            subscriptions=SubscriptionConceptMapping(
                subscription_id="sub_id",
                account_id="acc_id",
                start_date="start_date",
                end_date=None,
            ),
            churn_events=None,
        )
        resp = self.mapper.assess_readiness(req)
        self.assertEqual(resp.target_readiness, ReadinessStatus.UNSUPPORTED)
        self.assertTrue(any(i.code == "NO_CHURN_GROUND_TRUTH" for i in resp.validation_issues))

    def test_churn_derived_from_subscription_end_date(self) -> None:
        """Ensure subscriptions.end_date serves as valid alternative ground truth."""
        req = SchemaMappingRequest(
            accounts=AccountConceptMapping(account_id="acc_id", signup_date="signup_date"),
            subscriptions=SubscriptionConceptMapping(
                subscription_id="sub_id",
                account_id="acc_id",
                start_date="start_date",
                end_date="end_date",
            ),
            churn_events=None,
        )
        resp = self.mapper.assess_readiness(req)
        # Target readiness can be READY or PARTIALLY_READY, not UNSUPPORTED
        self.assertIn(
            resp.target_readiness, [ReadinessStatus.READY, ReadinessStatus.PARTIALLY_READY]
        )
        self.assertTrue(any(i.code == "DERIVED_CHURN_GROUND_TRUTH" for i in resp.validation_issues))

    def test_insufficient_date_span(self) -> None:
        """Ensure span < prediction_window_days marks target readiness as UNSUPPORTED."""
        short_dr = DateRange(min_date="2025-01-01", max_date="2025-01-20", span_days=19)
        profile = DatasetProfile(
            file_name="data.csv",
            file_size_bytes=1000,
            row_count=10,
            column_count=3,
            duplicate_row_count=0,
            columns=[],
            date_range=short_dr,
        )
        req = SchemaMappingRequest(
            accounts=AccountConceptMapping(account_id="acc_id", signup_date="signup_date"),
            subscriptions=SubscriptionConceptMapping(
                subscription_id="sub_id",
                account_id="acc_id",
                start_date="start_date",
                end_date="end_date",
            ),
            dataset_profiles=[profile],
            prediction_window_days=60,
        )
        resp = self.mapper.assess_readiness(req)
        self.assertEqual(resp.target_readiness, ReadinessStatus.UNSUPPORTED)
        self.assertTrue(any(i.code == "INSUFFICIENT_TEMPORAL_SPAN" for i in resp.validation_issues))

    def test_limited_lookback_window_warning(self) -> None:
        """Ensure date span between 60d and 150d marks target readiness as PARTIALLY_READY."""
        med_dr = DateRange(min_date="2025-01-01", max_date="2025-03-20", span_days=78)
        profile = DatasetProfile(
            file_name="data.csv",
            file_size_bytes=1000,
            row_count=10,
            column_count=3,
            duplicate_row_count=0,
            columns=[],
            date_range=med_dr,
        )
        req = SchemaMappingRequest(
            accounts=AccountConceptMapping(account_id="acc_id", signup_date="signup_date"),
            subscriptions=SubscriptionConceptMapping(
                subscription_id="sub_id",
                account_id="acc_id",
                start_date="start_date",
                end_date="end_date",
            ),
            dataset_profiles=[profile],
            prediction_window_days=60,
        )
        resp = self.mapper.assess_readiness(req)
        self.assertEqual(resp.target_readiness, ReadinessStatus.PARTIALLY_READY)
        self.assertTrue(any(i.code == "LIMITED_LOOKBACK_WINDOW" for i in resp.validation_issues))

    def test_partial_scoring_readiness_with_missing_usage_and_support(self) -> None:
        """Ensure missing usage and support leads to PARTIALLY_READY scoring and sentinel notice."""
        req = SchemaMappingRequest(
            accounts=AccountConceptMapping(account_id="acc_id", signup_date="signup_date"),
            subscriptions=SubscriptionConceptMapping(
                subscription_id="sub_id",
                account_id="acc_id",
                start_date="start_date",
                end_date="end_date",
            ),
            feature_usage=None,
            support_tickets=None,
        )
        resp = self.mapper.assess_readiness(req)
        self.assertEqual(resp.scoring_readiness, ReadinessStatus.PARTIALLY_READY)
        self.assertEqual(resp.feature_coverage.covered_count, 16)  # 6 account + 10 subscription
        self.assertEqual(resp.feature_coverage.missing_count, 29)
        self.assertTrue(any(i.code == "MODEL_DISTRIBUTION_SHIFT" for i in resp.validation_issues))

    def test_column_not_found_validation(self) -> None:
        """Ensure mapped column not in dataset profiles is flagged with an error."""
        profiles = self._create_mock_profiles()
        req = SchemaMappingRequest(
            accounts=AccountConceptMapping(
                account_id="non_existent_id_col",
                signup_date="signup_date",
            ),
            subscriptions=SubscriptionConceptMapping(
                subscription_id="subscription_id",
                account_id="account_id",
                start_date="start_date",
            ),
            dataset_profiles=profiles,
        )
        resp = self.mapper.assess_readiness(req)
        self.assertEqual(resp.overall_status, ReadinessStatus.UNSUPPORTED)
        self.assertTrue(any(i.code == "COLUMN_NOT_FOUND" for i in resp.validation_issues))


class TestOnboardingEndpoints(unittest.TestCase):
    """Integration tests for versioned onboarding API endpoints."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        settings = get_settings()
        settings.SUPABASE_JWT_SECRET = TEST_SECRET
        settings.SUPABASE_JWT_ALGORITHM = "HS256"
        settings.SUPABASE_JWT_AUDIENCE = "authenticated"
        settings.SUPABASE_JWT_ISSUER = None

    def test_profile_endpoint_unauthorized(self) -> None:
        """Verify endpoint rejects unauthenticated requests."""
        resp = self.client.post("/api/v1/onboarding/profile")
        self.assertEqual(resp.status_code, 401)

    def test_profile_endpoint_valid_single_file(self) -> None:
        """Verify profiling upload returns 200 with structured profile response."""
        token = _generate_valid_jwt()
        csv_data = "account_id,signup_date,mrr\nacc_1,2025-01-01,100.0\nacc_2,2025-01-02,200.0\n"

        files = [
            ("files", ("accounts.csv", csv_data.encode("utf-8"), "text/csv")),
        ]
        headers = {"Authorization": f"Bearer {token}"}

        resp = self.client.post("/api/v1/onboarding/profile", files=files, headers=headers)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(len(data["datasets"]), 1)
        dataset = data["datasets"][0]
        self.assertEqual(dataset["file_name"], "accounts.csv")
        self.assertEqual(dataset["row_count"], 2)
        self.assertEqual(dataset["column_count"], 3)
        self.assertIn("account_id", dataset["candidate_id_columns"])

    def test_profile_endpoint_multiple_files(self) -> None:
        """Verify profiling endpoint processes multiple CSV files in a single request."""
        token = _generate_valid_jwt()
        csv1 = "account_id,signup_date\nacc_1,2025-01-01\n"
        csv2 = "subscription_id,account_id,mrr\nsub_1,acc_1,500.0\n"

        files = [
            ("files", ("accounts.csv", csv1.encode("utf-8"), "text/csv")),
            ("files", ("subscriptions.csv", csv2.encode("utf-8"), "text/csv")),
        ]
        headers = {"Authorization": f"Bearer {token}"}

        resp = self.client.post("/api/v1/onboarding/profile", files=files, headers=headers)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(len(data["datasets"]), 2)
        names = [d["file_name"] for d in data["datasets"]]
        self.assertIn("accounts.csv", names)
        self.assertIn("subscriptions.csv", names)

    def test_profile_endpoint_empty_file_rejected(self) -> None:
        """Verify empty files return 400 Bad Request."""
        token = _generate_valid_jwt()
        files = [
            ("files", ("empty.csv", b"", "text/csv")),
        ]
        headers = {"Authorization": f"Bearer {token}"}

        resp = self.client.post("/api/v1/onboarding/profile", files=files, headers=headers)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("is empty", resp.json()["detail"])

    def test_map_endpoint_unauthorized(self) -> None:
        """Verify mapping endpoint rejects unauthenticated requests."""
        resp = self.client.post("/api/v1/onboarding/map", json={})
        self.assertEqual(resp.status_code, 401)

    def test_map_endpoint_valid_request(self) -> None:
        """Verify mapping endpoint evaluates readiness and returns structured report."""
        token = _generate_valid_jwt()
        headers = {"Authorization": f"Bearer {token}"}

        payload: dict[str, Any] = {
            "accounts": {
                "account_id": "account_id",
                "signup_date": "signup_date",
                "industry": "industry",
            },
            "subscriptions": {
                "subscription_id": "subscription_id",
                "account_id": "account_id",
                "start_date": "start_date",
                "end_date": "end_date",
                "mrr": "mrr",
            },
            "prediction_window_days": 60,
        }

        resp = self.client.post("/api/v1/onboarding/map", json=payload, headers=headers)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertIn(data["overall_status"], ["READY", "PARTIALLY_READY", "UNSUPPORTED"])
        self.assertIn("feature_coverage", data)
        self.assertIn("validation_issues", data)
        self.assertIn("recommendations", data)
        self.assertIn("next_steps", data)
        self.assertEqual(data["feature_coverage"]["total_features"], 45)


if __name__ == "__main__":
    unittest.main()
