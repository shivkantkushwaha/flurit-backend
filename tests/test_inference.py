"""Unit and Integration Tests for Flurit ML Inference API and Service.

Tests Supabase JWT authentication (valid, expired, tampered, missing, misconfigured),
scoring endpoint, customer risk and explainability endpoint, portfolio prioritization,
Pydantic v2 input validation, and artifact failure handling.
"""

import time
import unittest
from pathlib import Path
from typing import Any

from starlette.testclient import TestClient

from app.core.auth import create_jwt_token
from app.core.config import get_settings
from app.main import app
from app.ml.inference.schemas import (
    CanonicalFeaturesPayload,
    ScorePredictionRequest,
)
from app.ml.inference.service import ModelInferenceService

TEST_SECRET = "flurit-secure-test-jwt-secret-key-32-chars-min!"


def _get_valid_feature_payload() -> dict[str, Any]:
    """Helper returning a complete, valid dictionary of all 45 canonical features."""
    return {
        "account_age_days": 150,
        "account_industry": "Tech",
        "account_country": "US",
        "account_referral_source": "organic",
        "account_initial_seats": 20,
        "account_is_trial": 0,
        "sub_active_count": 2,
        "sub_current_mrr": 6500.0,
        "sub_current_arr": 78000.0,
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


def _make_auth_header(
    user_id: str = "test-user-uuid",
    secret: str = TEST_SECRET,
    expires_in: int = 3600,
    role: str = "authenticated",
) -> dict[str, str]:
    """Helper to create Authorization header with signed test JWT."""
    payload = {
        "sub": user_id,
        "email": f"{user_id}@example.com",
        "role": role,
        "exp": int(time.time()) + expires_in,
    }
    token = create_jwt_token(payload=payload, secret=secret)
    return {"Authorization": f"Bearer {token}"}


class TestAuthentication(unittest.TestCase):
    """Test suite for Supabase JWT verification and fail-closed security."""

    def setUp(self) -> None:
        self.settings = get_settings()
        self.original_secret = self.settings.SUPABASE_JWT_SECRET
        self.settings.SUPABASE_JWT_SECRET = TEST_SECRET
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.settings.SUPABASE_JWT_SECRET = self.original_secret

    def test_missing_auth_header_rejected(self) -> None:
        """Requests without Authorization header must return HTTP 401."""
        response = self.client.post(
            "/api/v1/predictions/score",
            json={"features": _get_valid_feature_payload()},
        )
        self.assertEqual(response.status_code, 401)
        self.assertIn("Bearer token is missing", response.json()["detail"])

    def test_expired_token_rejected(self) -> None:
        """Requests with an expired JWT token must return HTTP 401."""
        headers = _make_auth_header(expires_in=-300)  # Expired 5 minutes ago
        response = self.client.post(
            "/api/v1/predictions/score",
            headers=headers,
            json={"features": _get_valid_feature_payload()},
        )
        self.assertEqual(response.status_code, 401)
        self.assertIn("expired", response.json()["detail"].lower())

    def test_invalid_signature_rejected(self) -> None:
        """Requests with an invalid cryptographic signature must return HTTP 401."""
        headers = _make_auth_header(secret="wrong-secret-key-that-does-not-match!")
        response = self.client.post(
            "/api/v1/predictions/score",
            headers=headers,
            json={"features": _get_valid_feature_payload()},
        )
        self.assertEqual(response.status_code, 401)
        self.assertIn("signature", response.json()["detail"].lower())

    def test_malformed_token_rejected(self) -> None:
        """Malformed JWT strings must return HTTP 401."""
        headers = {"Authorization": "Bearer not-a-valid-jwt-token"}
        response = self.client.post(
            "/api/v1/predictions/score",
            headers=headers,
            json={"features": _get_valid_feature_payload()},
        )
        self.assertEqual(response.status_code, 401)

    def test_unconfigured_secret_fails_closed(self) -> None:
        """Missing SUPABASE_JWT_SECRET must fail closed with HTTP 500."""
        self.settings.SUPABASE_JWT_SECRET = None
        headers = _make_auth_header()
        response = self.client.post(
            "/api/v1/predictions/score",
            headers=headers,
            json={"features": _get_valid_feature_payload()},
        )
        self.assertEqual(response.status_code, 500)
        self.assertIn("SUPABASE_JWT_SECRET is required", response.json()["detail"])


class TestPredictionEndpoints(unittest.TestCase):
    """Test suite for /api/v1/predictions/score endpoint."""

    def setUp(self) -> None:
        self.settings = get_settings()
        self.original_secret = self.settings.SUPABASE_JWT_SECRET
        self.settings.SUPABASE_JWT_SECRET = TEST_SECRET
        self.client = TestClient(app)
        self.headers = _make_auth_header()

    def tearDown(self) -> None:
        self.settings.SUPABASE_JWT_SECRET = self.original_secret

    def test_score_endpoint_success(self) -> None:
        """Valid feature vector produces expected score and risk tier."""
        payload = {
            "account_id": "A-TEST-100",
            "snapshot_date": "2024-11-01",
            "features": _get_valid_feature_payload(),
        }
        response = self.client.post(
            "/api/v1/predictions/score",
            headers=self.headers,
            json=payload,
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["account_id"], "A-TEST-100")
        self.assertEqual(data["snapshot_date"], "2024-11-01")
        self.assertIn(data["risk_tier"], ["High Risk", "Moderate Risk", "Low Risk"])
        self.assertIn(data["predicted_risk_label"], [0, 1])
        self.assertGreaterEqual(data["model_score"], 0.0)
        self.assertLessEqual(data["model_score"], 1.0)
        self.assertEqual(data["model_name"], "flurit_churn_predictor")
        self.assertEqual(data["model_version"], "1.0.0")
        self.assertIn("uncalibrated risk rankings", data["score_interpretation"])

    def test_score_endpoint_missing_feature_rejected(self) -> None:
        """Missing canonical feature must be rejected with HTTP 422."""
        bad_features = _get_valid_feature_payload()
        del bad_features["sub_current_mrr"]
        response = self.client.post(
            "/api/v1/predictions/score",
            headers=self.headers,
            json={"features": bad_features},
        )
        self.assertEqual(response.status_code, 422)

    def test_score_endpoint_extra_field_rejected(self) -> None:
        """Extra/unexpected field must be rejected to prevent silent leakage."""
        bad_features = _get_valid_feature_payload()
        bad_features["target_churn_60d"] = 1  # Target leakage attempt
        response = self.client.post(
            "/api/v1/predictions/score",
            headers=self.headers,
            json={"features": bad_features},
        )
        self.assertEqual(response.status_code, 422)

    def test_score_endpoint_invalid_values_rejected(self) -> None:
        """Negative counts or out-of-range rates must fail validation."""
        bad_features = _get_valid_feature_payload()
        bad_features["sub_active_count"] = -5  # Invalid negative count
        response = self.client.post(
            "/api/v1/predictions/score",
            headers=self.headers,
            json={"features": bad_features},
        )
        self.assertEqual(response.status_code, 422)


class TestCustomerRiskEndpoint(unittest.TestCase):
    """Test suite for /api/v1/customers/{account_id}/risk endpoint."""

    def setUp(self) -> None:
        self.settings = get_settings()
        self.original_secret = self.settings.SUPABASE_JWT_SECRET
        self.settings.SUPABASE_JWT_SECRET = TEST_SECRET
        self.client = TestClient(app)
        self.headers = _make_auth_header()

    def tearDown(self) -> None:
        self.settings.SUPABASE_JWT_SECRET = self.original_secret

    def test_customer_risk_success(self) -> None:
        """Comprehensive risk assessment includes score, SHAP factors, and segment."""
        payload = {
            "snapshot_date": "2024-11-01",
            "features": _get_valid_feature_payload(),
        }
        response = self.client.post(
            "/api/v1/customers/A-00cac8/risk",
            headers=self.headers,
            json=payload,
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["account_id"], "A-00cac8")
        self.assertEqual(data["snapshot_date"], "2024-11-01")
        self.assertIn(data["risk_tier"], ["High Risk", "Moderate Risk", "Low Risk"])
        self.assertIn(data["segment_id"], [0, 1, 2])
        self.assertTrue(len(data["segment_name"]) > 0)
        self.assertTrue(len(data["segment_summary"]) > 0)

        # Factor breakdowns
        self.assertGreater(len(data["top_risk_increasing_factors"]), 0)
        self.assertGreater(len(data["top_risk_decreasing_factors"]), 0)
        for factor in data["top_risk_increasing_factors"]:
            self.assertEqual(factor["direction"], "increases_risk")
            self.assertGreater(factor["attribution"], 0.0)

        for factor in data["top_risk_decreasing_factors"]:
            self.assertEqual(factor["direction"], "decreases_risk")
            self.assertLess(factor["attribution"], 0.0)

        # Statistical association disclaimer
        self.assertIn("Statistical Association Note", data["association_disclaimer"])
        self.assertIn("NOT imply a direct causal relationship", data["association_disclaimer"])
        self.assertIn("Estimated model churn score is", data["plain_language_summary"])

    def test_customer_risk_whitespace_account_id_rejected(self) -> None:
        """Empty or whitespace-only account_id must be rejected."""
        payload = {
            "snapshot_date": "2024-11-01",
            "features": _get_valid_feature_payload(),
        }
        response = self.client.post(
            "/api/v1/customers/%20/risk",
            headers=self.headers,
            json=payload,
        )
        self.assertEqual(response.status_code, 400)


class TestPrioritizationEndpoint(unittest.TestCase):
    """Test suite for /api/v1/prioritization/quadrants endpoint."""

    def setUp(self) -> None:
        self.settings = get_settings()
        self.original_secret = self.settings.SUPABASE_JWT_SECRET
        self.settings.SUPABASE_JWT_SECRET = TEST_SECRET
        self.client = TestClient(app)
        self.headers = _make_auth_header()

    def tearDown(self) -> None:
        self.settings.SUPABASE_JWT_SECRET = self.original_secret

    def test_prioritization_quadrants_success(self) -> None:
        """Portfolio prioritization aggregates correctly into 5 tiers."""
        records = [
            {"account_id": "A1", "sub_current_mrr": 9500.0, "model_score": 0.65},
            {"account_id": "A2", "sub_current_mrr": 5500.0, "model_score": 0.70},
            {"account_id": "A3", "sub_current_mrr": 9200.0, "model_score": 0.40},
            {"account_id": "A4", "sub_current_mrr": 2000.0, "model_score": 0.55},
            {"account_id": "A5", "sub_current_mrr": 1500.0, "model_score": 0.20},
        ]
        response = self.client.post(
            "/api/v1/prioritization/quadrants",
            headers=self.headers,
            json={"records": records},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["total_accounts"], 5)
        self.assertEqual(data["total_mrr"], 27700.0)
        self.assertAlmostEqual(data["total_estimated_revenue_at_risk_mrr"], 15105.0, places=1)
        self.assertEqual(len(data["quadrants"]), 5)

        priorities = [q["priority_level"] for q in data["quadrants"]]
        self.assertEqual(priorities, [1, 2, 3, 4, 5])
        self.assertIn("operational prioritization score", data["risk_metric_interpretation"])

    def test_prioritization_quadrants_empty_records_rejected(self) -> None:
        """Empty records list must be rejected with HTTP 422."""
        response = self.client.post(
            "/api/v1/prioritization/quadrants",
            headers=self.headers,
            json={"records": []},
        )
        self.assertEqual(response.status_code, 422)


class TestInferenceServiceArtifactFailures(unittest.TestCase):
    """Test suite verifying safe artifact handling and failure modes."""

    def test_missing_model_artifact_raises(self) -> None:
        """Missing model directory must raise FileNotFoundError."""
        service = ModelInferenceService()
        with self.assertRaises(FileNotFoundError):
            service.load_artifacts(
                models_dir=Path("non_existent_models_dir"),
                explainability_dir=Path("artifacts/explainability"),
                segmentation_dir=Path("artifacts/segmentation"),
            )

    def test_missing_explainer_artifact_raises(self) -> None:
        """Missing explainer directory must raise FileNotFoundError."""
        service = ModelInferenceService()
        with self.assertRaises(FileNotFoundError):
            service.load_artifacts(
                models_dir=Path("artifacts/models/churn_prediction_v1"),
                explainability_dir=Path("non_existent_explainer_dir"),
                segmentation_dir=Path("artifacts/segmentation"),
            )

    def test_uninitialized_service_raises_runtime_error(self) -> None:
        """Calling score before load_artifacts must raise RuntimeError."""
        service = ModelInferenceService()
        req = ScorePredictionRequest(
            features=CanonicalFeaturesPayload(**_get_valid_feature_payload())
        )
        with self.assertRaises(RuntimeError):
            service.score(req)


if __name__ == "__main__":
    unittest.main()
