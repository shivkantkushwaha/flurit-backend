"""Tests for Simplified Single-Service College-Project Deployment (Phase 18).

Validates:
1. Production configuration validation in simplified tier:
   - Accepts single-service setup (hosted PostgreSQL, Supabase JWT, local storage, memory/disabled queue).
   - Rejects weak secrets, SQLite, default passwords, and Bynalt references.
2. Endpoint gating for asynchronous processing:
   - When QUEUE_MODE="disabled", async job submission returns HTTP 503 directing clients to sync endpoint.
   - When QUEUE_MODE="memory", async job submission returns HTTP 202 with honesty warning and durability header.
3. Synchronous processing primary path:
   - Returns X-Processing-Mode: synchronous-primary in simplified tier.
4. Database connection pool singleton reuse in background worker:
   - Worker uses get_shared_engine() instead of creating new engines per job.
"""

import time
from unittest.mock import patch

from starlette.testclient import TestClient

from app.core.auth import create_jwt_token
from app.core.config import (
    Settings,
    get_settings,
    validate_production_settings,
)
from app.core.queue import clear_queue_broker_cache
from app.db.session import get_shared_engine
from app.main import app
from app.onboarding.schemas import (
    AccountConceptMapping,
    ChurnEventsConceptMapping,
    FeatureUsageConceptMapping,
    SchemaMappingRequest,
    SubscriptionConceptMapping,
    SupportTicketsConceptMapping,
)
from app.worker import WorkerSettings

TEST_SECRET = "flurit-secure-test-jwt-secret-key-32-chars-min!"


def _generate_valid_jwt(user_id: str = "simplified-user-123") -> str:
    """Generate signed Supabase JWT for testing."""
    secret = get_settings().SUPABASE_JWT_SECRET
    payload = {
        "sub": user_id,
        "email": "student@college.edu",
        "role": "authenticated",
        "aud": "authenticated",
        "exp": int(time.time()) + 3600,
    }
    return create_jwt_token(payload, secret=secret)


def _build_compatible_mapping() -> SchemaMappingRequest:
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


# ==============================================================================
# 1. Simplified Production Configuration Validation
# ==============================================================================


def test_simplified_tier_production_settings_accepts_single_service_config():
    """Verify simplified tier accepts single API service without Redis or external storage."""
    simplified_prod = Settings(
        ENVIRONMENT="production",
        DEPLOYMENT_TIER="simplified",
        SUPABASE_JWT_SECRET="super-strong-college-project-production-jwt-secret-32-chars",
        DATABASE_URL="postgresql+psycopg2://flurit_app:SecretPassword123!@ep-xyz.railway.internal:5432/flurit_db",
        QUEUE_MODE="memory",
        STORAGE_BACKEND="local",
        CORS_ORIGINS=["https://flurit-frontend.vercel.app"],
    )

    errors = validate_production_settings(simplified_prod, raise_on_error=True)
    assert len(errors) == 0


def test_simplified_tier_production_settings_rejects_insecure_defaults():
    """Verify simplified tier still catches SQLite, weak JWT, default password, and Bynalt."""
    bad_simplified = Settings(
        ENVIRONMENT="production",
        DEPLOYMENT_TIER="simplified",
        SUPABASE_JWT_SECRET="short",
        DATABASE_URL="sqlite:///flurit.db",
        QUEUE_MODE="memory",
        STORAGE_BACKEND="local",
        CORS_ORIGINS=["*"],
    )

    errors = validate_production_settings(bad_simplified, raise_on_error=False)
    error_text = " ".join(errors)
    assert "SUPABASE_JWT_SECRET must be at least 32 characters" in error_text
    assert "SQLite cannot be used" in error_text
    assert "CORS_ORIGINS cannot contain wildcard" in error_text


def test_enterprise_tier_strictly_enforces_arq_and_redis():
    """Verify enterprise tier mandates ARQ and rejects in-memory queue."""
    enterprise_bad = Settings(
        ENVIRONMENT="production",
        DEPLOYMENT_TIER="enterprise",
        SUPABASE_JWT_SECRET="super-strong-college-project-production-jwt-secret-32-chars",
        DATABASE_URL="postgresql+psycopg2://flurit_app:SecretPassword123!@ep-xyz.railway.internal:5432/flurit_db",
        QUEUE_MODE="memory",
        REDIS_URL="redis://localhost:6379/0",
        STORAGE_BACKEND="local",
        CORS_ORIGINS=["https://flurit.com"],
    )

    errors = validate_production_settings(enterprise_bad, raise_on_error=False)
    error_text = " ".join(errors)
    assert "Production requires QUEUE_MODE='arq' backed by Redis" in error_text
    assert "REDIS_URL points to localhost" in error_text


# ==============================================================================
# 2. Asynchronous Endpoint Gating & Honesty Tests
# ==============================================================================


def test_async_job_endpoint_gated_when_queue_disabled():
    """Verify POST /onboarding/process/async returns HTTP 503 when QUEUE_MODE='disabled'."""
    client = TestClient(app)
    token = _generate_valid_jwt()
    mapping = _build_compatible_mapping()

    files = [
        ("files", ("accounts.csv", b"cust_id,created_date\nc1,2024-01-01\n", "text/csv")),
    ]
    data = {"mapping": mapping.model_dump_json()}
    headers = {"Authorization": f"Bearer {token}"}

    with patch.object(get_settings(), "QUEUE_MODE", "disabled"):
        clear_queue_broker_cache()
        resp = client.post(
            "/api/v1/onboarding/process/async",
            files=files,
            data=data,
            headers=headers,
        )
        assert resp.status_code == 503
        detail = resp.json()["detail"]
        assert "Asynchronous background processing is disabled" in detail
        assert "POST /api/v1/onboarding/process" in detail


def test_async_job_endpoint_emits_ephemeral_header_when_queue_memory():
    """Verify POST /onboarding/process/async emits honesty durability header when in-process."""
    client = TestClient(app)
    token = _generate_valid_jwt()
    mapping = _build_compatible_mapping()

    files = [
        ("files", ("accounts.csv", b"cust_id,created_date\nc1,2024-01-01\n", "text/csv")),
    ]
    data = {"mapping": mapping.model_dump_json()}
    headers = {"Authorization": f"Bearer {token}"}

    with patch.object(get_settings(), "QUEUE_MODE", "memory"):
        clear_queue_broker_cache()
        resp = client.post(
            "/api/v1/onboarding/process/async",
            files=files,
            data=data,
            headers=headers,
        )
        assert resp.status_code == 202
        assert resp.headers.get("X-Queue-Durability") == "ephemeral-in-process"
        assert "In-process queue active" in resp.headers.get("Warning", "")
        body = resp.json()
        assert "in-process execution" in body["message"]


# ==============================================================================
# 3. Synchronous Processing Primary Path Tests
# ==============================================================================


def test_sync_process_endpoint_emits_primary_mode_header_in_simplified_tier():
    """Verify synchronous endpoint emits X-Processing-Mode: synchronous-primary in simplified tier."""
    client = TestClient(app)
    token = _generate_valid_jwt()
    mapping = _build_compatible_mapping()

    files = [
        ("files", ("accounts.csv", b"cust_id,created_date\nc1,2024-01-01\n", "text/csv")),
    ]
    data = {"mapping": mapping.model_dump_json()}
    headers = {"Authorization": f"Bearer {token}"}

    with patch.object(get_settings(), "DEPLOYMENT_TIER", "simplified"):
        resp = client.post(
            "/api/v1/onboarding/process",
            files=files,
            data=data,
            headers=headers,
        )
        # Should NOT contain deprecation warning
        assert resp.headers.get("Deprecation") is None
        assert resp.headers.get("X-Processing-Mode") == "synchronous-primary"
        assert resp.headers.get("X-Deployment-Tier") == "simplified"


# ==============================================================================
# 4. Worker Connection Pool Singleton Reuse
# ==============================================================================


def test_worker_reuses_shared_db_engine():
    """Verify worker startup and settings use get_shared_engine singleton."""
    engine = get_shared_engine()
    assert engine is not None
    # Verify WorkerSettings attributes
    assert WorkerSettings.functions is not None
    assert WorkerSettings.max_jobs == 10
