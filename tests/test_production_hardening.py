"""Production Configuration, Database Reliability, and Operational Hardening Tests for Flurit.

Validates Phase 16 requirements:
1. Independent liveness probes (/health, /healthz, /api/v1/health).
2. Database readiness probes (/ready, /readyz, /api/v1/ready) with SELECT 1 and credential redaction.
3. Production configuration validation: strict rejection of missing secrets, insecure passwords,
   SQLite, wildcard CORS, and in-memory queue/storage in production.
4. Queue broker production enforcement: no silent fallback to MemoryQueueBroker.
5. Worker cancellation race condition guard: cancelled jobs are never overwritten to succeeded.
6. Stale job recovery for interrupted or crashed worker processes.
7. Request correlation ID middleware (X-Request-ID propagation).
8. Structured logging with sensitive data redaction (passwords, tokens, secrets).
9. Bounded SQLAlchemy connection pool configuration.
"""

import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker
from starlette.testclient import TestClient

from app.core.config import Settings, validate_production_settings
from app.core.logging import (
    JSONFormatter,
    _redact_sensitive_data,
    set_correlation_id,
)
from app.core.queue import get_queue_broker
from app.db.session import (
    check_db_readiness,
    create_db_engine,
    get_db,
    validate_database_url_isolation,
)
from app.main import app
from app.models.job import ProcessingJob
from app.models.workspace import Workspace, WorkspaceMember
from app.worker import execute_processing_job, recover_stale_jobs


@pytest.fixture(scope="module")
def local_engine():
    """Create test engine targeting local flurit_db with isolation validation."""
    settings = Settings()
    url = settings.DATABASE_URL
    validate_database_url_isolation(url)

    engine = sa.create_engine(url)
    try:
        with engine.connect() as conn:
            db_name = conn.execute(sa.text("SELECT current_database();")).scalar()
            assert db_name == "flurit_db"
    except Exception as exc:
        pytest.skip(f"Could not connect to PostgreSQL flurit_db: {exc}")

    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def local_session(local_engine):
    """Transactional session with teardown of hardening test fixtures."""
    SessionLocal = sessionmaker(bind=local_engine, autocommit=False, autoflush=False)
    session: Session = SessionLocal()

    prefix = "HARDEN_TEST_"
    try:
        yield session
    finally:
        session.rollback()
        try:
            with local_engine.begin() as cleanup_conn:
                cleanup_conn.execute(
                    sa.text(f"DELETE FROM processing_jobs WHERE workspace_id LIKE '{prefix}%';")
                )
                cleanup_conn.execute(
                    sa.text(f"DELETE FROM workspaces WHERE workspace_id LIKE '{prefix}%';")
                )
        except Exception:
            pass
        session.close()


@pytest.fixture
def client(local_session):
    """FastAPI TestClient with overridden get_db sharing local test session."""

    def _override_get_db():
        yield local_session

    app.dependency_overrides[get_db] = _override_get_db
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.pop(get_db, None)


# ==============================================================================
# 1. Liveness and Readiness Probes
# ==============================================================================


def test_liveness_probes_independent_of_database(client):
    """Verify /health and /healthz return 200 OK without requiring database connectivity."""
    # Even if check_db_readiness fails or raises, liveness must return 200 OK
    with patch("app.main.check_db_readiness", side_effect=Exception("Database down")):
        for route in ("/health", "/healthz", "/api/v1/health"):
            resp = client.get(route)
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert "app" in data
            assert "version" in data or "api_version" in data


def test_readiness_probe_success_against_local_postgres(client):
    """Verify /ready and /readyz return 200 OK with connected status against local flurit_db."""
    for route in ("/ready", "/readyz", "/api/v1/ready"):
        resp = client.get(route)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["database"] == "connected"
        assert "password" not in json.dumps(data).lower()
        assert "postgresql://" not in json.dumps(data).lower()


def test_readiness_probe_failure_handling_and_redaction(client):
    """Verify readiness probes return HTTP 503 when DB fails, without exposing credentials or hosts."""
    with patch("app.main.check_db_readiness", return_value=False):
        for route in ("/ready", "/readyz", "/api/v1/ready"):
            resp = client.get(route)
            assert resp.status_code == 503
            data = resp.json()
            assert data["status"] == "not_ready"
            assert data["database"] == "unavailable"
            assert data["detail"] == "Database connection probe failed."

            # Verify no secrets or connection strings leaked
            body_text = json.dumps(data).lower()
            assert "password" not in body_text
            assert "flurit_password" not in body_text
            assert "postgresql" not in body_text
            assert "localhost" not in body_text
            assert "5432" not in body_text


# ==============================================================================
# 2. Database Connection Pooling Configuration
# ==============================================================================


def test_database_pooling_settings_applied():
    """Verify SQLAlchemy engine creation configures bounded pooling and recycle settings."""
    settings = Settings(
        DATABASE_URL="postgresql+psycopg2://test_user:test_pass@localhost:5432/flurit_db",
        DB_POOL_SIZE=15,
        DB_MAX_OVERFLOW=7,
        DB_POOL_TIMEOUT_SECONDS=45,
        DB_POOL_RECYCLE_SECONDS=1200,
    )
    with patch("app.db.session.get_settings", return_value=settings):
        engine = create_db_engine(settings.DATABASE_URL)
        assert engine.pool.size() == 15
        assert engine.pool._max_overflow == 7
        assert engine.pool._timeout == 45
        assert engine.pool._recycle == 1200
        engine.dispose()


def test_check_db_readiness_helper():
    """Verify check_db_readiness performs lightweight SELECT 1 check."""
    # Real connection test
    is_ready = check_db_readiness()
    assert is_ready is True

    # Simulated connection error
    mock_engine = sa.create_engine("sqlite:///:memory:")
    with patch.object(
        mock_engine, "connect", side_effect=sa.exc.OperationalError("Mocked error", {}, None)
    ):
        failed_ready = check_db_readiness(mock_engine)
        assert failed_ready is False


# ==============================================================================
# 3. Production Configuration Validation
# ==============================================================================


def test_validate_production_settings_catches_all_violations():
    """Verify validate_production_settings catches missing secrets, SQLite, and insecure defaults."""
    # Bad production configuration
    bad_settings = Settings(
        ENVIRONMENT="production",
        SUPABASE_JWT_SECRET="",  # Missing secret
        DATABASE_URL="sqlite:///flurit.db",  # SQLite in production
        QUEUE_MODE="memory",  # In-memory queue in production
        REDIS_URL="redis://localhost:6379/0",  # Localhost redis in prod
        STORAGE_BACKEND="memory",  # In-memory storage in prod
        CORS_ORIGINS=["*"],  # Wildcard CORS
    )

    errors = validate_production_settings(bad_settings, raise_on_error=False)
    assert len(errors) >= 6

    error_text = " ".join(errors)
    assert "SUPABASE_JWT_SECRET is mandatory" in error_text
    assert "SQLite cannot be used" in error_text
    assert "Production requires QUEUE_MODE='arq'" in error_text
    assert "REDIS_URL points to localhost" in error_text
    assert "STORAGE_BACKEND cannot be 'memory'" in error_text
    assert "CORS_ORIGINS cannot contain wildcard" in error_text

    # Verify raise_on_error=True raises ValueError
    with pytest.raises(ValueError, match="CRITICAL PRODUCTION CONFIGURATION ERRORS DETECTED"):
        validate_production_settings(bad_settings, raise_on_error=True)


def test_validate_production_settings_accepts_valid_config():
    """Verify validate_production_settings passes cleanly with fully compliant production config."""
    valid_settings = Settings(
        ENVIRONMENT="production",
        SUPABASE_JWT_SECRET="super-secure-production-jwt-secret-with-sufficient-entropy-12345",
        DATABASE_URL="postgresql+psycopg2://flurit_prod_user:StrongSecretPass99!@db.render.internal:5432/flurit_db",
        QUEUE_MODE="arq",
        REDIS_URL="rediss://:prod_redis_pass@redis.render.internal:6379/0",
        STORAGE_BACKEND="local",
        CORS_ORIGINS=["https://app.flurit.io", "https://dashboard.flurit.io"],
    )

    errors = validate_production_settings(valid_settings, raise_on_error=True)
    assert len(errors) == 0


def test_validate_production_settings_es256_without_secret():
    """Verify ES256 production config does not require SUPABASE_JWT_SECRET when HS256 is disabled."""
    valid_es256 = Settings(
        ENVIRONMENT="production",
        SUPABASE_URL="https://jhankemazkagrybaoitu.supabase.co",
        SUPABASE_JWT_ISSUER="https://jhankemazkagrybaoitu.supabase.co/auth/v1",
        SUPABASE_JWT_ALGORITHM="ES256",
        SUPABASE_ALLOW_HS256=False,
        SUPABASE_JWT_SECRET=None,
        DATABASE_URL="postgresql+psycopg2://flurit_prod_user:StrongSecretPass99!@db.render.internal:5432/flurit_db",
        QUEUE_MODE="arq",
        REDIS_URL="rediss://:prod_redis_pass@redis.render.internal:6379/0",
        STORAGE_BACKEND="local",
        CORS_ORIGINS=["https://app.flurit.io"],
    )
    errors = validate_production_settings(valid_es256, raise_on_error=False)
    assert errors == []

    # But missing SUPABASE_URL or SUPABASE_JWT_ISSUER must be caught
    bad_es256 = Settings(
        ENVIRONMENT="production",
        SUPABASE_URL="",
        SUPABASE_JWT_ISSUER="",
        SUPABASE_JWT_ALGORITHM="ES256",
        SUPABASE_ALLOW_HS256=False,
        DATABASE_URL="postgresql+psycopg2://flurit_prod_user:StrongSecretPass99!@db.render.internal:5432/flurit_db",
        QUEUE_MODE="arq",
        REDIS_URL="rediss://:prod_redis_pass@redis.render.internal:6379/0",
        STORAGE_BACKEND="local",
        CORS_ORIGINS=["https://app.flurit.io"],
    )
    errs = validate_production_settings(bad_es256, raise_on_error=False)
    err_text = " ".join(errs)
    assert "SUPABASE_URL is mandatory" in err_text
    assert "SUPABASE_JWT_ISSUER is mandatory" in err_text


# ==============================================================================
# 4. Queue Broker Production Isolation
# ==============================================================================


def test_get_queue_broker_rejects_non_arq_in_production():
    """get_queue_broker must never silently fall back to MemoryQueueBroker in production."""
    prod_memory_settings = Settings(
        ENVIRONMENT="production",
        QUEUE_MODE="memory",
    )
    with pytest.raises(
        ValueError,
        match="In-memory or synchronous queue fallback is strictly prohibited in production",
    ):
        get_queue_broker(prod_memory_settings)

    prod_sync_settings = Settings(
        ENVIRONMENT="production",
        QUEUE_MODE="sync",
    )
    with pytest.raises(
        ValueError,
        match="In-memory or synchronous queue fallback is strictly prohibited in production",
    ):
        get_queue_broker(prod_sync_settings)


# ==============================================================================
# 5. Worker Cancellation Race Condition Guard
# ==============================================================================


@pytest.mark.anyio
async def test_worker_cancelled_job_cannot_become_succeeded(local_session):
    """Confirm that a job marked cancelled concurrently cannot later be reported as succeeded."""
    user_id = "user_cancel_race_1"
    ws_id = f"HARDEN_TEST_{uuid.uuid4().hex[:12]}"
    ws = Workspace(workspace_id=ws_id, workspace_name="Race Test WS", owner_user_id=user_id)
    mem = WorkspaceMember(workspace_id=ws_id, user_id=user_id, role="owner")
    local_session.add_all([ws, mem])

    job_id = f"job_race_{uuid.uuid4().hex[:12]}"
    job = ProcessingJob(
        job_id=job_id,
        workspace_id=ws_id,
        user_id=user_id,
        status="queued",
        mapping_payload={
            "accounts": {"account_id": "cust_id", "signup_date": "created_date"},
            "subscriptions": {
                "subscription_id": "sub_id",
                "account_id": "cust_id",
                "start_date": "start_d",
            },
        },
        file_keys=[],
        created_at=datetime.now(UTC),
    )
    local_session.add(job)
    local_session.commit()

    # Pre-cancel the job in the database
    job.status = "cancelled"
    job.stage_message = "Cancelled by user"
    job.completed_at = datetime.now(UTC)
    local_session.commit()

    # Worker executes on this job
    await execute_processing_job(job_id=job_id, workspace_id=ws_id)

    # Verify that worker did NOT overwrite status to failed or succeeded
    local_session.refresh(job)
    assert job.status == "cancelled"
    assert job.stage_message == "Cancelled by user"


# ==============================================================================
# 6. Stale Job Recovery
# ==============================================================================


def test_recover_stale_jobs_marks_expired_running_jobs_failed(local_session):
    """recover_stale_jobs must transition running jobs past timeout to failed."""
    user_id = "user_stale_test"
    ws_id = f"HARDEN_TEST_{uuid.uuid4().hex[:12]}"
    ws = Workspace(workspace_id=ws_id, workspace_name="Stale Test WS", owner_user_id=user_id)
    mem = WorkspaceMember(workspace_id=ws_id, user_id=user_id, role="owner")
    local_session.add_all([ws, mem])

    now_dt = datetime.now(UTC)
    stale_time = now_dt - timedelta(seconds=1200)  # 20 minutes ago
    fresh_time = now_dt - timedelta(seconds=60)  # 1 minute ago

    # Stale running job
    stale_job = ProcessingJob(
        job_id=f"job_stale_{uuid.uuid4().hex[:10]}",
        workspace_id=ws_id,
        user_id=user_id,
        status="running",
        started_at=stale_time,
        mapping_payload={"accounts": {"account_id": "c"}},
        created_at=stale_time,
    )
    # Fresh running job (should not be touched)
    fresh_job = ProcessingJob(
        job_id=f"job_fresh_{uuid.uuid4().hex[:10]}",
        workspace_id=ws_id,
        user_id=user_id,
        status="running",
        started_at=fresh_time,
        mapping_payload={"accounts": {"account_id": "c"}},
        created_at=fresh_time,
    )
    # Completed job (should not be touched)
    succeeded_job = ProcessingJob(
        job_id=f"job_succ_{uuid.uuid4().hex[:10]}",
        workspace_id=ws_id,
        user_id=user_id,
        status="succeeded",
        started_at=stale_time,
        mapping_payload={"accounts": {"account_id": "c"}},
        created_at=stale_time,
    )

    local_session.add_all([stale_job, fresh_job, succeeded_job])
    local_session.commit()

    # Recover stale jobs older than 600s (10 minutes)
    recovered = recover_stale_jobs(local_session, max_age_seconds=600)
    assert recovered >= 1

    local_session.refresh(stale_job)
    local_session.refresh(fresh_job)
    local_session.refresh(succeeded_job)

    assert stale_job.status == "failed"
    assert "interrupted" in stale_job.stage_message.lower()
    assert fresh_job.status == "running"
    assert succeeded_job.status == "succeeded"


# ==============================================================================
# 7. Correlation ID Middleware and Tracing
# ==============================================================================


def test_correlation_id_generated_and_propagated(client):
    """Verify correlation ID middleware generates X-Request-ID if absent and echoes back."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "X-Request-ID" in resp.headers
    assert resp.headers["X-Request-ID"].startswith("req_")


def test_correlation_id_preserved_when_supplied_by_client(client):
    """Verify incoming client X-Request-ID or X-Correlation-ID is preserved in response."""
    custom_id = "client-trace-id-abc-1234"
    resp = client.get("/health", headers={"X-Request-ID": custom_id})
    assert resp.status_code == 200
    assert resp.headers.get("X-Request-ID") == custom_id


# ==============================================================================
# 8. Structured Logging & Sensitive Data Redaction
# ==============================================================================


def test_sensitive_data_redaction_in_logging():
    """Verify _redact_sensitive_data scrubs passwords, tokens, and secrets."""
    payload = {
        "user_id": "user_123",
        "password": "super_secret_password",
        "jwt_token": "eyJhbGciOi...",
        "api_key": "flurit_sk_12345",
        "metadata": {
            "secret_value": "confidential",
            "safe_metric": 42.5,
        },
    }
    redacted = _redact_sensitive_data(payload)

    assert redacted["user_id"] == "user_123"
    assert redacted["password"] == "[REDACTED]"
    assert redacted["jwt_token"] == "[REDACTED]"
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["metadata"]["secret_value"] == "[REDACTED]"
    assert redacted["metadata"]["safe_metric"] == 42.5


def test_json_formatter_includes_correlation_id_and_redacts():
    """Verify JSONFormatter includes active request_id and redacts sensitive extra keys."""
    set_correlation_id("trace_req_998877")
    try:
        formatter = JSONFormatter(environment="production")
        record = logging.LogRecord(
            name="flurit.security",
            level=logging.INFO,
            pathname=__file__,
            lineno=100,
            msg="User login processed",
            args=(),
            exc_info=None,
        )
        record.__dict__["token"] = "sensitive_bearer_token"
        record.__dict__["account_id"] = "acc_001"

        formatted = formatter.format(record)
        log_json = json.loads(formatted)

        assert log_json["request_id"] == "trace_req_998877"
        assert log_json["extra"]["account_id"] == "acc_001"
        assert log_json["extra"]["token"] == "[REDACTED]"
    finally:
        set_correlation_id(None)
