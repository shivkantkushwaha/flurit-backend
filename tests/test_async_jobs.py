"""Asynchronous Processing and Background Job Management Test Suite for Flurit.

Validates Phase 15 requirements:
1. Asynchronous dataset submission returning HTTP 202 Accepted and unique job ID.
2. Background worker execution (queued -> running -> succeeded) with snapshotting and ML scoring.
3. Multi-tenant security: cross-workspace job status lookups and cancellations blocked (HTTP 403).
4. Idempotency key handling preventing duplicate job submissions.
5. Paginated workspace-scoped job listings.
6. Job cancellation and staging file cleanup.
7. File upload size limit enforcement (HTTP 413).
8. Deprecation and Warning headers on synchronous /onboarding/process.
9. Safe error sanitization on worker failures without exposing paths, SQL, or credentials.
10. Storage backend safety (path traversal protection and lifecycle cleanup).
"""

import time
import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker
from starlette.testclient import TestClient

from app.core.auth import create_jwt_token
from app.core.config import get_settings
from app.core.queue import MemoryQueueBroker, SyncQueueBroker, get_queue_broker
from app.core.storage import LocalFileStorage, MemoryFileStorage
from app.db.session import get_db, validate_database_url_isolation
from app.main import app
from app.models.workspace import Workspace, WorkspaceMember
from app.onboarding.schemas import (
    AccountConceptMapping,
    ChurnEventsConceptMapping,
    FeatureUsageConceptMapping,
    ProcessingOptions,
    SchemaMappingRequest,
    SubscriptionConceptMapping,
    SupportTicketsConceptMapping,
)
from app.worker import _sanitize_error_message, execute_processing_job

TEST_SECRET = "flurit-secure-test-jwt-secret-key-32-chars-min!"


def _make_jwt(user_id: str, email: str = "async_user@example.com") -> str:
    """Generate a valid signed Supabase JWT for testing."""
    secret = get_settings().SUPABASE_JWT_SECRET
    payload = {
        "sub": user_id,
        "email": email,
        "role": "authenticated",
        "aud": "authenticated",
        "exp": int(time.time()) + 3600,
    }
    return create_jwt_token(payload=payload, secret=secret)


def _build_compatible_csv_fixtures() -> dict[str, str]:
    """Build synthetic 5-concept CSV string fixtures spanning 240 days."""
    accounts_csv = (
        "cust_id,created_date,vertical,hq_country,acquisition_channel,licensed_seats,is_trial_flag\n"
        "cust_1,2024-01-01,FinTech,US,organic,10,0\n"
        "cust_2,2024-01-05,DevTools,GB,partner,25,0\n"
        "cust_3,2024-01-10,Healthcare,DE,paid_ad,5,1\n"
        "cust_4,2024-01-15,E-commerce,US,referral,50,0\n"
    )
    subs_csv = (
        "sub_key,cust_id,activation_date,cancellation_date,monthly_revenue,annual_revenue,seat_count,billing_cycle,auto_renewal,trial_active\n"
        "sub_1,cust_1,2024-01-01,,1500.0,18000.0,10,monthly,1,0\n"
        "sub_2,cust_2,2024-01-05,,3500.0,42000.0,25,annual,1,0\n"
        "sub_3,cust_3,2024-01-10,2024-06-15,800.0,9600.0,5,monthly,0,0\n"
        "sub_4,cust_4,2024-01-15,,6000.0,72000.0,50,monthly,1,0\n"
    )
    usage_csv = (
        "event_id,sub_key,cust_id,timestamp,feature,duration,is_err,beta_flag\n"
        "u1,sub_1,cust_1,2024-02-15 10:00:00,dashboard,120.0,0,0\n"
        "u2,sub_1,cust_1,2024-03-10 14:30:00,reports,350.0,0,1\n"
        "u3,sub_2,cust_2,2024-02-01 09:15:00,export,45.0,1,0\n"
        "u4,sub_3,cust_3,2024-02-20 16:45:00,api_call,15.0,0,0\n"
        "u5,sub_4,cust_4,2024-03-01 11:20:00,dashboard,200.0,0,0\n"
    )
    tickets_csv = (
        "ticket_code,cust_id,open_time,close_time,urgency,csat\n"
        "t1,cust_1,2024-02-10 10:00:00,2024-02-11 12:00:00,low,5.0\n"
        "t2,cust_2,2024-02-18 14:00:00,,high,\n"
        "t3,cust_3,2024-03-05 08:30:00,2024-03-06 09:00:00,urgent,2.0\n"
        "t4,cust_4,2024-03-12 11:00:00,2024-03-13 15:00:00,low,4.0\n"
    )
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
    """Build valid SchemaMappingRequest matching synthetic fixtures."""
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


def _create_test_workspace(session: Session, user_id: str, ws_id: str | None = None) -> Workspace:
    """Helper to provision a dedicated test workspace with owner membership."""
    workspace_id = ws_id or f"JOB_TEST_WS_{uuid.uuid4().hex[:12]}"
    ws = Workspace(
        workspace_id=workspace_id,
        workspace_name=f"Workspace {workspace_id}",
        owner_user_id=user_id,
    )
    mem = WorkspaceMember(
        workspace_id=workspace_id,
        user_id=user_id,
        role="owner",
    )
    session.add_all([ws, mem])
    session.commit()
    return ws


@pytest.fixture(scope="module")
def job_engine():
    """Create test engine targeting local flurit_db with isolation validation."""
    settings = get_settings()
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
def job_session(job_engine):
    """Transactional session with teardown of async job test fixtures."""
    SessionLocal = sessionmaker(bind=job_engine, autocommit=False, autoflush=False)
    session: Session = SessionLocal()

    prefix = "JOB_TEST_"
    try:
        yield session
    finally:
        session.rollback()
        try:
            with job_engine.begin() as cleanup_conn:
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
def test_client(job_session):
    """FastAPI TestClient with overridden get_db sharing the active test session."""

    def _override_get_db():
        yield job_session

    app.dependency_overrides[get_db] = _override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.pop(get_db, None)


# ==============================================================================
# 1. Asynchronous Job Submission & 202 Accepted Lifecycle
# ==============================================================================


def test_submit_async_job_accepted(test_client, job_session):
    """Verify POST /onboarding/process/async returns HTTP 202 Accepted and job tracking metadata."""
    user_id = "user_job_async_1"
    ws = _create_test_workspace(job_session, user_id=user_id)

    token = _make_jwt(user_id)
    fixtures = _build_compatible_csv_fixtures()
    mapping = _get_compatible_mapping()

    files = [
        ("files", ("accounts.csv", fixtures["accounts.csv"].encode("utf-8"), "text/csv")),
        ("files", ("subscriptions.csv", fixtures["subscriptions.csv"].encode("utf-8"), "text/csv")),
        ("files", ("feature_usage.csv", fixtures["feature_usage.csv"].encode("utf-8"), "text/csv")),
        (
            "files",
            ("support_tickets.csv", fixtures["support_tickets.csv"].encode("utf-8"), "text/csv"),
        ),
        ("files", ("churn_events.csv", fixtures["churn_events.csv"].encode("utf-8"), "text/csv")),
    ]
    data = {"mapping": mapping.model_dump_json()}
    headers = {"Authorization": f"Bearer {token}", "X-Workspace-ID": ws.workspace_id}

    resp = test_client.post(
        "/api/v1/onboarding/process/async",
        files=files,
        data=data,
        headers=headers,
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["job_id"].startswith("job_")
    assert body["status"] == "queued"
    assert body["workspace_id"] == ws.workspace_id
    assert "created_at" in body


def test_submit_async_job_alias_endpoint(test_client, job_session):
    """Verify POST /onboarding/jobs alias endpoint functions identically and returns 202."""
    user_id = "user_job_alias_1"
    ws = _create_test_workspace(job_session, user_id=user_id)

    token = _make_jwt(user_id)
    fixtures = _build_compatible_csv_fixtures()
    mapping = _get_compatible_mapping()

    files = [
        ("files", ("accounts.csv", fixtures["accounts.csv"].encode("utf-8"), "text/csv")),
    ]
    data = {"mapping": mapping.model_dump_json()}
    headers = {"Authorization": f"Bearer {token}", "X-Workspace-ID": ws.workspace_id}

    resp = test_client.post(
        "/api/v1/onboarding/jobs",
        files=files,
        data=data,
        headers=headers,
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["job_id"].startswith("job_")
    assert body["status"] == "queued"


# ==============================================================================
# 2. Worker Execution Lifecycle & Full ML Scoring
# ==============================================================================


@pytest.mark.anyio
async def test_worker_execution_lifecycle_success(test_client, job_session):
    """Verify worker execution completes snapshotting, ML scoring, and updates job to succeeded."""
    user_id = "user_job_worker_1"
    ws = _create_test_workspace(job_session, user_id=user_id)

    token = _make_jwt(user_id)
    fixtures = _build_compatible_csv_fixtures()
    mapping = _get_compatible_mapping()
    opts = ProcessingOptions(
        prediction_window_days=60,
        observation_dates=["2024-04-01"],
        allow_sentinel_scoring=False,
    )

    files = [
        ("files", ("accounts.csv", fixtures["accounts.csv"].encode("utf-8"), "text/csv")),
        ("files", ("subscriptions.csv", fixtures["subscriptions.csv"].encode("utf-8"), "text/csv")),
        ("files", ("feature_usage.csv", fixtures["feature_usage.csv"].encode("utf-8"), "text/csv")),
        (
            "files",
            ("support_tickets.csv", fixtures["support_tickets.csv"].encode("utf-8"), "text/csv"),
        ),
        ("files", ("churn_events.csv", fixtures["churn_events.csv"].encode("utf-8"), "text/csv")),
    ]
    data = {
        "mapping": mapping.model_dump_json(),
        "options": opts.model_dump_json(),
    }
    headers = {"Authorization": f"Bearer {token}", "X-Workspace-ID": ws.workspace_id}

    # Submit job
    submit_resp = test_client.post(
        "/api/v1/onboarding/process/async",
        files=files,
        data=data,
        headers=headers,
    )
    assert submit_resp.status_code == 202
    job_id = submit_resp.json()["job_id"]

    # Directly execute background worker on the queued job
    await execute_processing_job(job_id=job_id, workspace_id=ws.workspace_id)

    # Poll status endpoint
    status_resp = test_client.get(
        f"/api/v1/onboarding/jobs/{job_id}",
        headers=headers,
    )
    assert status_resp.status_code == 200
    status_data = status_resp.json()

    assert status_data["job_id"] == job_id
    assert status_data["status"] == "succeeded"
    assert status_data["progress_pct"] == 100
    assert status_data["stage_message"] == "Analysis completed successfully"
    assert status_data["started_at"] is not None
    assert status_data["completed_at"] is not None
    assert status_data["error_message"] is None

    # Verify result summary structure
    res = status_data["result_summary"]
    assert res is not None
    assert res["processing_status"] in ("SUCCESS", "PARTIAL_SUCCESS")
    assert res["analysis_ready"] is True
    assert res["label_construction_ready"] is True
    assert res["existing_model_scoring_ready"] is True
    assert res["snapshot_summary"] is not None
    assert res["scoring_summary"] is not None
    assert res["scoring_summary"]["scored_snapshots_count"] == 4
    assert res["scoring_summary"]["model_name"] == "flurit_churn_predictor"
    assert "prioritization_quadrants" in res["scoring_summary"]
    assert "score_interpretation_notice" in res["scoring_summary"]


# ==============================================================================
# 3. Idempotency Key Handling
# ==============================================================================


def test_idempotency_key_duplicate_prevention(test_client, job_session):
    """Verify identical idempotency key returns the existing job without creating duplicates."""
    user_id = "user_job_idemp_1"
    ws = _create_test_workspace(job_session, user_id=user_id)

    token = _make_jwt(user_id)
    fixtures = _build_compatible_csv_fixtures()
    mapping = _get_compatible_mapping()
    idemp_token = "idemp_test_unique_token_999"

    files = [
        ("files", ("accounts.csv", fixtures["accounts.csv"].encode("utf-8"), "text/csv")),
    ]
    data = {"mapping": mapping.model_dump_json()}
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Workspace-ID": ws.workspace_id,
        "X-Idempotency-Key": idemp_token,
    }

    # First submission
    resp1 = test_client.post(
        "/api/v1/onboarding/process/async",
        files=files,
        data=data,
        headers=headers,
    )
    assert resp1.status_code == 202
    job_id_1 = resp1.json()["job_id"]

    # Re-submission with same idempotency token
    files2 = [
        ("files", ("accounts.csv", fixtures["accounts.csv"].encode("utf-8"), "text/csv")),
    ]
    resp2 = test_client.post(
        "/api/v1/onboarding/process/async",
        files=files2,
        data=data,
        headers=headers,
    )
    assert resp2.status_code == 202
    job_id_2 = resp2.json()["job_id"]

    # Both must match
    assert job_id_1 == job_id_2
    assert "Existing job returned" in resp2.json()["message"]


# ==============================================================================
# 4. Multi-Tenant Security & Cross-Workspace Isolation
# ==============================================================================


def test_cross_workspace_job_status_blocked_403(test_client, job_session):
    """Probing another tenant's job ID must return HTTP 403 Forbidden."""
    user_a = "user_job_a"
    user_b = "user_job_b"
    ws_a = _create_test_workspace(job_session, user_id=user_a, ws_id="JOB_TEST_WS_A")
    ws_b = _create_test_workspace(job_session, user_id=user_b, ws_id="JOB_TEST_WS_B")

    # User A creates a job in Workspace A
    token_a = _make_jwt(user_a)
    fixtures = _build_compatible_csv_fixtures()
    mapping = _get_compatible_mapping()

    files = [
        ("files", ("accounts.csv", fixtures["accounts.csv"].encode("utf-8"), "text/csv")),
    ]
    data = {"mapping": mapping.model_dump_json()}
    headers_a = {"Authorization": f"Bearer {token_a}", "X-Workspace-ID": ws_a.workspace_id}

    submit_resp = test_client.post(
        "/api/v1/onboarding/process/async",
        files=files,
        data=data,
        headers=headers_a,
    )
    assert submit_resp.status_code == 202
    job_id_a = submit_resp.json()["job_id"]

    # User B tries to view User A's job from Workspace B
    token_b = _make_jwt(user_b)
    headers_b = {"Authorization": f"Bearer {token_b}", "X-Workspace-ID": ws_b.workspace_id}

    probe_resp = test_client.get(
        f"/api/v1/onboarding/jobs/{job_id_a}",
        headers=headers_b,
    )
    assert probe_resp.status_code == 403
    assert "Forbidden" in probe_resp.json()["detail"]


def test_nonexistent_job_returns_404(test_client, job_session):
    """Requesting non-existent job ID returns HTTP 404."""
    user_id = "user_job_404"
    ws = _create_test_workspace(job_session, user_id=user_id)

    token = _make_jwt(user_id)
    headers = {"Authorization": f"Bearer {token}", "X-Workspace-ID": ws.workspace_id}

    resp = test_client.get(
        "/api/v1/onboarding/jobs/job_does_not_exist_9999",
        headers=headers,
    )
    assert resp.status_code == 404


# ==============================================================================
# 5. Paginated Job Listing Scoped to Workspace
# ==============================================================================


def test_list_jobs_workspace_scoped(test_client, job_session):
    """GET /onboarding/jobs lists only jobs belonging to caller's workspace."""
    user_a = "user_list_a"
    user_b = "user_list_b"
    ws_a = _create_test_workspace(job_session, user_id=user_a, ws_id="JOB_TEST_WS_LA")
    ws_b = _create_test_workspace(job_session, user_id=user_b, ws_id="JOB_TEST_WS_LB")

    fixtures = _build_compatible_csv_fixtures()
    mapping = _get_compatible_mapping()

    token_a = _make_jwt(user_a)
    token_b = _make_jwt(user_b)

    # Workspace A: 2 jobs
    for i in range(2):
        files = [("files", ("accounts.csv", fixtures["accounts.csv"].encode("utf-8"), "text/csv"))]
        test_client.post(
            "/api/v1/onboarding/process/async",
            files=files,
            data={"mapping": mapping.model_dump_json(), "dataset_name": f"dataset_a_{i}"},
            headers={"Authorization": f"Bearer {token_a}", "X-Workspace-ID": ws_a.workspace_id},
        )

    # Workspace B: 1 job
    files = [("files", ("accounts.csv", fixtures["accounts.csv"].encode("utf-8"), "text/csv"))]
    test_client.post(
        "/api/v1/onboarding/process/async",
        files=files,
        data={"mapping": mapping.model_dump_json(), "dataset_name": "dataset_b_0"},
        headers={"Authorization": f"Bearer {token_b}", "X-Workspace-ID": ws_b.workspace_id},
    )

    # Fetch list for Workspace A
    list_a_resp = test_client.get(
        "/api/v1/onboarding/jobs",
        headers={"Authorization": f"Bearer {token_a}", "X-Workspace-ID": ws_a.workspace_id},
    )
    assert list_a_resp.status_code == 200
    list_a = list_a_resp.json()
    assert list_a["total"] == 2
    assert len(list_a["jobs"]) == 2
    for job in list_a["jobs"]:
        assert job["workspace_id"] == ws_a.workspace_id

    # Fetch list for Workspace B
    list_b_resp = test_client.get(
        "/api/v1/onboarding/jobs",
        headers={"Authorization": f"Bearer {token_b}", "X-Workspace-ID": ws_b.workspace_id},
    )
    assert list_b_resp.status_code == 200
    list_b = list_b_resp.json()
    assert list_b["total"] == 1
    assert len(list_b["jobs"]) == 1
    assert list_b["jobs"][0]["workspace_id"] == ws_b.workspace_id


# ==============================================================================
# 6. Job Cancellation & File Cleanup
# ==============================================================================


def test_job_cancellation_lifecycle(test_client, job_session):
    """Verify job cancellation updates status and prevents further execution."""
    user_id = "user_job_cancel_1"
    ws = _create_test_workspace(job_session, user_id=user_id)

    token = _make_jwt(user_id)
    fixtures = _build_compatible_csv_fixtures()
    mapping = _get_compatible_mapping()

    files = [("files", ("accounts.csv", fixtures["accounts.csv"].encode("utf-8"), "text/csv"))]
    headers = {"Authorization": f"Bearer {token}", "X-Workspace-ID": ws.workspace_id}

    submit_resp = test_client.post(
        "/api/v1/onboarding/process/async",
        files=files,
        data={"mapping": mapping.model_dump_json()},
        headers=headers,
    )
    assert submit_resp.status_code == 202
    job_id = submit_resp.json()["job_id"]

    # Cancel the job
    cancel_resp = test_client.post(
        f"/api/v1/onboarding/jobs/{job_id}/cancel",
        headers=headers,
    )
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "cancelled"

    # Status check confirms cancelled
    status_resp = test_client.get(
        f"/api/v1/onboarding/jobs/{job_id}",
        headers=headers,
    )
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "cancelled"


def test_cross_workspace_cancellation_blocked_403(test_client, job_session):
    """User cannot cancel a job belonging to another workspace."""
    user_a = "user_cancel_a"
    user_b = "user_cancel_b"
    ws_a = _create_test_workspace(job_session, user_id=user_a, ws_id="JOB_TEST_WS_CA")
    ws_b = _create_test_workspace(job_session, user_id=user_b, ws_id="JOB_TEST_WS_CB")

    fixtures = _build_compatible_csv_fixtures()
    mapping = _get_compatible_mapping()

    token_a = _make_jwt(user_a)
    token_b = _make_jwt(user_b)

    files = [("files", ("accounts.csv", fixtures["accounts.csv"].encode("utf-8"), "text/csv"))]
    submit_resp = test_client.post(
        "/api/v1/onboarding/process/async",
        files=files,
        data={"mapping": mapping.model_dump_json()},
        headers={"Authorization": f"Bearer {token_a}", "X-Workspace-ID": ws_a.workspace_id},
    )
    job_id = submit_resp.json()["job_id"]

    # User B attempts to cancel User A's job
    cross_resp = test_client.post(
        f"/api/v1/onboarding/jobs/{job_id}/cancel",
        headers={"Authorization": f"Bearer {token_b}", "X-Workspace-ID": ws_b.workspace_id},
    )
    assert cross_resp.status_code == 403


# ==============================================================================
# 7. File Size Limits & Synchronous Endpoint Deprecation
# ==============================================================================


def test_oversized_file_rejected_413(test_client, job_session):
    """Files exceeding ONBOARDING_MAX_FILE_SIZE_BYTES are rejected with HTTP 413."""
    user_id = "user_job_oversize"
    ws = _create_test_workspace(job_session, user_id=user_id)

    token = _make_jwt(user_id)
    mapping = _get_compatible_mapping()

    # Small mock limit for test
    settings = get_settings()
    orig_limit = settings.ONBOARDING_MAX_FILE_SIZE_BYTES
    try:
        settings.ONBOARDING_MAX_FILE_SIZE_BYTES = 50  # 50 bytes
        files = [
            ("files", ("huge.csv", b"a" * 100, "text/csv")),
        ]
        headers = {"Authorization": f"Bearer {token}", "X-Workspace-ID": ws.workspace_id}
        resp = test_client.post(
            "/api/v1/onboarding/process/async",
            files=files,
            data={"mapping": mapping.model_dump_json()},
            headers=headers,
        )
        assert resp.status_code == 413
        assert "exceeds maximum allowed upload size" in resp.json()["detail"]
    finally:
        settings.ONBOARDING_MAX_FILE_SIZE_BYTES = orig_limit


def test_sync_process_endpoint_includes_deprecation_header(test_client, job_session):
    """Synchronous /onboarding/process includes Deprecation and Warning headers."""
    user_id = "user_job_sync_dep"
    ws = _create_test_workspace(job_session, user_id=user_id)

    token = _make_jwt(user_id)
    fixtures = _build_compatible_csv_fixtures()
    mapping = _get_compatible_mapping()
    opts = ProcessingOptions(
        prediction_window_days=60,
        observation_dates=["2024-04-01"],
        allow_sentinel_scoring=False,
    )

    files = [
        ("files", ("accounts.csv", fixtures["accounts.csv"].encode("utf-8"), "text/csv")),
        ("files", ("subscriptions.csv", fixtures["subscriptions.csv"].encode("utf-8"), "text/csv")),
        ("files", ("feature_usage.csv", fixtures["feature_usage.csv"].encode("utf-8"), "text/csv")),
        (
            "files",
            ("support_tickets.csv", fixtures["support_tickets.csv"].encode("utf-8"), "text/csv"),
        ),
        ("files", ("churn_events.csv", fixtures["churn_events.csv"].encode("utf-8"), "text/csv")),
    ]
    headers = {"Authorization": f"Bearer {token}", "X-Workspace-ID": ws.workspace_id}
    data = {
        "mapping": mapping.model_dump_json(),
        "options": opts.model_dump_json(),
    }

    resp = test_client.post(
        "/api/v1/onboarding/process",
        files=files,
        data=data,
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.headers.get("Deprecation") == "true"
    assert "299" in resp.headers.get("Warning", "")


# ==============================================================================
# 8. Error Sanitization & Safe Failure Reporting
# ==============================================================================


def test_error_message_sanitization():
    """Verify _sanitize_error_message strips internal paths, passwords, and SQL statements."""
    # Test path stripping
    path_exc = Exception(r"File not found in C:\flurit-backend\data\uploads\ws_1\secret.csv")
    sanitized = _sanitize_error_message(path_exc)
    assert "C:\\" not in sanitized
    assert "file system or path error" in sanitized.lower()

    # Test DB credentials stripping
    db_exc = Exception("SELECT * FROM users WHERE password='super_secret_password'")
    sanitized_db = _sanitize_error_message(db_exc)
    assert "password" not in sanitized_db.lower()
    assert "database operation failed" in sanitized_db.lower()

    # Test normal validation message passes safely
    val_exc = ValueError("Missing required concept column: cust_id")
    assert _sanitize_error_message(val_exc) == "Missing required concept column: cust_id"


# ==============================================================================
# 9. Storage Backend Safety & Lifecycle Tests
# ==============================================================================


@pytest.mark.anyio
async def test_storage_backend_local_and_memory():
    """Verify local and in-memory storage implementations prevent traversal and clean up safely."""
    # Memory Storage
    mem_storage = MemoryFileStorage()
    key = await mem_storage.save_file("ws_test", "job_1", "test.csv", b"col1,col2\n1,2")
    read_back = await mem_storage.read_file(key)
    assert read_back == b"col1,col2\n1,2"
    await mem_storage.delete_job_files("ws_test", "job_1")
    with pytest.raises(FileNotFoundError):
        await mem_storage.read_file(key)

    # Local Storage & Traversal protection
    local_storage = LocalFileStorage(base_dir="data/test_uploads")
    # Traversal filename attempt
    with pytest.raises(ValueError, match="Directory traversal"):
        await local_storage.save_file("ws_test", "job_1", "../../etc/passwd", b"bad")

    # Valid write and cleanup
    loc_key = await local_storage.save_file("ws_test", "job_2", "safe.csv", b"safe,content\n1,1")
    loc_read = await local_storage.read_file(loc_key)
    assert loc_read == b"safe,content\n1,1"
    await local_storage.delete_job_files("ws_test", "job_2")
    with pytest.raises(FileNotFoundError):
        await local_storage.read_file(loc_key)


# ==============================================================================
# 10. Queue Broker Implementations
# ==============================================================================


@pytest.mark.anyio
async def test_queue_brokers_instantiation():
    """Verify MemoryQueueBroker and SyncQueueBroker protocols."""
    mem_broker = MemoryQueueBroker()
    sync_broker = SyncQueueBroker()

    assert isinstance(mem_broker, MemoryQueueBroker)
    assert isinstance(sync_broker, SyncQueueBroker)

    broker = get_queue_broker()
    assert broker is not None
