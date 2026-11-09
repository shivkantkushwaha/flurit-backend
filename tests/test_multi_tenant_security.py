"""Comprehensive Multi-Tenant Data Isolation and Security Test Suite for Flurit.

Verifies:
1. Server-side Supabase user ('sub') to workspace mapping and membership verification.
2. Cross-workspace request blocking (HTTP 403 Forbidden) when accessing unauthorized workspaces.
3. Cross-workspace customer probe protection (HTTP 403) when guessing account IDs from other tenants.
4. Independent co-existence of overlapping account IDs across workspaces without collisions.
5. Idempotent prediction persistence scoped strictly to the authorized tenant workspace.
6. Database-level unique constraints preventing duplicate predictions and snapshots per workspace.
"""

import time
from datetime import date

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker
from starlette.testclient import TestClient

from app.core.auth import create_jwt_token
from app.core.config import get_settings
from app.core.workspace import get_or_create_default_workspace, verify_user_workspace_access
from app.db.session import get_db, validate_database_url_isolation
from app.main import app
from app.ml.inference.batch import BatchPredictionOutput
from app.models import (
    Account,
    CustomerPrediction,
    CustomerSnapshot,
    Workspace,
    WorkspaceMember,
)
from app.repositories.prediction_repository import PredictionRepository

TEST_SECRET = "flurit-secure-test-jwt-secret-key-32-chars-min!"


def _make_jwt(user_id: str, email: str = "user@example.com") -> str:
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


def _get_valid_feature_payload() -> dict:
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


@pytest.fixture(scope="module")
def sec_engine():
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
def sec_session(sec_engine):
    """Transactional session with teardown of security test fixtures."""
    SessionLocal = sessionmaker(bind=sec_engine, autocommit=False, autoflush=False)
    session: Session = SessionLocal()

    prefix = "SEC_"
    try:
        yield session
    finally:
        session.rollback()
        try:
            with sec_engine.begin() as cleanup_conn:
                cleanup_conn.execute(
                    sa.text(f"DELETE FROM accounts WHERE workspace_id LIKE '{prefix}%';")
                )
                cleanup_conn.execute(
                    sa.text(f"DELETE FROM workspaces WHERE workspace_id LIKE '{prefix}%';")
                )
                cleanup_conn.execute(
                    sa.text("DELETE FROM model_runs WHERE model_name = 'sec_test_model';")
                )
        except Exception:
            pass
        session.close()


@pytest.fixture
def test_client(sec_session):
    """FastAPI TestClient with overridden get_db to share active session."""

    def _override_get_db():
        yield sec_session

    app.dependency_overrides[get_db] = _override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.pop(get_db, None)


# ==============================================================================
# 1. Server-Side Workspace Mapping & Cross-Workspace Header Authorization
# ==============================================================================


def test_workspace_auto_provisioning_and_membership_check(sec_session):
    """Verify get_or_create_default_workspace provisions a personal workspace for new users."""
    user_id = "user_sec_new_123"
    ws, member = get_or_create_default_workspace(
        sec_session, user_id=user_id, email="alice@corp.com"
    )
    sec_session.commit()

    assert ws.workspace_id.startswith("ws_")
    assert ws.owner_user_id == user_id
    assert member.user_id == user_id
    assert member.role == "owner"

    # Verifying membership lookup
    authorized_member = verify_user_workspace_access(
        sec_session, user_id=user_id, workspace_id=ws.workspace_id
    )
    assert authorized_member is not None
    assert authorized_member.role == "owner"

    # Verifying unauthorized user
    unauthorized = verify_user_workspace_access(
        sec_session, user_id="intruder_999", workspace_id=ws.workspace_id
    )
    assert unauthorized is None


def test_cross_workspace_header_access_blocked_with_403(test_client, sec_session):
    """User A attempting to access Workspace B via X-Workspace-ID header must be rejected with 403."""
    user_a = "user_sec_a"
    user_b = "user_sec_b"
    ws_a_id = "SEC_WS_A"
    ws_b_id = "SEC_WS_B"

    # Create distinct workspaces and memberships
    ws_a = Workspace(workspace_id=ws_a_id, workspace_name="Workspace A", owner_user_id=user_a)
    ws_b = Workspace(workspace_id=ws_b_id, workspace_name="Workspace B", owner_user_id=user_b)
    mem_a = WorkspaceMember(workspace_id=ws_a_id, user_id=user_a, role="owner")
    mem_b = WorkspaceMember(workspace_id=ws_b_id, user_id=user_b, role="owner")

    sec_session.add_all([ws_a, ws_b, mem_a, mem_b])
    sec_session.commit()

    token_a = _make_jwt(user_id=user_a)

    # 1. User A accessing Workspace A -> 200 OK
    resp_valid = test_client.post(
        "/api/v1/predictions/score",
        json={"features": _get_valid_feature_payload()},
        headers={"Authorization": f"Bearer {token_a}", "X-Workspace-ID": ws_a_id},
    )
    assert resp_valid.status_code == 200

    # 2. User A attempting to access Workspace B -> 403 Forbidden
    resp_forbidden = test_client.post(
        "/api/v1/predictions/score",
        json={"features": _get_valid_feature_payload()},
        headers={"Authorization": f"Bearer {token_a}", "X-Workspace-ID": ws_b_id},
    )
    assert resp_forbidden.status_code == 403
    assert "Access denied" in resp_forbidden.json()["detail"]


# ==============================================================================
# 2. Cross-Workspace Customer Account ID Probe Protection
# ==============================================================================


def test_cross_workspace_customer_risk_probe_rejected(test_client, sec_session):
    """User B cannot probe customer risk for an account that belongs strictly to Workspace A."""
    user_a = "user_sec_probe_a"
    user_b = "user_sec_probe_b"
    ws_a_id = "SEC_WS_PROBE_A"
    ws_b_id = "SEC_WS_PROBE_B"
    secret_acc_id = "ACC_SECRET_ENTERPRISE_001"

    ws_a = Workspace(workspace_id=ws_a_id, workspace_name="Workspace A", owner_user_id=user_a)
    ws_b = Workspace(workspace_id=ws_b_id, workspace_name="Workspace B", owner_user_id=user_b)
    mem_a = WorkspaceMember(workspace_id=ws_a_id, user_id=user_a, role="owner")
    mem_b = WorkspaceMember(workspace_id=ws_b_id, user_id=user_b, role="owner")

    # Account exists in Workspace A
    account_a = Account(
        workspace_id=ws_a_id,
        account_id=secret_acc_id,
        account_name="Confidential Customer Inc",
        industry="Finance",
        country="US",
        signup_date=date(2024, 1, 1),
        referral_source="direct",
        plan_tier="Enterprise",
        seats=100,
        is_trial=False,
        churn_flag=False,
    )
    sec_session.add_all([ws_a, ws_b, mem_a, mem_b, account_a])
    sec_session.commit()

    token_b = _make_jwt(user_id=user_b)

    # User B (operating in Workspace B) tries to query risk for Workspace A's account
    resp = test_client.post(
        f"/api/v1/customers/{secret_acc_id}/risk",
        json={"features": _get_valid_feature_payload(), "snapshot_date": "2024-10-01"},
        headers={"Authorization": f"Bearer {token_b}", "X-Workspace-ID": ws_b_id},
    )
    assert resp.status_code == 403
    assert "does not belong to authorized workspace" in resp.json()["detail"]


# ==============================================================================
# 3. Overlapping Account Identifiers Across Workspaces Without Collisions
# ==============================================================================


def test_overlapping_account_ids_across_workspaces(sec_session):
    """External customer IDs (e.g. ACC_001) can overlap across workspaces without collisions."""
    ws_1 = "SEC_WS_OVERLAP_1"
    ws_2 = "SEC_WS_OVERLAP_2"
    shared_acc_id = "ACC_SHARED_SAME_ID"

    sec_session.add(Workspace(workspace_id=ws_1, workspace_name="Tenant 1", owner_user_id="user_1"))
    sec_session.add(Workspace(workspace_id=ws_2, workspace_name="Tenant 2", owner_user_id="user_2"))
    sec_session.flush()

    # Create account with SAME account_id in Tenant 1
    acc_1 = Account(
        workspace_id=ws_1,
        account_id=shared_acc_id,
        account_name="Tenant 1 Alpha Client",
        industry="Retail",
        country="US",
        signup_date=date(2024, 1, 1),
        referral_source="organic",
        plan_tier="Basic",
        seats=5,
        is_trial=False,
        churn_flag=False,
    )
    # Create account with SAME account_id in Tenant 2
    acc_2 = Account(
        workspace_id=ws_2,
        account_id=shared_acc_id,
        account_name="Tenant 2 Beta Client",
        industry="Healthcare",
        country="CA",
        signup_date=date(2024, 2, 1),
        referral_source="partner",
        plan_tier="Enterprise",
        seats=50,
        is_trial=False,
        churn_flag=False,
    )

    sec_session.add_all([acc_1, acc_2])
    sec_session.flush()

    # Create snapshots for SAME account_id on SAME date across both workspaces
    snap_1 = CustomerSnapshot(
        workspace_id=ws_1,
        account_id=shared_acc_id,
        snapshot_date=date(2024, 6, 1),
        is_active=True,
    )
    snap_2 = CustomerSnapshot(
        workspace_id=ws_2,
        account_id=shared_acc_id,
        snapshot_date=date(2024, 6, 1),
        is_active=True,
    )
    sec_session.add_all([snap_1, snap_2])
    sec_session.commit()

    # Query back: both must exist independently
    fetched_1 = sec_session.get(Account, (ws_1, shared_acc_id))
    fetched_2 = sec_session.get(Account, (ws_2, shared_acc_id))

    assert fetched_1 is not None
    assert fetched_2 is not None
    assert fetched_1.account_name == "Tenant 1 Alpha Client"
    assert fetched_2.account_name == "Tenant 2 Beta Client"
    assert fetched_1.industry == "Retail"
    assert fetched_2.industry == "Healthcare"


# ==============================================================================
# 4. Idempotent Prediction Persistence Scoped to Tenant Workspace
# ==============================================================================


def test_prediction_idempotency_and_tenant_isolation(sec_session):
    """Verify save_prediction_batch scopes updates to workspace and does not leak or cross-contaminate."""
    ws_1 = "SEC_WS_PRED_1"
    ws_2 = "SEC_WS_PRED_2"
    acc_id = "ACC_SHARED_PRED_001"

    sec_session.add(Workspace(workspace_id=ws_1, workspace_name="Tenant 1", owner_user_id="user_1"))
    sec_session.add(Workspace(workspace_id=ws_2, workspace_name="Tenant 2", owner_user_id="user_2"))
    sec_session.flush()

    repo = PredictionRepository()
    model_run = repo.get_or_create_model_run(
        session=sec_session,
        model_name="sec_test_model",
        version="v1.0.0-sec",
    )
    sec_session.flush()

    # Prediction for Tenant 1
    pred_t1 = BatchPredictionOutput(
        account_id=acc_id,
        snapshot_date="2024-10-01",
        snapshot_id=None,
        model_name="sec_test_model",
        model_version="v1.0.0-sec",
        model_score=0.25,
        predicted_risk_label=0,
        risk_tier="Low Risk",
        cluster_id=1,
        cluster_name="Healthy",
        value_tier="Mid Value",
        top_risk_increasing_factors=[{"feature_name": "tickets_count_30d", "attribution": 0.05}],
        top_risk_decreasing_factors=[{"feature_name": "usage_count_30d", "attribution": -0.2}],
        shap_values={"usage_count_30d": -0.2},
        plain_language_summary="Healthy account in Tenant 1",
        feature_payload={"sub_current_mrr": 2000.0},
    )

    # Prediction for Tenant 2 for the SAME account_id
    pred_t2 = BatchPredictionOutput(
        account_id=acc_id,
        snapshot_date="2024-10-01",
        snapshot_id=None,
        model_name="sec_test_model",
        model_version="v1.0.0-sec",
        model_score=0.89,
        predicted_risk_label=1,
        risk_tier="High Risk",
        cluster_id=2,
        cluster_name="At-Risk",
        value_tier="High Value",
        top_risk_increasing_factors=[{"feature_name": "tickets_count_30d", "attribution": 0.65}],
        top_risk_decreasing_factors=[],
        shap_values={"tickets_count_30d": 0.65},
        plain_language_summary="High risk account in Tenant 2",
        feature_payload={"sub_current_mrr": 9000.0},
    )

    repo.save_prediction_batch(sec_session, model_run.model_run_id, [pred_t1], workspace_id=ws_1)
    repo.save_prediction_batch(sec_session, model_run.model_run_id, [pred_t2], workspace_id=ws_2)
    sec_session.commit()

    # Query back by workspace: Tenant 1 should have score 0.25, Tenant 2 should have score 0.89
    p1 = repo.get_latest_prediction_by_account(sec_session, acc_id, workspace_id=ws_1)
    p2 = repo.get_latest_prediction_by_account(sec_session, acc_id, workspace_id=ws_2)

    assert p1 is not None
    assert p2 is not None
    assert pytest.approx(p1.churn_probability, 0.001) == 0.25
    assert pytest.approx(p2.churn_probability, 0.001) == 0.89
    assert p1.risk_level == "Low Risk"
    assert p2.risk_level == "High Risk"

    # Idempotent re-write in Tenant 1 with updated score
    pred_t1_updated = BatchPredictionOutput(
        account_id=acc_id,
        snapshot_date="2024-10-01",
        snapshot_id=None,
        model_name="sec_test_model",
        model_version="v1.0.0-sec",
        model_score=0.33,
        predicted_risk_label=0,
        risk_tier="Low Risk",
        cluster_id=1,
        cluster_name="Healthy",
        value_tier="Mid Value",
        top_risk_increasing_factors=[],
        top_risk_decreasing_factors=[],
        shap_values={},
        plain_language_summary="Updated score for Tenant 1",
        feature_payload={"sub_current_mrr": 2200.0},
    )
    repo.save_prediction_batch(
        sec_session, model_run.model_run_id, [pred_t1_updated], workspace_id=ws_1
    )
    sec_session.commit()

    # Verify Tenant 1 updated to 0.33, Tenant 2 remains 0.89 unchanged
    p1_after = repo.get_latest_prediction_by_account(sec_session, acc_id, workspace_id=ws_1)
    p2_after = repo.get_latest_prediction_by_account(sec_session, acc_id, workspace_id=ws_2)

    assert pytest.approx(p1_after.churn_probability, 0.001) == 0.33
    assert pytest.approx(p2_after.churn_probability, 0.001) == 0.89

    # Verify exactly 1 prediction in ws_1 and 1 prediction in ws_2
    t1_preds = repo.get_predictions_by_model_run(
        sec_session, model_run.model_run_id, workspace_id=ws_1
    )
    t2_preds = repo.get_predictions_by_model_run(
        sec_session, model_run.model_run_id, workspace_id=ws_2
    )
    assert len(t1_preds) == 1
    assert len(t2_preds) == 1


# ==============================================================================
# 5. Database-Level Multi-Tenant Unique Constraint Enforcement
# ==============================================================================


def test_database_prediction_unique_constraint_enforced(sec_session):
    """Direct duplicate insertion of (workspace_id, model_run_id, account_id, snapshot_id) must raise IntegrityError."""
    ws_id = "SEC_WS_UQ_TEST"
    acc_id = "ACC_UQ_TEST_001"

    sec_session.add(
        Workspace(workspace_id=ws_id, workspace_name="UQ Test WS", owner_user_id="user_uq")
    )
    sec_session.flush()

    account = Account(
        workspace_id=ws_id,
        account_id=acc_id,
        account_name="UQ Test Corp",
        industry="Tech",
        country="US",
        signup_date=date(2024, 1, 1),
        referral_source="direct",
        plan_tier="Pro",
        seats=10,
        is_trial=False,
        churn_flag=False,
    )
    sec_session.add(account)
    sec_session.flush()

    snapshot = CustomerSnapshot(
        workspace_id=ws_id,
        account_id=acc_id,
        snapshot_date=date(2024, 1, 1),
        is_active=True,
    )
    sec_session.add(snapshot)
    sec_session.flush()

    repo = PredictionRepository()
    model_run = repo.get_or_create_model_run(
        session=sec_session,
        model_name="sec_test_model",
        version="v1.0.0-sec",
    )
    sec_session.flush()

    pred1 = CustomerPrediction(
        workspace_id=ws_id,
        model_run_id=model_run.model_run_id,
        account_id=acc_id,
        snapshot_id=snapshot.snapshot_id,
        churn_probability=0.45,
        risk_level="Low Risk",
    )
    sec_session.add(pred1)
    sec_session.flush()

    # Attempt to insert identical prediction record in same workspace
    pred2 = CustomerPrediction(
        workspace_id=ws_id,
        model_run_id=model_run.model_run_id,
        account_id=acc_id,
        snapshot_id=snapshot.snapshot_id,
        churn_probability=0.75,
        risk_level="High Risk",
    )
    sec_session.add(pred2)

    with pytest.raises(sa.exc.IntegrityError) as exc_info:
        sec_session.flush()

    assert (
        "uq_prediction_workspace_model_account_snapshot" in str(exc_info.value)
        or "unique constraint" in str(exc_info.value).lower()
    )
    sec_session.rollback()


# ==============================================================================
# 6. Onboarding Endpoints Tenant Authorization
# ==============================================================================


def test_onboarding_endpoints_tenant_authorization(test_client, sec_session):
    """Verify onboarding endpoints enforce workspace ownership and block cross-tenant requests."""
    user_a = "user_onb_sec_a"
    ws_a_id = "SEC_WS_ONB_A"
    ws_b_id = "SEC_WS_ONB_B"

    ws_a = Workspace(workspace_id=ws_a_id, workspace_name="Onboarding WS A", owner_user_id=user_a)
    mem_a = WorkspaceMember(workspace_id=ws_a_id, user_id=user_a, role="owner")
    sec_session.add_all([ws_a, mem_a])
    sec_session.commit()

    token_a = _make_jwt(user_id=user_a)

    # 1. Profile with authorized workspace
    csv_bytes = b"account_id,signup_date\nACC_001,2024-01-01\n"
    files = [("files", ("accounts.csv", csv_bytes, "text/csv"))]
    resp_ok = test_client.post(
        "/api/v1/onboarding/profile",
        files=files,
        headers={"Authorization": f"Bearer {token_a}", "X-Workspace-ID": ws_a_id},
    )
    assert resp_ok.status_code == 200

    # 2. Profile with unauthorized workspace -> 403 Forbidden
    files = [("files", ("accounts.csv", csv_bytes, "text/csv"))]
    resp_blocked = test_client.post(
        "/api/v1/onboarding/profile",
        files=files,
        headers={"Authorization": f"Bearer {token_a}", "X-Workspace-ID": ws_b_id},
    )
    assert resp_blocked.status_code == 403

    # 3. Map schema with authorized workspace
    map_payload = {
        "dataset_name": "test_map",
        "accounts": {"account_id": "account_id", "signup_date": "signup_date"},
        "subscriptions": {
            "subscription_id": "sub_id",
            "account_id": "account_id",
            "start_date": "start_date",
        },
    }
    resp_map_ok = test_client.post(
        "/api/v1/onboarding/map",
        json=map_payload,
        headers={"Authorization": f"Bearer {token_a}", "X-Workspace-ID": ws_a_id},
    )
    assert resp_map_ok.status_code == 200

    # 4. Map schema with unauthorized workspace -> 403 Forbidden
    resp_map_blocked = test_client.post(
        "/api/v1/onboarding/map",
        json=map_payload,
        headers={"Authorization": f"Bearer {token_a}", "X-Workspace-ID": ws_b_id},
    )
    assert resp_map_blocked.status_code == 403


# ==============================================================================
# 7. Repository Stub Account Isolation
# ==============================================================================


def test_ensure_accounts_exist_workspace_isolation(sec_session):
    """Verify ensure_accounts_exist scopes account checks and creation to the specified workspace."""
    ws_1 = "SEC_WS_STUB_1"
    ws_2 = "SEC_WS_STUB_2"
    sec_session.add(
        Workspace(workspace_id=ws_1, workspace_name="Stub WS 1", owner_user_id="user_1")
    )
    sec_session.add(
        Workspace(workspace_id=ws_2, workspace_name="Stub WS 2", owner_user_id="user_2")
    )
    sec_session.flush()

    repo = PredictionRepository()
    stub_acc = "ACC_STUB_ISOLATED"

    # Create stub in Workspace 1
    count1 = repo.ensure_accounts_exist(sec_session, [stub_acc], workspace_id=ws_1)
    assert count1 == 1

    # In Workspace 2, account should NOT exist yet
    acc_in_ws2 = sec_session.get(Account, (ws_2, stub_acc))
    assert acc_in_ws2 is None

    # Now create stub in Workspace 2
    count2 = repo.ensure_accounts_exist(sec_session, [stub_acc], workspace_id=ws_2)
    assert count2 == 1

    # Re-running in Workspace 1 creates 0 new accounts (idempotent)
    count1_repeat = repo.ensure_accounts_exist(sec_session, [stub_acc], workspace_id=ws_1)
    assert count1_repeat == 0


# ==============================================================================
# 8. Transformer & Processor Workspace Propagation
# ==============================================================================


def test_transformer_and_processor_propagate_workspace_id():
    """Verify SchemaTransformer and DatasetProcessingEngine tag canonical frames with workspace_id."""
    import pandas as pd

    from app.onboarding.processor import DatasetProcessingEngine
    from app.onboarding.schemas import (
        AccountConceptMapping,
        SchemaMappingRequest,
        SubscriptionConceptMapping,
    )
    from app.onboarding.transformer import SchemaTransformer

    target_ws = "SEC_WS_CANONICAL_TEST"
    transformer = SchemaTransformer()

    raw_dfs = {
        "accounts": pd.DataFrame(
            {"aid": ["ACC_T1", "ACC_T2"], "sdate": ["2024-01-01", "2024-01-15"]}
        ),
        "subscriptions": pd.DataFrame(
            {
                "sid": ["SUB_T1", "SUB_T2"],
                "aid": ["ACC_T1", "ACC_T2"],
                "sdate": ["2024-01-01", "2024-01-15"],
                "edate": [None, None],
                "mrr": [1000.0, 2000.0],
            }
        ),
    }
    mapping = SchemaMappingRequest(
        dataset_name="canonical_test",
        accounts=AccountConceptMapping(account_id="aid", signup_date="sdate"),
        subscriptions=SubscriptionConceptMapping(
            subscription_id="sid",
            account_id="aid",
            start_date="sdate",
            end_date="edate",
            mrr_amount="mrr",
        ),
    )

    canonical = transformer.transform(raw_dfs, mapping, workspace_id=target_ws)
    assert canonical.workspace_id == target_ws
    assert "workspace_id" in canonical.accounts_df.columns
    assert (canonical.accounts_df["workspace_id"] == target_ws).all()
    assert "workspace_id" in canonical.subscriptions_df.columns
    assert (canonical.subscriptions_df["workspace_id"] == target_ws).all()

    # Process preview
    processor = DatasetProcessingEngine()
    preview = processor.preview(canonical)
    assert preview.analysis_ready is True
