"""Integration tests verifying persistence against the local dedicated PostgreSQL database (flurit_db).

CRITICAL SECURITY AND ISOLATION:
- Connects ONLY to localhost:5432/flurit_db using the dedicated flurit_app role.
- Never connects to or references bynalt, bynalt_test, or postgres.
- Uses synthetic test data only; never imports or touches raw CSVs.
- Cleans up synthetic records upon completion.
"""

from datetime import date

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.db.session import validate_database_url_isolation
from app.ml.inference.batch import BatchPredictionOutput
from app.models import (
    Account,
    CustomerExplanation,
    CustomerPrediction,
    CustomerSegment,
    CustomerSnapshot,
    Subscription,
)
from app.repositories.prediction_repository import PredictionRepository


@pytest.fixture(scope="module")
def pg_engine():
    """Create an engine targeting only flurit_db using flurit_app."""
    settings = get_settings()
    url = settings.DATABASE_URL
    validate_database_url_isolation(url)

    if not url.startswith("postgresql"):
        pytest.skip("Local PostgreSQL is not configured in DATABASE_URL")

    engine = sa.create_engine(url)
    try:
        with engine.connect() as conn:
            db_name, curr_user = conn.execute(
                sa.text("SELECT current_database(), current_user;")
            ).fetchone()
            assert db_name == "flurit_db", f"Expected flurit_db, got {db_name}"
            assert curr_user == "flurit_app", f"Expected flurit_app, got {curr_user}"
    except Exception as exc:
        pytest.skip(f"Could not connect to local PostgreSQL: {exc}")

    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def pg_session(pg_engine):
    """Provide a transactional session for PostgreSQL testing with synthetic cleanup."""
    SessionLocal = sessionmaker(bind=pg_engine, autocommit=False, autoflush=False)
    session: Session = SessionLocal()

    # Prefix for all synthetic test data to facilitate clean teardown
    test_acc_prefix = "ACC_SYNTH_PG_"

    try:
        yield session
    finally:
        session.rollback()
        # Clean up any test accounts created during tests (cascades to subscriptions, snapshots, predictions)
        try:
            with pg_engine.begin() as cleanup_conn:
                cleanup_conn.execute(
                    sa.text(f"DELETE FROM accounts WHERE account_id LIKE '{test_acc_prefix}%';")
                )
                cleanup_conn.execute(
                    sa.text(
                        "DELETE FROM model_runs WHERE model_name = 'pg_integration_test_model';"
                    )
                )
        except Exception:
            pass
        session.close()


def test_postgres_connection_identity_and_isolation(pg_engine):
    """Verify connected database identity is strictly flurit_db with user flurit_app."""
    with pg_engine.connect() as conn:
        row = conn.execute(
            sa.text("SELECT current_database(), current_user, session_user;")
        ).fetchone()
        db_name, curr_user, sess_user = row
        assert db_name == "flurit_db"
        assert curr_user == "flurit_app"
        assert sess_user == "flurit_app"


def test_postgres_tables_exist(pg_engine):
    """Verify that Alembic migrations created all expected domain and ML analytics tables."""
    expected_tables = {
        "workspaces",
        "workspace_members",
        "processing_jobs",
        "accounts",
        "alembic_version",
        "churn_events",
        "customer_explanations",
        "customer_features",
        "customer_predictions",
        "customer_segments",
        "customer_snapshots",
        "feature_usage",
        "model_runs",
        "subscriptions",
        "support_tickets",
    }
    with pg_engine.connect() as conn:
        result = conn.execute(
            sa.text(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
            )
        ).fetchall()
        actual_tables = {r[0] for r in result}
        assert expected_tables.issubset(actual_tables), (
            f"Missing tables: {expected_tables - actual_tables}"
        )


def test_postgres_synthetic_persistence_and_relationships(pg_session):
    """Verify end-to-end synthetic persistence across accounts, snapshots, predictions, explanations, and segments."""
    repo = PredictionRepository()
    account_id = "ACC_SYNTH_PG_001"
    workspace_id = "ws_default"

    # 1. Create synthetic Account
    account = Account(
        workspace_id=workspace_id,
        account_id=account_id,
        account_name="Acme Synthetic Corp",
        industry="Technology",
        country="US",
        signup_date=date(2024, 1, 15),
        referral_source="direct",
        plan_tier="Enterprise",
        seats=25,
        is_trial=False,
        churn_flag=False,
    )
    pg_session.add(account)
    pg_session.flush()

    # 2. Create synthetic Subscription
    subscription = Subscription(
        workspace_id=workspace_id,
        subscription_id="SUB_SYNTH_PG_001",
        account_id=account_id,
        start_date=date(2024, 1, 15),
        plan_tier="Enterprise",
        seats=25,
        mrr_amount=4500.0,
        arr_amount=54000.0,
        is_trial=False,
        billing_frequency="annual",
        auto_renew_flag=True,
    )
    pg_session.add(subscription)
    pg_session.flush()

    # 3. Create synthetic Snapshot
    snapshot = CustomerSnapshot(
        workspace_id=workspace_id,
        account_id=account_id,
        snapshot_date=date(2024, 10, 1),
        is_active=True,
    )
    pg_session.add(snapshot)
    pg_session.flush()

    # 4. Create synthetic ModelRun
    repo = PredictionRepository()
    model_run = repo.get_or_create_model_run(
        session=pg_session,
        model_name="pg_integration_test_model",
        version="v1.0.0-pgtest",
        metadata={
            "decision_threshold": 0.50,
            "hyperparameters": {"penalty": "l2", "C": 1.0},
            "metrics": {"validation_pr_auc": 0.74, "roc_auc": 0.82},
            "feature_names": ["sub_current_mrr", "usage_count_30d"],
            "artifact_path": "artifacts/models/test_model.pkl",
        },
    )
    pg_session.flush()
    assert model_run.model_run_id is not None

    # 5. Build synthetic batch prediction output
    pred_output = BatchPredictionOutput(
        account_id=account_id,
        snapshot_date="2024-10-01",
        snapshot_id=snapshot.snapshot_id,
        model_name="pg_integration_test_model",
        model_version="v1.0.0-pgtest",
        model_score=0.725,
        predicted_risk_label=1,
        risk_tier="High Risk",
        cluster_id=2,
        cluster_name="Cluster 2: Stagnant Accounts",
        value_tier="High Value",
        top_risk_increasing_factors=[
            {"feature_name": "usage_trend_pct_30d", "attribution": 0.35},
            {"feature_name": "tickets_count_30d", "attribution": 0.22},
        ],
        top_risk_decreasing_factors=[{"feature_name": "sub_current_mrr", "attribution": -0.18}],
        shap_values={
            "sub_current_mrr": -0.18,
            "usage_trend_pct_30d": 0.35,
            "tickets_count_30d": 0.22,
        },
        plain_language_summary="High risk of churn driven by declining 30d usage trend.",
        feature_payload={"sub_current_mrr": 4500.0, "usage_trend_pct_30d": -15.0},
    )

    # 6. Persist batch predictions via repository
    saved_count = repo.save_prediction_batch(
        session=pg_session,
        model_run_id=model_run.model_run_id,
        predictions=[pred_output],
        workspace_id=workspace_id,
    )
    pg_session.commit()

    assert saved_count == 1

    # 7. Query back and verify persistence & relationship integrity
    persisted_pred = (
        pg_session.query(CustomerPrediction)
        .filter_by(
            workspace_id=workspace_id, account_id=account_id, model_run_id=model_run.model_run_id
        )
        .first()
    )
    assert persisted_pred is not None
    pred_id = persisted_pred.prediction_id
    assert persisted_pred.account_id == account_id
    assert persisted_pred.workspace_id == workspace_id
    assert pytest.approx(persisted_pred.churn_probability, 0.001) == 0.725
    assert persisted_pred.risk_level == "High Risk"

    # Verify Explanation 1-to-1 relationship
    explanation = (
        pg_session.query(CustomerExplanation)
        .filter_by(workspace_id=workspace_id, prediction_id=pred_id)
        .first()
    )
    assert explanation is not None
    assert explanation.account_id == account_id
    assert explanation.workspace_id == workspace_id
    assert len(explanation.top_risk_factors) == 2
    assert explanation.top_risk_factors[0]["feature_name"] == "usage_trend_pct_30d"

    # Verify Segment assignment
    segment = (
        pg_session.query(CustomerSegment)
        .filter_by(
            workspace_id=workspace_id, model_run_id=model_run.model_run_id, account_id=account_id
        )
        .first()
    )
    assert segment is not None
    assert segment.cluster_id == 2
    assert segment.cluster_name == "Cluster 2: Stagnant Accounts"
    assert segment.workspace_id == workspace_id


def test_postgres_foreign_key_constraint_enforcement(pg_session):
    """Verify PostgreSQL strictly enforces foreign-key constraints and rejects orphaned records."""
    non_existent_account = "ACC_SYNTH_PG_NONEXISTENT"

    pred = CustomerPrediction(
        workspace_id="ws_default",
        model_run_id=999999,  # Non-existent model run
        account_id=non_existent_account,  # Non-existent account
        churn_probability=0.88,
        risk_level="High Risk",
    )
    pg_session.add(pred)

    with pytest.raises(sa.exc.IntegrityError) as exc_info:
        pg_session.flush()

    assert "foreign key constraint" in str(exc_info.value).lower()
    pg_session.rollback()


def test_postgres_transaction_rollback_behavior(pg_session):
    """Verify atomic transaction rollback leaves the database clean upon mid-batch failure."""
    account_id = "ACC_SYNTH_PG_ROLLBACK"
    workspace_id = "ws_default"

    # Create account in initial clean transaction
    account = Account(
        workspace_id=workspace_id,
        account_id=account_id,
        account_name="Rollback Test Inc",
        industry="Retail",
        country="CA",
        signup_date=date(2024, 2, 1),
        referral_source="partner",
        plan_tier="Pro",
        seats=10,
        is_trial=False,
        churn_flag=False,
    )
    pg_session.add(account)
    pg_session.commit()

    # Now attempt a unit of work that inserts one valid subscription and one invalid record
    try:
        sub_valid = Subscription(
            workspace_id=workspace_id,
            subscription_id="SUB_SYNTH_PG_VALID",
            account_id=account_id,
            start_date=date(2024, 2, 1),
            plan_tier="Pro",
            seats=10,
            mrr_amount=1200.0,
            arr_amount=14400.0,
            is_trial=False,
            billing_frequency="monthly",
            auto_renew_flag=True,
        )
        pg_session.add(sub_valid)
        pg_session.flush()

        # Deliberately violate foreign key in the same transaction
        sub_invalid = Subscription(
            workspace_id=workspace_id,
            subscription_id="SUB_SYNTH_PG_INVALID",
            account_id="ACC_DOES_NOT_EXIST",
            start_date=date(2024, 2, 1),
            plan_tier="Pro",
            seats=10,
            mrr_amount=1200.0,
            arr_amount=14400.0,
            is_trial=False,
            billing_frequency="monthly",
            auto_renew_flag=True,
        )
        pg_session.add(sub_invalid)
        pg_session.flush()
        pg_session.commit()
    except sa.exc.IntegrityError:
        pg_session.rollback()

    # Verify sub_valid was rolled back and NOT persisted
    persisted_sub = pg_session.get(Subscription, (workspace_id, "SUB_SYNTH_PG_VALID"))
    assert persisted_sub is None, (
        "Transactional rollback failed: sub_valid was committed despite failure"
    )


def test_postgres_unique_constraint_enforcement(pg_session):
    """Verify PostgreSQL enforces unique constraint on (workspace_id, account_id, snapshot_date)."""
    account_id = "ACC_SYNTH_PG_UQ_TEST"
    workspace_id = "ws_default"
    account = Account(
        workspace_id=workspace_id,
        account_id=account_id,
        account_name="Unique Constraint Test",
        industry="Finance",
        country="US",
        signup_date=date(2024, 3, 1),
        referral_source="inbound",
        plan_tier="Enterprise",
        seats=50,
        is_trial=False,
        churn_flag=False,
    )
    pg_session.add(account)
    pg_session.flush()

    snap1 = CustomerSnapshot(
        workspace_id=workspace_id,
        account_id=account_id,
        snapshot_date=date(2024, 5, 1),
        is_active=True,
    )
    pg_session.add(snap1)
    pg_session.flush()

    # Attempt to insert identical account_id and snapshot_date
    snap2 = CustomerSnapshot(
        workspace_id=workspace_id,
        account_id=account_id,
        snapshot_date=date(2024, 5, 1),
        is_active=False,
    )
    pg_session.add(snap2)

    with pytest.raises(sa.exc.IntegrityError) as exc_info:
        pg_session.flush()

    assert (
        "uq_workspace_account_snapshot_date" in str(exc_info.value)
        or "unique constraint" in str(exc_info.value).lower()
    )
    pg_session.rollback()


def test_postgres_idempotent_batch_writes(pg_session):
    """Verify save_prediction_batch updates existing records cleanly without duplicate creation."""
    repo = PredictionRepository()
    account_id = "ACC_SYNTH_PG_IDEMPOTENT"
    workspace_id = "ws_default"

    account = Account(
        workspace_id=workspace_id,
        account_id=account_id,
        account_name="Idempotent Test Corp",
        industry="SaaS",
        country="US",
        signup_date=date(2024, 1, 1),
        referral_source="organic",
        plan_tier="Growth",
        seats=15,
        is_trial=False,
        churn_flag=False,
    )
    pg_session.add(account)
    pg_session.flush()

    model_run = repo.get_or_create_model_run(
        session=pg_session,
        model_name="pg_integration_test_model",
        version="v1.0.0-pgtest",
    )
    pg_session.flush()

    pred_v1 = BatchPredictionOutput(
        account_id=account_id,
        snapshot_date="2024-11-01",
        snapshot_id=None,
        model_name="pg_integration_test_model",
        model_version="v1.0.0-pgtest",
        model_score=0.45,
        predicted_risk_label=0,
        risk_tier="Low Risk",
        cluster_id=1,
        cluster_name="Cluster 1: Healthy",
        value_tier="Mid Value",
        top_risk_increasing_factors=[{"feature_name": "tickets_count_30d", "attribution": 0.1}],
        top_risk_decreasing_factors=[{"feature_name": "usage_count_30d", "attribution": -0.3}],
        shap_values={"tickets_count_30d": 0.1, "usage_count_30d": -0.3},
        plain_language_summary="Initial score",
        feature_payload={"usage_count_30d": 150},
    )

    # First write
    count1 = repo.save_prediction_batch(
        pg_session, model_run.model_run_id, [pred_v1], workspace_id=workspace_id
    )
    pg_session.commit()
    assert count1 == 1

    # Second write with updated score for same account and model run
    pred_v2 = BatchPredictionOutput(
        account_id=account_id,
        snapshot_date="2024-11-01",
        snapshot_id=None,
        model_name="pg_integration_test_model",
        model_version="v1.0.0-pgtest",
        model_score=0.88,
        predicted_risk_label=1,
        risk_tier="High Risk",
        cluster_id=2,
        cluster_name="Cluster 2: At-Risk",
        value_tier="Mid Value",
        top_risk_increasing_factors=[{"feature_name": "tickets_count_30d", "attribution": 0.6}],
        top_risk_decreasing_factors=[{"feature_name": "usage_count_30d", "attribution": -0.1}],
        shap_values={"tickets_count_30d": 0.6, "usage_count_30d": -0.1},
        plain_language_summary="Updated score",
        feature_payload={"usage_count_30d": 120},
    )

    count2 = repo.save_prediction_batch(
        pg_session, model_run.model_run_id, [pred_v2], workspace_id=workspace_id
    )
    pg_session.commit()
    assert count2 == 1

    # Verify exactly 1 prediction record exists with updated values
    preds = (
        pg_session.query(CustomerPrediction)
        .filter_by(
            workspace_id=workspace_id, account_id=account_id, model_run_id=model_run.model_run_id
        )
        .all()
    )
    assert len(preds) == 1, f"Expected 1 record, found {len(preds)}"
    assert pytest.approx(preds[0].churn_probability, 0.001) == 0.88
    assert preds[0].risk_level == "High Risk"


def test_postgres_isolation_guard_rejection():
    """Verify database isolation guard strictly blocks connection strings referencing Bynalt."""
    with pytest.raises(ValueError) as exc1:
        validate_database_url_isolation(
            "postgresql+psycopg2://flurit_app:pass@localhost:5432/bynalt"
        )
    assert "CRITICAL SECURITY VIOLATION" in str(exc1.value)

    with pytest.raises(ValueError) as exc2:
        validate_database_url_isolation(
            "postgresql+psycopg2://bynalt_user:pass@localhost:5432/flurit_db"
        )
    assert "CRITICAL SECURITY VIOLATION" in str(exc2.value)

    with pytest.raises(ValueError) as exc3:
        validate_database_url_isolation("postgresql://postgres:pass@localhost:5432/bynalt_test")
    assert "CRITICAL SECURITY VIOLATION" in str(exc3.value)
