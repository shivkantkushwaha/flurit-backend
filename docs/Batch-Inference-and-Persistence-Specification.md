# Flurit Batch Inference & Persistence Layer Specification

**Phase**: Phase 9 — Batch Inference & Persistence Layer  
**Date**: October 14, 2026  
**Auditor & Architect**: Senior Software Architect & ML Platform Engineer  
**Active Models**:
- Model: `artifacts/models/churn_prediction_v1/model.joblib` (v1.0.0, Logistic Regression)
- Explainer: `artifacts/explainability/explainer.joblib` (SHAP LinearExplainer)
- Segmentation: `artifacts/segmentation/segmentation_model.joblib` (K-Means, K=3)  
**Status**: Implemented, Verified, Documented, and Test-Covered (102/102 passing)

---

## 1. Executive Summary & Architecture Overview

Phase 9 implements Flurit's high-throughput batch inference pipeline and ACID-compliant SQLAlchemy persistence layer. This layer operationalizes the frozen ML artifacts to score, explain, segment, and persist customer predictions at scale while strictly isolating all database operations from external infrastructure.

### Core Architectural Invariants
1. **Deterministic Batch Scoring**: Accepts validated, point-in-time customer feature records and processes them in configurable chunks (default `batch_size=100`), ensuring deterministic score and explanation outputs.
2. **ACID Transaction Boundaries & Idempotency**: Database persistence operates within nested transactions (`session.begin_nested()`). Repeat runs for the same `(model_run_id, account_id, snapshot_id)` update existing predictions in place without creating duplicate rows or constraint violations. Any failure triggers a clean session rollback.
3. **Hardened Supabase JWT Security**: Following our pre-implementation audit, the JWT verification engine now validates expected signing algorithms (`SUPABASE_JWT_ALGORITHM`), audience (`SUPABASE_JWT_AUDIENCE`), optional issuer (`SUPABASE_JWT_ISSUER`), and not-before (`nbf`) claims. Fails closed with HTTP 500 if the secret is unconfigured.
4. **Strict Architectural Isolation**: Database configurations and repository queries reference only dedicated Flurit models (`app/models/`). Zero connections or credentials touch Bynalt infrastructure.
5. **No Synthetic Data / No Silent Skips**: When input records fail validation, they are explicitly collected in diagnostic error objects (`BatchRecordError`) rather than silently dropped or fabricated.

---

## 2. Supabase JWT Authentication Audit & Fixes

### Pre-Implementation Audit Findings
During the Phase 8 inspection, the following security and compatibility gaps were audited and addressed:
- **Algorithm Confusion Risk**: Verification was hardcoded to `HS256` without verifying that the token's header `alg` matched the server's expected configuration. If an algorithm mismatch or `none` algorithm was supplied, error handling was generic.
- **Audience & Issuer Omission**: Supabase GoTrue tokens default to `aud: "authenticated"` and optional issuer URLs (`https://<project-ref>.supabase.co/auth/v1`). Audience and issuer checks were not enforced in Phase 8.
- **Not-Before (`nbf`) Handling**: Future-dated tokens were not explicitly checked.

### Security Hardening Implemented
1. **Configuration-Driven Parameters** ([`app/core/config.py`](file:///c:/flurit-backend/app/core/config.py)):
   - `SUPABASE_JWT_ALGORITHM`: Configured expected algorithm (default `"HS256"`).
   - `SUPABASE_JWT_AUDIENCE`: Expected audience claim (default `"authenticated"`, nullable).
   - `SUPABASE_JWT_ISSUER`: Optional expected issuer claim (nullable).
2. **Hardened Claims Verification** ([`app/core/auth.py`](file:///c:/flurit-backend/app/core/auth.py)):
   - Verifies header `alg == expected_algorithm` (guards against algorithm confusion).
   - Verifies constant-time HMAC-SHA256 signature with `hmac.compare_digest`.
   - Validates `exp` (must be numeric and in the future).
   - Validates `nbf` (if present, must be in the past).
   - Validates `aud` (supports both string equality and list membership).
   - Validates `iss` (if configured, matches exact URL).
   - Validates `sub` (must be non-empty subject UUID).
   - **Fails Closed**: Unconfigured secret triggers HTTP 500 immediately.

---

## 3. Batch Inference Pipeline Architecture

Located in [`app/ml/inference/batch.py`](file:///c:/flurit-backend/app/ml/inference/batch.py).

### Data Contracts
- **`BatchInputRecord`**:
  ```python
  @dataclass(frozen=True)
  class BatchInputRecord:
      account_id: str
      snapshot_date: str | date
      features: CanonicalFeaturesPayload | dict[str, Any]
      snapshot_id: int | None = None
  ```
- **`BatchPredictionOutput`**: Contains `account_id`, `snapshot_date`, `snapshot_id`, `model_name`, `model_version`, `model_score`, `predicted_risk_label`, `risk_tier`, `cluster_id`, `cluster_name`, `value_tier`, `top_risk_increasing_factors`, `top_risk_decreasing_factors`, `shap_values`, `plain_language_summary`, and `feature_payload`.
- **`BatchPipelineSummary`**: Aggregate summary containing `total_records`, `successful_records`, `failed_records`, `batch_count`, `duration_seconds`, `predictions`, `errors`, and `risk_metric_interpretation`.
- **`BatchRecordError`**: Captures `index`, `account_id`, `error_type`, and `error_message`.

### Batch Processing Semantics
- **Chunking**: Records are processed in chunks of size `batch_size` (default: 100).
- **Validation**: Each record is strictly validated against `CanonicalFeaturesPayload` (all 45 features required, non-negative bounds checked, extra fields forbidden).
- **Error Modes**:
  - `fail_fast=False` (default): Invalid records are logged in `errors` list; valid records in the chunk are scored. Zero records are fabricated or silently ignored.
  - `fail_fast=True`: Raises `ValueError` on the first invalid record encountered.
- **Vectorized Scoring**: Model probability scoring and segmentation clustering are evaluated across the batch chunk DataFrame for computational efficiency. Local SHAP factor attributions are generated deterministically per customer.

---

## 4. Persistence Layer Architecture

Located in [`app/repositories/prediction_repository.py`](file:///c:/flurit-backend/app/repositories/prediction_repository.py) and [`app/services/prediction_service.py`](file:///c:/flurit-backend/app/services/prediction_service.py).

### Repository Contracts (`PredictionRepository`)
| Method | Description |
| :--- | :--- |
| `ensure_accounts_exist(session, account_ids)` | Verifies presence of referenced accounts; creates stub accounts if missing to maintain foreign key integrity |
| `get_or_create_model_run(session, name, version, meta)` | Idempotently retrieves or creates `model_runs` record with hyperparameters and metrics |
| `get_active_model_run(session)` | Queries the latest active model run (`is_active == True`) |
| `save_prediction_batch(session, model_run_id, predictions)` | Idempotently upserts predictions, linked explanations, and cluster segments within an atomic transaction savepoint |
| `get_latest_prediction_by_account(session, account_id)` | Retrieves most recent prediction for an account |
| `get_prediction_with_explanation(session, prediction_id)` | Retrieves prediction and associated SHAP explanation |
| `get_customer_segment(session, account_id, model_run_id)` | Retrieves segment assignment for an account |
| `get_predictions_by_model_run(session, model_run_id, ...)` | Paginated retrieval of predictions by model run |

### Transaction Boundaries & Idempotency
- **Atomic Savepoints**: `save_prediction_batch` executes within `session.begin_nested()`. If any database constraint or write fails, the savepoint rolls back cleanly without leaving partial records.
- **Idempotent Updates**: Existing records for `(model_run_id, account_id, snapshot_id)` are detected. Existing rows are updated in place with updated probabilities, factor attributions, and cluster assignments rather than inserting duplicate records.

---

## 5. Model Limitations & Disclaimers

1. **Uncalibrated Model Scores**:
   - Model scores represent uncalibrated relative risk rankings, not actuarial probabilities of customer churn.
   - As documented in Phase 6.1, the balanced class weighting shifts raw sigmoid probabilities upward, resulting in negative Brier skill scores ($\text{BSS} = -0.2554$).
2. **Observational vs. Causal Interpretation**:
   - SHAP factor attributions quantify statistical log-odds shifts learned from historical data. They do not constitute causal proof that operational interventions will alter churn outcomes.
3. **Prior Test-Set Inspection Disclosure**:
   - Recorded across all specifications: an unpurged test partition (`2024-09-01` to `2024-11-01`) was inspected in Phase 6 prior to implementing the Phase 6.1 target-window purging audit.

---

## 6. Separately Authorized Database Integration Procedure

When the technical lead authorizes connecting Flurit to live PostgreSQL / Supabase infrastructure, execute the following steps in order:

### Pre-Requisites Check
1. Confirm live PostgreSQL instance is provisioned and completely separate from Bynalt infrastructure.
2. Confirm valid credentials: `DATABASE_URL=postgresql+psycopg2://flurit_user:<password>@<host>:5432/flurit_db`.

### Step-by-Step Execution Sequence
1. **Environment Configuration**:
   - Update `.env` with live connection string:
     ```bash
     DATABASE_URL=postgresql+psycopg2://flurit_user:strong_password@db.flurit.internal:5432/flurit_db
     SUPABASE_JWT_SECRET=your-production-supabase-jwt-secret-here
     SUPABASE_JWT_AUDIENCE=authenticated
     ```
2. **Schema Migration**:
   - Execute Alembic migration to provision the 11 declared tables:
     ```bash
     alembic upgrade head
     ```
3. **Database Health Verification**:
   - Verify connection pool health and isolation check:
     ```bash
     python -c "from app.db.session import create_db_engine; engine = create_db_engine(); print('Engine alive:', engine.connect())"
     ```
4. **Initial Model Run Registration**:
   - Seed the initial active model run from saved metadata:
     ```bash
     python -c "from app.db.session import create_db_engine; from sqlalchemy.orm import Session; from app.repositories import PredictionRepository; from app.ml.inference.service import get_inference_service; engine = create_db_engine(); s = get_inference_service(); repo = PredictionRepository(); db = Session(bind=engine); repo.get_or_create_model_run(db, s._model_name, s._model_version, s.metadata); db.commit()"
     ```
5. **Scheduled Batch Scoring Trigger**:
   - Configure a recurring cron / Celery / Airflow job to execute `BatchPredictionService.run_and_persist_batch()` following monthly snapshot generation.

---

## 7. Verification & Test Suite Summary

- **Total Test Suite**: **102 tests passed** in 6.06s across all 14 test modules:
  - `tests/test_auth.py` (10 tests): JWT algorithm mismatch, audience validation, issuer validation, nbf claim, missing subject, fail-closed configuration.
  - `tests/test_batch_inference.py` (6 tests): Batch chunking, deterministic outputs, fail-fast vs error aggregation, batch sizing, empty collections.
  - `tests/test_persistence.py` (7 tests): ModelRun idempotence, batch prediction insertion, repeat run updates, transaction rollback on failure, query methods, end-to-end service.
  - Existing test suite (79 tests): App, config, db, explainability, features, inference, logging, models, segmentation, snapshots, training.
- **Code Quality**:
  - `ruff check .`: All checks passed.
  - `ruff format --check .`: 90 source files cleanly formatted.
  - `mypy app tests scripts alembic`: Success: 0 issues in 77 source files.
