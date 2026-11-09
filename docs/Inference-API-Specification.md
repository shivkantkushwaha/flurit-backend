# Flurit Inference Engine & Prediction Serving API Specification

**Phase**: Phase 8 — Inference Engine & Prediction Serving API  
**Date**: October 14, 2026  
**Auditor & Architect**: Senior Software Architect & ML Platform Engineer  
**Active Models**:
- Model: `artifacts/models/churn_prediction_v1/model.joblib` (v1.0.0, Logistic Regression)
- Explainer: `artifacts/explainability/explainer.joblib` (SHAP LinearExplainer)
- Segmentation: `artifacts/segmentation/segmentation_model.joblib` (K-Means, K=3)  
**Status**: Implemented, Verified, Documented, and Test-Covered (79/79 passing)

---

## 1. Executive Summary & Architecture Overview

Phase 8 implements Flurit's high-performance, read-only inference and prediction serving API on top of the frozen Phase 6.1 champion model, Phase 7 SHAP explainer, and Phase 7 $K=3$ customer segmentation artifacts.

### Key Architectural Invariants
1. **Zero Database Dependency**: All inference operations, feature validation, and Supabase JWT authentication operate entirely in memory without requiring a database connection or network roundtrip to PostgreSQL or Bynalt.
2. **Strict Canonical Validation**: Incoming payloads must provide all 45 canonical point-in-time features defined in `FEATURE_DEFINITIONS`. Extra fields (including forbidden retrospective labels like `target_churn_60d` and `churn_flag`) and missing fields are rejected with HTTP 422 to prevent target leakage and silent defaults.
3. **Artifact Pre-Warming & In-Memory Reusability**: Model pipelines, background maskers, and segmentation centroids are loaded once into a singleton service (`ModelInferenceService`) on application startup, eliminating per-request disk I/O and enabling sub-50ms response times.
4. **Configuration-Driven Fail-Closed Authentication**: Supabase JWT authentication verifies HMAC-SHA256 signatures using `SUPABASE_JWT_SECRET`. If the secret is missing or empty, the API fails closed with HTTP 500. Expired or forged tokens receive HTTP 401.
5. **Rigorous Risk Disclaimers**: Churn scores are explicitly labeled as uncalibrated risk rankings rather than actuarially calibrated probabilities. Estimated Revenue at Risk ($\text{MRR} \times \text{score}$) is presented as an operational prioritization metric, not guaranteed loss.

---

## 2. Authentication & Security Configuration

### Configuration-Driven Verification
Flurit secures all prediction endpoints using standard Supabase Bearer JWT tokens. Verification is implemented in `app/core/auth.py` using constant-time cryptographic primitives:

```text
Authorization: Bearer <supabase_jwt_token>
```

| Parameter | Source | Required | Description |
| :--- | :--- | :---: | :--- |
| `SUPABASE_JWT_SECRET` | Environment (`.env`) | **Yes** | HMAC-SHA256 secret key shared with the Supabase project |
| `SUPABASE_URL` | Environment (`.env`) | Optional | Base URL for the Supabase instance |

### Security Assertions & Failure Modes
- **Fail Closed**: If `SUPABASE_JWT_SECRET` is unset, null, or empty, all authenticated endpoints immediately abort with `HTTP 500 Internal Server Error: Authentication system is misconfigured: SUPABASE_JWT_SECRET is required`.
- **Missing Header**: Missing `Authorization` header triggers `HTTP 401 Unauthorized: Authentication required: Bearer token is missing`.
- **Expired Token**: Tokens where `time.time() > exp` trigger `HTTP 401 Unauthorized: Authentication token has expired`.
- **Invalid Signature**: Tampered or improperly signed tokens trigger `HTTP 401 Unauthorized: Invalid authentication token signature`.
- **Missing Subject**: Tokens lacking a non-empty `sub` claim trigger `HTTP 401 Unauthorized: Invalid token: missing or empty subject claim (sub)`.

---

## 3. Versioned Endpoint Contracts

All prediction routes are mounted under the `/api/v1` namespace.

### 3.1 POST `/api/v1/predictions/score`

Scores a single canonical customer feature vector and assigns a model risk tier.

#### Request Headers
```http
Authorization: Bearer <valid_jwt>
Content-Type: application/json
```

#### Request Body
```json
{
  "account_id": "A-00cac8",
  "snapshot_date": "2024-11-01",
  "features": {
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
    "churn_prior_refund_total": 0.0
  }
}
```

#### Response Body (`HTTP 200 OK`)
```json
{
  "account_id": "A-00cac8",
  "snapshot_date": "2024-11-01",
  "model_name": "flurit_churn_predictor",
  "model_version": "1.0.0",
  "model_score": 0.5628,
  "predicted_risk_label": 1,
  "decision_threshold": 0.5,
  "risk_tier": "High Risk",
  "score_interpretation": "Model Score Interpretation: Flurit model churn scores represent uncalibrated risk rankings learned by balanced Logistic Regression. Because Phase 6.1 demonstrated negative Brier skill scores (-0.2554) due to class_weight='balanced' probability shifts, the score is an operational prioritization index in [0.0, 1.0], NOT an actuarially calibrated probability of customer churn."
}
```

---

### 3.2 POST `/api/v1/customers/{account_id}/risk`

Computes the churn score, extracts local SHAP factor attributions (top 5 raising vs. top 5 lowering risk), and predicts the customer behavioral segment ($K=3$).

#### Request Path & Body
```http
POST /api/v1/customers/A-00cac8/risk
```
```json
{
  "snapshot_date": "2024-11-01",
  "features": { ... 45 canonical features ... }
}
```

#### Response Body (`HTTP 200 OK`)
```json
{
  "account_id": "A-00cac8",
  "snapshot_date": "2024-11-01",
  "model_version": "1.0.0",
  "model_score": 0.5628,
  "predicted_risk_label": 1,
  "decision_threshold": 0.5,
  "risk_tier": "High Risk",
  "segment_id": 1,
  "segment_name": "Cluster 1: High-Value Enterprise Accounts",
  "segment_summary": "High MRR ($9,622.05), high seat volume (125.3 seats), intensive telemetry activity (39.3 30d usage actions), mature tenure (164.2 days), healthcare/tech heavy.",
  "top_risk_increasing_factors": [
    {
      "feature_name": "sub_active_count",
      "feature_value": 7,
      "attribution": 0.6192,
      "direction": "increases_risk",
      "feature_description": "Count of subscriptions actively un-terminated at snapshot date."
    },
    {
      "feature_name": "usage_distinct_features_30d",
      "feature_value": 1,
      "attribution": 0.2437,
      "direction": "increases_risk",
      "feature_description": "Count of distinct product module action types executed in past 30 days."
    }
  ],
  "top_risk_decreasing_factors": [
    {
      "feature_name": "sub_tenure_days",
      "feature_value": 381.0,
      "attribution": -0.3919,
      "direction": "decreases_risk",
      "feature_description": "Commercial contract duration in days from first subscription start date to snapshot date."
    },
    {
      "feature_name": "usage_active_days_90d",
      "feature_value": 4,
      "attribution": -0.2826,
      "direction": "decreases_risk",
      "feature_description": "Distinct calendar days with recorded usage activity in past 90 days."
    }
  ],
  "plain_language_summary": "Estimated model churn score is 0.56. Main factors associated with higher score: sub active count (7), usage distinct features 30d (1). Mitigating factors associated with lower score: sub tenure days (381.0), usage active days 90d (4).",
  "association_disclaimer": "Statistical Association Note: Feature attributions reflect correlational relationships learned by the Logistic Regression model from historical observations. They quantify how much each feature shifts the predicted log-odds relative to the background baseline, and do NOT imply a direct causal relationship or guaranteed real-world outcome.",
  "score_interpretation": "Model Score Interpretation: Flurit model churn scores represent uncalibrated risk rankings learned by balanced Logistic Regression..."
}
```

---

### 3.3 POST `/api/v1/prioritization/quadrants`

Groups a portfolio of scored customer accounts into Value $\times$ Risk priority tiers and computes total portfolio Estimated Revenue at Risk ($\text{MRR} \times \text{score}$).

#### Request Body
```json
{
  "records": [
    { "account_id": "A1", "sub_current_mrr": 9500.0, "model_score": 0.65 },
    { "account_id": "A2", "sub_current_mrr": 5500.0, "model_score": 0.70 },
    { "account_id": "A3", "sub_current_mrr": 9200.0, "model_score": 0.40 },
    { "account_id": "A4", "sub_current_mrr": 2000.0, "model_score": 0.55 },
    { "account_id": "A5", "sub_current_mrr": 1500.0, "model_score": 0.20 }
  ],
  "mrr_high_threshold": 8000.0,
  "mrr_mid_threshold": 4000.0,
  "score_high_threshold": 0.50,
  "score_mid_threshold": 0.35
}
```

#### Response Body (`HTTP 200 OK`)
```json
{
  "total_accounts": 5,
  "total_mrr": 27700.0,
  "total_estimated_revenue_at_risk_mrr": 15105.0,
  "quadrants": [
    {
      "quadrant_name": "Priority 1: High Value - High Risk",
      "priority_level": 1,
      "account_count": 1,
      "account_percentage": 20.0,
      "total_mrr": 9500.0,
      "estimated_revenue_at_risk_mrr": 6175.0,
      "action_recommendation": "Executive outreach, dedicated customer success intervention, and immediate contract review."
    },
    {
      "quadrant_name": "Priority 2: Mid Value - High Risk",
      "priority_level": 2,
      "account_count": 1,
      "account_percentage": 20.0,
      "total_mrr": 5500.0,
      "estimated_revenue_at_risk_mrr": 3850.0,
      "action_recommendation": "Account manager consultation, technical health audit, and proactive feature adoption check."
    },
    {
      "quadrant_name": "Priority 3: High Value - Moderate Risk",
      "priority_level": 3,
      "account_count": 1,
      "account_percentage": 20.0,
      "total_mrr": 9200.0,
      "estimated_revenue_at_risk_mrr": 3680.0,
      "action_recommendation": "Executive sponsorship check-in, expansion exploration, and satisfaction benchmarking."
    },
    {
      "quadrant_name": "Priority 4: Core Value - High Risk",
      "priority_level": 4,
      "account_count": 1,
      "account_percentage": 20.0,
      "total_mrr": 2000.0,
      "estimated_revenue_at_risk_mrr": 1100.0,
      "action_recommendation": "Automated retention workflows, product education surveys, and self-service re-engagement."
    },
    {
      "quadrant_name": "Priority 5: Standard Monitoring",
      "priority_level": 5,
      "account_count": 1,
      "account_percentage": 20.0,
      "total_mrr": 1500.0,
      "estimated_revenue_at_risk_mrr": 300.0,
      "action_recommendation": "Ongoing lifecycle engagement, standard health check-ins, and roadmap updates."
    }
  ],
  "risk_metric_interpretation": "Estimated Revenue at Risk is calculated as MRR * Model Churn Score. Because Phase 6.1 demonstrated negative Brier skill scores for raw model probabilities, this metric serves strictly as an operational prioritization score, NOT an actuarially calibrated financial guarantee or literal probability of monetary loss."
}
```

---

## 4. Model & Calibration Limitations

1. **Uncalibrated Model Scores**:
   - The model pipeline utilizes `LogisticRegression(class_weight='balanced')`. The balanced class weighting forces the decision hyperplane to balance false positives and false negatives under class imbalance ($12\%$ prevalence in training).
   - As established in the Phase 6.1 audit, this shifts the raw sigmoid probabilities upward, yielding negative Brier skill scores ($\text{BSS} = -0.2554$).
   - **Operational Rule**: Scores must be treated as a relative risk ranking, not as calibrated probabilities.
2. **Prior Test-Set Inspection Disclosure**:
   - Disclosed in model metadata and all specifications: an unpurged test partition (`2024-09-01` to `2024-11-01`) was inspected during Phase 6 before the Phase 6.1 target-window purging audit was implemented.
3. **Observational vs. Causal Attribution**:
   - SHAP factor attributions quantify statistical log-odds shifts learned by the linear estimator. They do not prove that altering an operational parameter will causally change retention outcomes.

---

## 5. Verification & Test Suite Summary

The API and inference engine are verified with a 79-test automated test suite:
- **`tests/test_inference.py`** (16 tests):
  - Token signature verification, expiry handling, missing claims, and fail-closed secret configuration.
  - Endpoint scoring, schema enforcement, boundary value validation, and forbidden field rejection.
  - Customer risk attributions, plain-language summaries, and $K=3$ segment mapping.
  - Quadrant aggregations, sorting, and portfolio revenue-at-risk mathematics.
  - Missing artifact error handling and uninitialized service state guards.
- **Code Quality**:
  - `pytest`: 79/79 passed in 7.33s.
  - `ruff check .`: All checks passed.
  - `ruff format --check .`: 81 source files cleanly formatted.
  - `mypy app tests scripts alembic`: Success: 0 issues in 69 source files.
