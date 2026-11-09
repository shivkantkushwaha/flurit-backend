# Flurit PostgreSQL Data Model & Schema Architecture

**Phase**: Phase 3 — PostgreSQL Schema & Data Model  
**Date**: October 13, 2026  
**Auditor & Architect**: Senior Software Architect & ML Platform Engineer  
**Database Isolation**: 100% Dedicated Flurit Instance (Zero Bynalt Access)  

---

## 1. Resolution of Snapshot Dates & Eligibility Rules

### A. Resolution of the 19 vs. 11 Snapshot Discrepancy
In the Phase 2 dataset inspection report, the simulation table displayed **11 representative snapshot dates** (alternating monthly cohorts) for presentation brevity, while reporting that 19–20 monthly cohorts were available.

Here is the **complete chronological census of all 20 monthly snapshot dates** spanning the dataset's observation window:

| Index | Snapshot Date ($T_{\text{obs}}$) | Cumulative Signed Up | Active Accounts at $T$ | Churned in $(T, T+60\text{d}]$ | Empirical 60-Day Churn Rate |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `2023-04-01` | 55 | 49 | 4 | **8.2%** |
| 2 | `2023-05-01` | 71 | 62 | 7 | **11.3%** |
| 3 | `2023-06-01` | 98 | 87 | 8 | **9.2%** |
| 4 | `2023-07-01` | 110 | 94 | 10 | **10.6%** |
| 5 | `2023-08-01` | 123 | 101 | 9 | **8.9%** |
| 6 | `2023-09-01` | 139 | 111 | 10 | **9.0%** |
| 7 | `2023-10-01` | 162 | 130 | 17 | **13.1%** |
| 8 | `2023-11-01` | 182 | 140 | 16 | **11.4%** |
| 9 | `2023-12-01` | 208 | 157 | 24 | **15.3%** |
| 10 | `2024-01-01` | 228 | 163 | 19 | **11.7%** |
| 11 | `2024-02-01` | 243 | 163 | 19 | **11.7%** |
| 12 | `2024-03-01` | 258 | 172 | 25 | **14.5%** |
| 13 | `2024-04-01` | 283 | 179 | 27 | **15.1%** |
| 14 | `2024-05-01` | 305 | 186 | 32 | **17.2%** |
| 15 | `2024-06-01` | 327 | 185 | 38 | **20.5%** |
| 16 | `2024-07-01` | 348 | 188 | 46 | **24.5%** |
| 17 | `2024-08-01` | 374 | 187 | 41 | **21.9%** |
| 18 | `2024-09-01` | 396 | 181 | 44 | **24.3%** |
| 19 | `2024-10-01` | 420 | 179 | 47 | **26.3%** |
| 20 | `2024-11-01` | 452 | 176 | 42 | **23.9%** |

### B. The 60-Day Right-Censoring Cutoff Calculation
* **Global Max Event Timestamp**: `2024-12-31`.
* **Prediction Window ($W$)**: $60\text{ calendar days}$.
* **Right-Censoring Cutoff Date ($T_{\text{cutoff}}$)**:
  $$T_{\text{cutoff}} = 2024\text{-}12\text{-}31 - 60\text{ days} = 2024\text{-}11\text{-}01$$
* **Right-Censored Samples**: Any snapshot after `2024-11-01` (e.g., `2024-12-01`) has less than 60 days of observable future before the dataset terminates. In accordance with the Anti-Leakage Rule (*"Don't create negative labels where the future is unknown"*), snapshots after $T_{\text{cutoff}}$ must have `target_churn_60d = NULL` and must be excluded from training sets.

### C. Exact Snapshot Eligibility Rules
An account $A$ is eligible for inclusion in snapshot cohort $T_{\text{obs}}$ if and only if:
1. **Signup Requirement**: $A.\text{signup\_date} \le T_{\text{obs}}$ (the customer existed on or before the observation date).
2. **Active Status Requirement**: At date $T_{\text{obs}}$, account $A$ is not in a churned state without subsequent reactivation. That is:
   $$\max(A.\text{churn\_events}.\text{churn\_date} \le T_{\text{obs}}) < \max(A.\text{subscriptions}.\text{start\_date} \le T_{\text{obs}})$$
3. **Point-in-Time Feature Isolation**: All feature inputs for $(A, T_{\text{obs}})$ must use records with timestamp $\le T_{\text{obs}}$.
4. **Target Label Assignment**:
   - `target_churn_60d = 1` if $A$ has an event in `churn_events` with $\text{churn\_date} \in (T_{\text{obs}}, T_{\text{obs}} + 60\text{d}]$.
   - `target_churn_60d = 0` if $A$ remains active through $T_{\text{obs}} + 60\text{d}$ and $T_{\text{obs}} \le T_{\text{cutoff}}$.
   - `target_churn_60d = NULL` if $T_{\text{obs}} > T_{\text{cutoff}}$ (right-censored holdout).

---

## 2. Account-Level Churn Semantics & Distinctions

In the RavenStack simulated dataset, churn cannot be reduced to a single naive assumption. We formally distinguish four distinct concepts:

### A. Subscription Cancellation vs. Account Churn
* **Subscription Cancellation**: Recorded in `subscriptions` where `churn_flag = True` and `end_date` is populated. Accounts can have multiple concurrent or sequential subscriptions. A single subscription terminating does not necessarily mean the customer has departed the business.
* **Account-Level Churn Event**: Recorded in `churn_events` with a distinct `churn_date`, `reason_code`, and `refund_amount_usd`. This represents an explicit exit transaction.

### B. Why `MRR → 0` is Not Assumed
In synthetic datasets like RavenStack, the generator created edge cases:
- Account `A-526d93` experienced churn events on `2024-07-10`, `2024-08-05`, and `2024-09-25`, yet held active subscription records with `end_date = NULL` through late 2024.
- Because subscription records were simulated independently of churn logs, **`MRR → 0` is an operational SaaS objective, not an invariant proven by raw synthetic records.**
- **Resolution**: The predictive target is defined explicitly by the arrival of a recorded `churn_events` record within $(T_{\text{obs}}, T_{\text{obs}} + 60\text{d}]$, rather than checking whether cumulative active subscription MRR sums to zero.

### C. Reactivation & Repeated Churn Events
- **Repeated Churn**: 175 accounts have $\ge 2$ churn events (up to 5 churn events for a single account).
- **Reactivations**: 61 churn events (10.2%) are explicitly flagged with `is_reactivation = True`.
- When an account churns, they can return with a new subscription start date. If they are active at snapshot date $T_{\text{obs}}$, they re-enter the candidate pool to predict their next churn risk.

---

## 3. Database Schema Design (11 Tables)

```
================================================================================
LAYER 1: RAW SOURCE TABLES (Mirrors 5 RavenStack CSV Entities)
================================================================================
1. accounts
   - account_id (VARCHAR(32), PK)
   - account_name, industry, country, signup_date (INDEX), referral_source,
     plan_tier, seats, is_trial, churn_flag (retrospective)

2. subscriptions
   - subscription_id (VARCHAR(32), PK)
   - account_id (VARCHAR(32), FK -> accounts.account_id ON DELETE CASCADE, INDEX)
   - start_date (INDEX), end_date (INDEX), plan_tier, seats, mrr_amount, arr_amount,
     is_trial, upgrade_flag, downgrade_flag, churn_flag, billing_frequency, auto_renew_flag

3. feature_usage (SURROGATE PK TO PRESERVE 21 DUPLICATE SOURCE USAGE_IDS)
   - id (BIGINT, PK, AUTOINCREMENT)  <-- Surrogate PK prevents primary key violation
   - usage_id (VARCHAR(32), INDEX)   <-- Preserves natural source ID without data loss
   - subscription_id (VARCHAR(32), FK -> subscriptions.subscription_id ON DELETE CASCADE, INDEX)
   - usage_date (INDEX), feature_name (INDEX), usage_count, usage_duration_secs, error_count, is_beta_feature

4. support_tickets
   - ticket_id (VARCHAR(32), PK)
   - account_id (VARCHAR(32), FK -> accounts.account_id ON DELETE CASCADE, INDEX)
   - submitted_at (INDEX), closed_at, resolution_time_hours, priority (INDEX),
     first_response_time_minutes, satisfaction_score (NULLABLE), escalation_flag

5. churn_events
   - churn_event_id (VARCHAR(32), PK)
   - account_id (VARCHAR(32), FK -> accounts.account_id ON DELETE CASCADE, INDEX)
   - churn_date (INDEX), reason_code (INDEX), refund_amount_usd, preceding_upgrade_flag,
     preceding_downgrade_flag, is_reactivation, feedback_text

================================================================================
LAYER 2: DERIVED ML & ANALYTICS TABLES (Time-Aware Monolith Store)
================================================================================
6. customer_snapshots
   - snapshot_id (BIGINT, PK, AUTOINCREMENT)
   - account_id (VARCHAR(32), FK -> accounts.account_id, INDEX)
   - snapshot_date (DATE, INDEX)
   - is_active (BOOLEAN), target_churn_60d (INT NULLABLE)
   - created_at (TIMESTAMP WITH TIME ZONE)
   - UNIQUE CONSTRAINT (account_id, snapshot_date)

7. customer_features
   - feature_id (BIGINT, PK, AUTOINCREMENT)
   - snapshot_id (BIGINT, UNIQUE, FK -> customer_snapshots.snapshot_id, INDEX)
   - account_id (VARCHAR(32), FK -> accounts.account_id, INDEX)
   - snapshot_date (DATE, INDEX)
   - current_mrr, current_arr, tenure_days, usage_count_30d, usage_duration_30d,
     usage_change_pct, ticket_count_30d, avg_satisfaction, has_unrated_tickets,
     feature_payload (JSON), computed_at

8. model_runs
   - model_run_id (BIGINT, PK, AUTOINCREMENT)
   - model_name, version, is_active, decision_threshold, hyperparameters (JSON),
     metrics (JSON: ROC-AUC, PR-AUC, F1, Brier), feature_names (JSON),
     artifact_path, trained_at, status

9. customer_predictions
   - prediction_id (BIGINT, PK, AUTOINCREMENT)
   - model_run_id (BIGINT, FK -> model_runs.model_run_id, INDEX)
   - account_id (VARCHAR(32), FK -> accounts.account_id, INDEX)
   - snapshot_id (BIGINT, FK -> customer_snapshots.snapshot_id, INDEX)
   - churn_probability (FLOAT), risk_level (VARCHAR(16)), predicted_at

10. customer_explanations
    - explanation_id (BIGINT, PK, AUTOINCREMENT)
    - prediction_id (BIGINT, UNIQUE, FK -> customer_predictions.prediction_id, INDEX)
    - account_id (VARCHAR(32), FK -> accounts.account_id, INDEX)
    - top_risk_factors (JSON), shap_values (JSON), summary_text, computed_at

11. customer_segments
    - segment_id (BIGINT, PK, AUTOINCREMENT)
    - model_run_id (BIGINT NULLABLE, FK -> model_runs.model_run_id, INDEX)
    - account_id (VARCHAR(32), FK -> accounts.account_id, INDEX)
    - cluster_id (INT), cluster_name (VARCHAR(100)), risk_tier, value_tier, created_at
```

---

## 4. Handling `feature_usage.usage_id` Duplicates Without Data Loss

In `ravenstack_feature_usage.csv`, exactly 21 `usage_id` values appear twice (e.g. `U-25b56c`, `U-662254`).
If `usage_id` were declared as the PostgreSQL `PRIMARY KEY`, loading the source dataset would abort with duplicate key violations or require deduplication that silently discards usage events.

**Design Resolution**:
- Added an auto-incrementing surrogate primary key: `id BIGSERIAL PRIMARY KEY`.
- Preserved `usage_id VARCHAR(32) NOT NULL` as an indexed column.
- Result: **All 25,000 source records are preserved with 100% fidelity and zero data loss.**

---

## 5. Migration Instructions & Isolation Safeguards

### Migration Version: `0001_initial_schema`
Location: `alembic/versions/0001_initial_schema.py`

### Reviewing Migrations Offline (No Database Connection Required)
To inspect the generated SQL DDL statements offline:
```powershell
.\.venv\Scripts\alembic.exe upgrade 0001_initial_schema --sql
```

### Applying Migrations to Dedicated Flurit Database
When local PostgreSQL is ready (and only with Flurit's isolated credentials):
```powershell
.\.venv\Scripts\alembic.exe upgrade head
```

### Bynalt Isolation Enforcement
- `app/db/session.py` and `alembic/env.py` execute `validate_database_url_isolation()`.
- If the substring `bynalt` appears in `DATABASE_URL`, execution is immediately halted with a security violation exception before any connection socket is opened.
