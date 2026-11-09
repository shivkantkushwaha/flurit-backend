# Flurit Point-in-Time Feature Engineering Specification

**Phase**: Phase 5 — Feature Engineering Engine  
**Date**: October 14, 2026  
**Auditor & Architect**: Senior Software Architect & ML Platform Engineer  
**Dataset**: RavenStack SaaS Analytics (`data/raw/ravenstack/`)  
**Target Snapshot Cohort**: Approved `subscription_active` (4,284 records, 20 monthly dates)  
**Status**: Implemented, Verified, and Tested  

---

## 1. Executive Summary

Flurit's feature engineering pipeline transforms raw relational SaaS tables (`accounts`, `subscriptions`, `feature_usage`, `support_tickets`, `churn_events`) into time-aware, leakage-free feature vectors evaluated strictly at each customer snapshot date $T_{\text{obs}}$.

### Key Architecture Highlights
- **Exact Point-in-Time Boundaries**: Every feature is computed using information known on or before $T_{\text{obs}}$ ($t \le T_{\text{obs}}$). Any record occurring strictly after $T_{\text{obs}}$ ($t > T_{\text{obs}}$) is completely invisible to the pipeline.
- **Zero Target & Post-Hoc Leakage**: Prohibited static fields (`accounts.churn_flag`, `subscriptions.churn_flag`, `target_churn_60d`) and post-churn metadata (`feedback_text`, `reason_code`, future refunds) are strictly isolated from predictive features.
- **Complete Feature Matrix**: Produces **45 domain-specific features** across 5 categories with **0 missing values** (using structured indicator/sentinel pairs for sparse histories).
- **Future Invariance Guarantee**: Rigorously proven by unit tests: perturbing or appending future records to the database has **0.0% impact** on historical feature values.

---

## 2. Feature Catalog & Data Lineage

The pipeline extracts **45 predictive features** structured across 5 distinct domains:

### A. Account Profile Features (6 Features)
| Feature Name | Source Field | Lookback | Dtype | Description | Null Handling |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `account_age_days` | `accounts.signup_date` | None | `int64` | Days between customer signup and snapshot date $T_{\text{obs}}$. | $\ge 0$; 0 if signup on $T_{\text{obs}}$. |
| `account_industry` | `accounts.industry` | None | `category` | Industry vertical (e.g. DevTools, FinTech, E-commerce). | 'unknown' if missing. |
| `account_country` | `accounts.country` | None | `category` | Account headquarters country code (e.g. US, UK, DE). | 'unknown' if missing. |
| `account_referral_source` | `accounts.referral_source` | None | `category` | Acquisition channel (e.g. organic, partner, paid_ad). | 'unknown' if missing. |
| `account_initial_seats` | `accounts.seats` | None | `int64` | Seat count provisioned at account creation. | Defaults to 1 if missing. |
| `account_is_trial` | `accounts.is_trial` | None | `int64` | Binary indicator (1/0) whether account started on a free trial. | 0 if missing. |

---

### B. Subscription & Commercial Contract Features (10 Features)
Evaluated across subscriptions active at $T_{\text{obs}}$ ($\text{start\_date} \le T_{\text{obs}} \land (\text{end\_date IS NULL} \lor \text{end\_date} > T_{\text{obs}})$) and historical contracts started $\le T_{\text{obs}}$:

| Feature Name | Source Field | Lookback | Dtype | Description | Null Handling |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `sub_active_count` | `subscriptions` | None | `int64` | Count of active subscriptions held at snapshot date. | 0 if none. |
| `sub_current_mrr` | `subscriptions.mrr_amount` | None | `float64` | Total MRR ($) summed across active subscriptions at $T_{\text{obs}}$. | 0.0 if none. |
| `sub_current_arr` | `subscriptions.arr_amount` | None | `float64` | Total ARR ($) summed across active subscriptions at $T_{\text{obs}}$. | 0.0 if none. |
| `sub_total_active_seats` | `subscriptions.seats` | None | `int64` | Total licensed seats summed across active subscriptions at $T_{\text{obs}}$. | 0 if none. |
| `sub_tenure_days` | `subscriptions.start_date` | None | `int64` | Days from earliest subscription start date to $T_{\text{obs}}$. | 0 if none. |
| `sub_billing_frequency` | `subscriptions.billing_frequency` | None | `category` | Active billing frequency: `'monthly'`, `'annual'`, `'mixed'`, or `'none'`. | `'none'` if none. |
| `sub_has_auto_renew` | `subscriptions.auto_renew_flag` | None | `int64` | 1 if any active subscription has auto-renewal enabled; else 0. | 0 if none. |
| `sub_has_active_trial` | `subscriptions.is_trial` | None | `int64` | 1 if any active subscription is in trial status; else 0. | 0 if none. |
| `sub_historical_upgrade_count` | `subscriptions.upgrade_flag` | Lifetime | `int64` | Historical count of subscriptions with `upgrade_flag=True` started $\le T_{\text{obs}}$. | 0 if none. |
| `sub_historical_downgrade_count` | `subscriptions.downgrade_flag` | Lifetime | `int64` | Historical count of subscriptions with `downgrade_flag=True` started $\le T_{\text{obs}}$. | 0 if none. |

---

### C. Product Feature Usage Telemetry (14 Features)
Evaluated across telemetry records where `usage_date <= T_obs` mapped via `subscription_id` to `account_id`:

| Feature Name | Source Field | Lookback | Dtype | Description | Null Handling |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `usage_count_30d` | `feature_usage.usage_count` | 30 days | `int64` | Total product actions in $(T_{\text{obs}} - 30\text{d}, T_{\text{obs}}]$. | 0 if no usage. |
| `usage_count_60d` | `feature_usage.usage_count` | 60 days | `int64` | Total product actions in $(T_{\text{obs}} - 60\text{d}, T_{\text{obs}}]$. | 0 if no usage. |
| `usage_count_90d` | `feature_usage.usage_count` | 90 days | `int64` | Total product actions in $(T_{\text{obs}} - 90\text{d}, T_{\text{obs}}]$. | 0 if no usage. |
| `usage_duration_secs_30d` | `feature_usage.usage_duration_secs` | 30 days | `float64` | Active session duration in seconds in past 30 days. | 0.0 if no usage. |
| `usage_duration_secs_90d` | `feature_usage.usage_duration_secs` | 90 days | `float64` | Active session duration in seconds in past 90 days. | 0.0 if no usage. |
| `usage_error_count_30d` | `feature_usage.error_count` | 30 days | `int64` | Execution errors logged in past 30 days. | 0 if none. |
| `usage_error_rate_30d` | Calculated | 30 days | `float64` | Error ratio: $\text{errors\_30d} / \text{usage\_count\_30d}$. | 0.0 if usage is 0. |
| `usage_active_days_30d` | `feature_usage.usage_date` | 30 days | `int64` | Distinct calendar days with activity in past 30 days (0–30). | 0 if no usage. |
| `usage_active_days_90d` | `feature_usage.usage_date` | 90 days | `int64` | Distinct calendar days with activity in past 90 days (0–90). | 0 if no usage. |
| `usage_distinct_features_30d` | `feature_usage.feature_name` | 30 days | `int64` | Count of distinct feature modules utilized in past 30 days. | 0 if no usage. |
| `usage_beta_feature_count_30d` | `feature_usage.is_beta_feature` | 30 days | `int64` | Actions on beta features (`is_beta_feature=True`) in past 30 days. | 0 if no beta usage. |
| `usage_days_since_last` | `feature_usage.usage_date` | Lifetime | `float64` | Days from latest usage event $\le T_{\text{obs}}$ to $T_{\text{obs}}$. | -1.0 sentinel if never used. |
| `usage_has_history` | Calculated | Lifetime | `int64` | 1 if account has logged at least one usage event $\le T_{\text{obs}}$; else 0. | 0 if never used. |
| `usage_trend_pct_30d` | Calculated | 60 days | `float64` | Relative velocity change: $\frac{U_{30\text{d}} - U_{\text{prev30d}}}{\max(U_{\text{prev30d}}, 1.0)}$, clipped in $[-1.0, 5.0]$. | 0.0 if both windows 0. |

---

### D. Support Tickets & SLAs (11 Features)
Evaluated across tickets submitted on or before $T_{\text{obs}}$ (`submitted_at <= T_obs`):

| Feature Name | Source Field | Lookback | Dtype | Description | Null Handling |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `tickets_count_30d` | `support_tickets` | 30 days | `int64` | Tickets submitted in past 30 days $(T_{\text{obs}} - 30\text{d}, T_{\text{obs}}]$. | 0 if none. |
| `tickets_count_90d` | `support_tickets` | 90 days | `int64` | Tickets submitted in past 90 days $(T_{\text{obs}} - 90\text{d}, T_{\text{obs}}]$. | 0 if none. |
| `tickets_count_lifetime` | `support_tickets` | Lifetime | `int64` | Total tickets submitted on or before $T_{\text{obs}}$. | 0 if none. |
| `tickets_unresolved_count` | `closed_at` | Lifetime | `int64` | Tickets open at $T_{\text{obs}}$ (`closed_at IS NULL` or `closed_at > T_obs`). | 0 if all closed. |
| `tickets_has_unresolved` | Calculated | Lifetime | `int64` | 1 if `tickets_unresolved_count > 0`; else 0. | 0 if none open. |
| `tickets_urgent_count_90d` | `support_tickets.priority` | 90 days | `int64` | Tickets submitted in 90d with `priority='urgent'`. | 0 if none. |
| `tickets_high_or_urgent_count_90d`| `support_tickets.priority` | 90 days | `int64` | Tickets submitted in 90d with `priority in ('high', 'urgent')`. | 0 if none. |
| `tickets_escalated_count_90d` | `support_tickets.escalation_flag` | 90 days | `int64` | Tickets submitted in 90d with `escalation_flag=True`. | 0 if none. |
| `tickets_avg_satisfaction_90d` | `support_tickets.satisfaction_score`| 90 days | `float64` | Average CSAT rating for rated tickets submitted in 90d (1.0 to 5.0). | -1.0 sentinel if unrated. |
| `tickets_has_unrated_90d` | Calculated | 90 days | `int64` | 1 if customer had unrated tickets in 90d; else 0. | 0 if none unrated. |
| `tickets_avg_first_response_minutes_90d` | `first_response_time_minutes` | 90 days | `float64` | Average first response time in minutes in past 90 days. | -1.0 sentinel if no tickets. |

---

### E. Churn History (4 Features)
Evaluated strictly using churn events logged on or before snapshot date (`churn_date <= T_obs`):

| Feature Name | Source Field | Lookback | Dtype | Description | Null Handling |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `churn_prior_count` | `churn_events` | Lifetime | `int64` | Historical churn event count logged $\le T_{\text{obs}}$. | 0 if customer never churned. |
| `churn_has_prior` | Calculated | Lifetime | `int64` | 1 if customer experienced $\ge 1$ prior churn event $\le T_{\text{obs}}$; else 0. | 0 if no prior churn. |
| `churn_days_since_last` | `churn_events.churn_date` | Lifetime | `float64` | Days from latest prior churn event to $T_{\text{obs}}$. | -1.0 sentinel if no prior churn. |
| `churn_prior_refund_total` | `churn_events.refund_amount_usd`| Lifetime | `float64` | Total refund dollars granted across prior churn events $\le T_{\text{obs}}$. | 0.0 if no prior refunds. |

---

## 3. Strict Anti-Leakage Invariants

1. **No Static Retrospective Columns**:
   - `accounts.churn_flag` and `subscriptions.churn_flag` are strictly barred from the pipeline and asserted absent by `FeaturePipeline.verify_no_leakage()`.
2. **Strict Lower & Upper Boundary Isolation**:
   - For an observation at date $T_{\text{obs}}$, all aggregations query strictly $[T_{\text{obs}} - W, T_{\text{obs}}]$.
   - Telemetry or tickets occurring on $T_{\text{obs}} + 1\text{d}$ are excluded.
3. **Future Perturbation Invariance Proof**:
   - Verified by unit test `test_future_records_do_not_alter_historical_features`: modifying, inserting, or deleting future records in raw tables yields **identical feature vectors** at $T_{\text{obs}}$.
4. **Prediction Target Isolation**:
   - `target_churn_60d` is evaluated over $(T_{\text{obs}}, T_{\text{obs}} + 60\text{d}]$ and is strictly isolated from predictive features.

---

## 4. Reproducibility & CLI Execution

To run feature engineering across the full dataset:

```powershell
.\.venv\Scripts\python.exe scripts\build_features.py --data-dir data/raw/ravenstack
```

### Verified Census Output
- **Total Customer Snapshots**: **4,284**
- **Feature Vector Dimension**: **45 features**
- **Composite Key Uniqueness**: `(account_id, snapshot_date)` has **0 duplicate rows**.
- **Missing Value Count**: **0 nulls across all 45 feature columns** (100% complete).
- **Target Distribution**:
  - `target_churn_60d = 1`: **680 (15.87%)**
  - `target_churn_60d = 0`: **3,604 (84.13%)**
