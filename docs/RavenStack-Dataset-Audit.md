# RavenStack Dataset Audit & Feasibility Report

**Phase**: Phase 2 — Dataset Inspection & Relational Validation  
**Date**: October 13, 2026  
**Auditor**: Senior Software Architect & ML Platform Engineer  
**Dataset**: RavenStack SaaS Subscription & Churn Analytics (`data/raw/ravenstack/`)  

---

## 1. Executive Summary & File Inventory

The dataset located in `data/raw/ravenstack/` was directly inspected and audited using reproducible scripts (`scripts/inspect_dataset.py` and `scripts/validate_dataset.py`).

| Source File | Entity | Records | Columns | Primary Key | Key Status | Missing Values |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ravenstack_accounts.csv` | Accounts | 500 | 10 | `account_id` | **100% Unique** | 0 nulls across all cols |
| `ravenstack_subscriptions.csv` | Subscriptions | 5,000 | 14 | `subscription_id` | **100% Unique** | `end_date`: 4,514 (90.3% active) |
| `ravenstack_feature_usage.csv` | Feature Usage | 25,000 | 8 | `usage_id` | **21 Duplicate IDs** | 0 nulls across all cols |
| `ravenstack_support_tickets.csv` | Support Tickets | 2,000 | 9 | `ticket_id` | **100% Unique** | `satisfaction_score`: 825 (41.2% null) |
| `ravenstack_churn_events.csv` | Churn Events | 600 | 9 | `churn_event_id` | **100% Unique** | `feedback_text`: 148 (24.7% null) |

---

## 2. Referential Integrity & Foreign Key Analysis

Cross-table foreign key relationships were verified with 100% referential integrity across all five tables:

```
                  accounts (500 accounts)
                     │
       ┌─────────────┼─────────────┐
       │             │             │
       ▼ (0 orphans) ▼ (0 orphans) ▼ (0 orphans)
 subscriptions support_tickets churn_events
 (5,000 rows)   (2,000 rows)   (600 rows)
       │
       ▼ (0 orphans)
 feature_usage
 (25,000 rows)
```

* `subscriptions.account_id -> accounts.account_id`: **0 orphan records** (all 500 accounts referenced).
* `support_tickets.account_id -> accounts.account_id`: **0 orphan records** (492 distinct accounts referenced).
* `churn_events.account_id -> accounts.account_id`: **0 orphan records** (352 distinct accounts referenced).
* `feature_usage.subscription_id -> subscriptions.subscription_id`: **0 orphan records** (4,967 distinct subscriptions referenced).

---

## 3. Temporal Consistency & Chronological Ordering

All date/timestamp fields were evaluated for chronological causal order:

1. **Signup vs Subscription Start**: Zero violations. Every subscription starts on or after `accounts.signup_date`.
2. **Subscription Start vs End**: Zero violations. Every terminated subscription satisfies `end_date >= start_date`.
3. **Support Tickets**: Zero violations. All tickets satisfy `closed_at >= submitted_at`.
4. **Churn vs Signup**: Zero violations. Every churn event occurred on or after `accounts.signup_date`.
5. **Global Temporal Bounds**:
   - Earliest global event: `2023-01-01` (feature usage / accounts signup).
   - Latest global event: `2024-12-31`.
   - Total continuous temporal span: **730 days (~24.0 calendar months)**.

---

## 4. Churn Event Semantics & Target Construction Findings

Crucial operational nuances were uncovered regarding how churn is represented in RavenStack:

### A. Churn Event Multiplicity & Reactivations
* There are **600 churn event records** distributed across **352 distinct accounts**:
  - Exactly 1 churn event: 177 accounts
  - Exactly 2 churn events: 116 accounts
  - 3 or more churn events: 59 accounts (maximum: 5 churn events for a single account)
* **Reactivations**: 61 churn events (10.2%) have `is_reactivation = True`.
* **Implication**: Accounts in RavenStack can churn, subsequently resubscribe/reactivate, and potentially churn again. Churn prediction at snapshot date $T_{\text{obs}}$ must predict whether an **active** account at $T_{\text{obs}}$ will experience a churn event in $(T_{\text{obs}}, T_{\text{obs}} + 60\text{d}]$.

### B. Static `accounts.churn_flag` vs Event Log `churn_events.csv`
* `accounts.churn_flag == True` accounts for only 110 accounts, whereas 352 accounts appear in `churn_events.csv`.
* 277 accounts appear in `churn_events.csv` but have `accounts.churn_flag == False` because they reactivated or hold active subscriptions at the end of the dataset timeline.
* **CRITICAL ANTI-LEAKAGE RULE**: `accounts.churn_flag` represents a static final-state indicator. It **must never be used as a feature** (target leakage), nor should it be used as the prediction target. The prediction target must be constructed strictly from point-in-time forward event matching against `churn_events.csv`.

---

## 5. 60-Day Prediction Window Feasibility & Simulation

With a global dataset end date of `2024-12-31`, the right-censoring boundary for a 60-day prediction window is:
$$T_{\text{cutoff}} = 2024-12-31 - 60\text{ days} = 2024-11-01$$

Any customer snapshot evaluated on or before `2024-11-01` has a complete, fully observed 60-day future window.

### Monthly Snapshot Simulation
Simulating active accounts and forward 60-day churn events across monthly snapshot dates ($T_{\text{obs}}$):

| Snapshot Date | Cumulative Signed Up | Active at Snapshot | Churned in $(T, T+60\text{d}]$ | Empirical 60-Day Churn Rate |
| :--- | :--- | :--- | :--- | :--- |
| `2023-04-01` | 55 | 49 | 4 | **8.2%** |
| `2023-06-01` | 98 | 87 | 8 | **9.2%** |
| `2023-08-01` | 123 | 101 | 9 | **8.9%** |
| `2023-10-01` | 162 | 130 | 17 | **13.1%** |
| `2023-12-01` | 208 | 157 | 24 | **15.3%** |
| `2024-02-01` | 243 | 163 | 19 | **11.7%** |
| `2024-04-01` | 283 | 179 | 27 | **15.1%** |
| `2024-06-01` | 327 | 185 | 38 | **20.5%** |
| `2024-08-01` | 374 | 187 | 41 | **21.9%** |
| `2024-10-01` | 420 | 179 | 47 | **26.3%** |
| `2024-11-01` | 452 | 176 | 42 | **23.9%** |

### Feasibility Conclusion
1. **Window Duration**: The proposed **60-day prediction window is fully viable and supported**.
2. **Snapshot Volume**: Allows **19 to 20 monthly cohort snapshots** between `2023-04-01` and `2024-11-01`.
3. **Class Balance**: The empirical 60-day churn rate ranges between **$8\%$ and $26\%$** across observation dates, providing a healthy, non-degenerate class balance for supervised classification models.

---

## 6. Data Quality Issues & Recommendations for Phase 3 (PostgreSQL Schema)

| Table | Issue Observed | Impact | Phase 3 Recommendation |
| :--- | :--- | :--- | :--- |
| `feature_usage` | 21 duplicate `usage_id` values (24,979 unique out of 25,000 rows) | Using `usage_id` as a PostgreSQL `PRIMARY KEY UNIQUE` constraint will fail during ingestion. | In PostgreSQL schema, assign a surrogate primary key (`id BIGSERIAL PRIMARY KEY`), storing `usage_id` as an indexed non-unique column. |
| `support_tickets` | `satisfaction_score` is null in 41.2% of records (825 / 2,000) | Normal behavior (customers don't rate all tickets). | In feature engineering, compute both average satisfaction and a boolean/count indicator for unrated tickets. Do not impute with mean without an indicator. |
| `subscriptions` | `end_date` is null in 90.3% of records | Normal behavior for active/ongoing subscriptions. | In schema, define `end_date DATE NULL`. Treat `NULL` as currently active. |
| `churn_events` | `feedback_text` is null in 24.7% of records | Normal optional feedback. | Define column as nullable text in PostgreSQL. |

---

## 7. Observed Facts vs. Assumptions

| Item | Status | Details |
| :--- | :--- | :--- |
| RavenStack is synthetic data | **Observed Fact** | Explicitly stated in `README.md` and confirmed by structural patterns. |
| 5-table relational integrity | **Observed Fact** | 100% confirmed by script validation. Zero orphaned foreign keys. |
| 21 duplicate `usage_id`s | **Observed Fact** | Confirmed duplicate ID strings in `ravenstack_feature_usage.csv`. |
| 24-month time span | **Observed Fact** | All dates lie between `2023-01-01` and `2024-12-31`. |
| 60-day prediction window feasibility | **Observed Fact** | Verified: 19 monthly snapshot cohorts with $8\% - 26\%$ churn rates prior to $T_{\max} - 60\text{d}$. |
| Monthly snapshot frequency choice | **Architectural Recommendation** | Monthly snapshots ($T_{\text{obs}} = \text{1st of month}$) provide standard SaaS cohort boundaries without excessive row duplication. |
| Modeling unit at Account level | **Architectural Recommendation** | While accounts hold multiple subscriptions, Customer Success retention plays operate at the Account level. |
