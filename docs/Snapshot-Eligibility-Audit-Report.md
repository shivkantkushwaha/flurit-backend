# Flurit Phase 4.1: Snapshot Eligibility & Label Audit Report

**Phase**: Phase 4.1 — Snapshot Eligibility & Label Audit  
**Date**: October 13, 2026  
**Auditor**: Senior Software Architect & ML Platform Engineer  
**Dataset**: RavenStack SaaS Dataset (`data/raw/ravenstack/`)  
**Status**: Completed & Validated Against Raw Data  

---

## 1. Executive Summary

This audit reconciles the point-in-time account snapshot cohorts and 60-day churn target labels constructed in Phase 4 with the preliminary simulation conducted in Phase 2.

### Key Audit Findings
1. **Source of Discrepancy (4,284 vs 2,890 snapshots)**:
   - Phase 2 simulated an active cohort using `signup_date <= T_obs` and **permanently excluded any account with prior churn** (`account_id not in prior_churns`). This corresponds exactly to the `churn_free` policy (yielding **2,890 snapshots** and 485 churn events across the 20 monthly dates).
   - Phase 4's default implementation used the `subscription_active` policy, which qualifies any account with an un-terminated subscription at $T_{\text{obs}}$ (`start_date <= T_obs` and `end_date IS NULL` or `end_date > T_obs`), yielding **4,284 snapshots** and 680 churn events.
2. **RavenStack Synthetic Data Characteristics**:
   - **Open-Ended Subscriptions**: 4,514 out of 5,000 subscriptions (**90.3%**) have `end_date IS NULL`. Even for accounts with `accounts.churn_flag == True`, 89.8% of their subscriptions have `end_date IS NULL`.
   - **Subscription Decoupling**: In 545 out of 600 churn events (**90.8%**), the customer held active subscriptions on their `churn_date`, and only **6 churn events** coincided with a subscription `end_date`. The target denotes a customer-recorded churn event log entry, **not** guaranteed subscription cancellation.
   - **Signup vs Subscription Lag**: 475 out of 500 accounts (**95.0%**) have their earliest subscription start date *after* their signup date (median lag: 18.0 days, max lag: 432 days). 53 churn events occur *before* the customer's first subscription start date.
3. **Churn Event Data Integrity & Count Reconciliation (600 vs. 598 Narrative Typo)**:
   - Raw source file `ravenstack_churn_events.csv` contains exactly **600 records** (601 lines with header).
   - All 600 records have unique `churn_event_id`s, 0 nulls across mandatory fields, and 100% foreign-key referential integrity to `accounts`.
   - Exactly **599 unique `(account_id, churn_date)` tuples** exist: Account `A-cc102e` contains two distinct churn events on `2024-11-19` (`C-e20cc6` and `C-ce47da`). Both records are valid and contribute to the target window.
   - Exactly **0 records are dropped or excluded**. The reference to "598" in previous narrative was an erroneous transcription typo.
4. **Window Boundary & Policy Approval**:
   - Churn events occurring exactly on $T_{\text{obs}}$ (**14 events**) are strictly excluded from the forward target (anti-leakage guarantee).
   - Churn events occurring exactly on $T_{\text{obs}} + 60\text{d}$ (**18 events**) are legitimately included.
   - 123 snapshot instances feature $\ge 2$ churn events within the 60-day window; all evaluate to binary target `1`.
   - **`subscription_active` is confirmed and approved as the primary policy** for CLI generation and Phase 5 feature engineering. Zero labels have been changed.

---

## 2. Reconciling Phase 2 Simulation vs Phase 4 Engine

### Phase 2 Calculation (`scripts/validate_dataset.py`, lines 196–216)
```python
cohort = accounts[accounts["parsed_signup"] <= snap].copy()
prior_churns = set(churn_events[churn_events["parsed_churn"] <= snap]["account_id"])
active_cohort = cohort[~cohort["account_id"].isin(prior_churns)]
```
- **Assumed Logic**: Phase 2 treated any signed-up customer who had not yet recorded a churn event as "active".
- **Limitations**:
  1. It completely bypassed the `subscriptions` table. On `2023-04-01`, it counted 49 accounts as active, but **34 of those accounts had no subscription started yet**!
  2. It permanently removed accounts after one churn event, causing cohort sizes to stagnate around ~176–187 accounts by late 2024, despite company growth.
  3. Across the 20 monthly dates (`2023-04-01` to `2024-11-01`), this formula produces exactly **2,890 total snapshots with 485 positive targets (16.78%)**. In the Phase 2 report, an 11-row bimonthly table was displayed for brevity.

### Phase 4 Calculation (`app/ml/snapshots/engine.py`, lines 108–120)
```python
active_subs = self.subscriptions[
    (self.subscriptions["parsed_start_date"] <= snap_dt)
    & (
        self.subscriptions["parsed_end_date"].isnull()
        | (self.subscriptions["parsed_end_date"] > snap_dt)
    )
]
eligible = sorted(signed_up & set(active_subs["account_id"].unique()))
```
- **Realized Logic**: Evaluates active commercial contracts at $T_{\text{obs}}$.
- **Behavior**:
  - Excludes pre-subscription accounts (early cohorts are smaller: 21 on `2023-04-01` vs 49 in Phase 2).
  - Retains multi-subscription and repeat accounts whose contracts remain active (late cohorts are larger: 441 on `2024-11-01` vs 176 in Phase 2).
  - Produces **4,284 total snapshots with 680 positive targets (15.87%)**.

---

## 3. Policy Census & Exclusion Breakdown

Across all 20 monthly dates ($T_{\text{obs}} \in [2023\text{-}04\text{-}01, 2024\text{-}11\text{-}01]$):

### A. `subscription_active` Policy Census & Exclusions (Total Accounts = 500)
| Snapshot Date | Eligible Cohort | Target = 1 | Target = 0 | Churn Rate | Excl: Not Signed Up | Excl: No Sub Started | Excl: All Subs Ended |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `2023-04-01` | 21 | 3 | 18 | 14.3% | 445 | 34 | 0 |
| `2023-05-01` | 37 | 5 | 32 | 13.5% | 429 | 34 | 0 |
| `2023-06-01` | 50 | 6 | 44 | 12.0% | 402 | 48 | 0 |
| `2023-07-01` | 71 | 5 | 66 | 7.0% | 390 | 39 | 0 |
| `2023-08-01` | 90 | 9 | 81 | 10.0% | 377 | 33 | 0 |
| `2023-09-01` | 117 | 12 | 105 | 10.3% | 361 | 22 | 0 |
| `2023-10-01` | 131 | 14 | 117 | 10.7% | 338 | 31 | 0 |
| `2023-11-01` | 149 | 15 | 134 | 10.1% | 318 | 33 | 0 |
| `2023-12-01` | 172 | 24 | 148 | 14.0% | 292 | 35 | 1 |
| `2024-01-01` | 191 | 19 | 172 | 9.9% | 272 | 36 | 1 |
| `2024-02-01` | 216 | 24 | 192 | 11.1% | 257 | 26 | 1 |
| `2024-03-01` | 235 | 33 | 202 | 14.0% | 242 | 23 | 0 |
| `2024-04-01` | 264 | 30 | 234 | 11.4% | 217 | 19 | 0 |
| `2024-05-01` | 282 | 44 | 238 | 15.6% | 195 | 23 | 0 |
| `2024-06-01` | 313 | 52 | 261 | 16.6% | 173 | 14 | 0 |
| `2024-07-01` | 338 | 59 | 279 | 17.5% | 152 | 10 | 0 |
| `2024-08-01` | 366 | 66 | 300 | 18.0% | 126 | 8 | 0 |
| `2024-09-01` | 385 | 76 | 309 | 19.7% | 104 | 11 | 0 |
| `2024-10-01` | 415 | 85 | 330 | 20.5% | 80 | 4 | 1 |
| `2024-11-01` | 441 | 99 | 342 | 22.4% | 48 | 11 | 0 |
| **TOTAL** | **4,284** | **680** | **3,604** | **15.87%** | — | — | — |

---

### B. `churn_free` Policy Census & Exclusions (Total Accounts = 500)
| Snapshot Date | Eligible Cohort | Target = 1 | Target = 0 | Churn Rate | Excl: Not Signed Up | Excl: Prior Churn Occurred |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `2023-04-01` | 49 | 4 | 45 | 8.2% | 445 | 6 |
| `2023-05-01` | 62 | 7 | 55 | 11.3% | 429 | 9 |
| `2023-06-01` | 87 | 8 | 79 | 9.2% | 402 | 11 |
| `2023-07-01` | 94 | 10 | 84 | 10.6% | 390 | 16 |
| `2023-08-01` | 101 | 9 | 92 | 8.9% | 377 | 22 |
| `2023-09-01` | 111 | 10 | 101 | 9.0% | 361 | 28 |
| `2023-10-01` | 130 | 17 | 113 | 13.1% | 338 | 32 |
| `2023-11-01` | 140 | 16 | 124 | 11.4% | 318 | 42 |
| `2023-12-01` | 157 | 24 | 133 | 15.3% | 292 | 51 |
| `2024-01-01` | 163 | 19 | 144 | 11.7% | 272 | 65 |
| `2024-02-01` | 163 | 19 | 144 | 11.7% | 257 | 80 |
| `2024-03-01` | 172 | 25 | 147 | 14.5% | 242 | 86 |
| `2024-04-01` | 179 | 27 | 152 | 15.1% | 217 | 104 |
| `2024-05-01` | 186 | 32 | 154 | 17.2% | 195 | 119 |
| `2024-06-01` | 185 | 38 | 147 | 20.5% | 173 | 142 |
| `2024-07-01` | 188 | 46 | 142 | 24.5% | 152 | 160 |
| `2024-08-01` | 187 | 41 | 146 | 21.9% | 126 | 187 |
| `2024-09-01` | 181 | 44 | 137 | 24.3% | 104 | 215 |
| `2024-10-01` | 179 | 47 | 132 | 26.3% | 80 | 241 |
| `2024-11-01` | 176 | 42 | 134 | 23.9% | 48 | 276 |
| **TOTAL** | **2,890** | **485** | **2,405** | **16.78%** | — | — |

---

## 4. Deep Relational Audit: Churn Events vs Subscriptions

### A. Accounts with Churn Events but Active Subscriptions
- In `subscription_active`, **1,839 snapshots (42.9%)** belong to accounts that had $\ge 1$ churn event prior to $T_{\text{obs}}$.
- **Cause**: RavenStack was generated synthetically without closing subscriptions upon churn events (only 9.7% of subscriptions ever have an `end_date`).
- **Engine Handling**: The engine treats the customer as active at $T_{\text{obs}}$ because their contract is open. If they experience a subsequent churn event in $(T_{\text{obs}}, T_{\text{obs}} + 60\text{d}]$, `target_churn_60d = 1`. If no churn event occurs in the forward window, `target_churn_60d = 0`.

### B. Accounts without Active Subscriptions but No Prior Churn
- In `churn_free`, **445 snapshots (15.4%)** belong to accounts that **do not hold any active subscription** at $T_{\text{obs}}$ (their first subscription starts after $T_{\text{obs}}$).
- **Hazard Observed**: Under `churn_free`, **82 snapshots** receive target `1` for churn events that occur *before the customer's first subscription ever started* (e.g. Account `A-c7ffc2` signed up 2023-01-27, churned 2023-04-15, but earliest subscription start is 2023-05-25).
- In `subscription_active`, these pre-subscription accounts are strictly **excluded** from eligibility, preventing pre-contract synthetic churn leakage.

---

## 5. Window Boundary & Churn Multiplicity Verification

1. **Lower Boundary ($c.\text{churn\_date} = T_{\text{obs}}$)**:
   - There are **14 churn events** occurring exactly on a monthly snapshot date $T_{\text{obs}}$.
   - Code path: `acc_churns["parsed_churn_date"].dt.date > snapshot_date` (strict inequality).
   - **Verification**: Zero events on $T_{\text{obs}}$ leak into future targets. All 14 are correctly treated as past/present.
2. **Upper Boundary ($c.\text{churn\_date} = T_{\text{obs}} + 60\text{d}$)**:
   - There are **18 churn events** occurring on the exact 60th day ($T_{\text{obs}} + 60\text{d}$).
   - Code path: `acc_churns["parsed_churn_date"].dt.date <= window_end` (inclusive inequality).
   - **Verification**: All 18 events are captured as target `1`.
3. **Repeat Churn Events within a Single 60-Day Window**:
   - There are **123 snapshot instances** where an account records $\ge 2$ churn events within the 60-day prediction window (up to 4 churn events in a single 60-day window).
   - Code path: `1 if len(in_window) > 0 else 0`.
   - **Verification**: All evaluate to binary target `1` without duplicate rows or runtime errors.

---

## 6. Policy Recommendations & Comparison

| Policy | Snapshot Count | Positives (Target=1) | Churn Rate | Prediction Question & Operational Fit | Synthetic Data Limitation |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **`subscription_active`** *(Current Default)* | **4,284** | 680 | 15.87% | *"Given an account holding an active subscription at $T_{\text{obs}}$, will they experience a churn event within 60 days?"* Standard monthly retention scoring. | Includes accounts that experienced churn events in previous months because 90.3% of synthetic subscriptions never end. |
| **`churn_free`** *(Phase 2 Replication)* | **2,890** | 485 | 16.78% | *"Given an account that has never churned, will they experience their first churn event in 60 days?"* First-time churn only. | Qualifies accounts before their subscription starts; 82 targets are assigned before contract inception. |
| **`contract_active_first_churn`** *(Hybrid Recommendation)* | **2,445** | 403 | 16.48% | *"Given an account holding an active subscription at $T_{\text{obs}}$ AND having zero prior churn events, will they experience their first churn within 60 days?"* | Best of both worlds: Excludes pre-subscription accounts and excludes un-terminated repeat churners. |

### Architectural Recommendation for Phase 5 Feature Engineering
- **Recommendation**: Maintain `subscription_active` as the operational model baseline, or adopt `contract_active_first_churn` if the project objective is strictly **first-time churn prediction**.
- Both policies are deterministic, fully implemented, and verified in the test suite.
