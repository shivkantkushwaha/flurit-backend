# Flurit Snapshot & Target Construction Specification

**Phase**: Phase 4 — Snapshot & Target Construction Engine  
**Date**: October 13, 2026  
**Auditor & Architect**: Senior Software Architect & ML Platform Engineer  
**Status**: Implemented & Validated against RavenStack Dataset  

---

## 1. Investigation of `is_reactivation` & Account Lifecycle Ambiguities

### A. The Core Dilemma
In Task 1, we investigated how `is_reactivation` in `churn_events.csv` relates to the customer lifecycle:
* The RavenStack `README.md` states:
  > `| is_reactivation | boolean | 10% were previously churned |`
* In `churn_events.csv`, exactly **61 records (10.2%)** have `is_reactivation = True`.

### B. Empirical Source Findings
A rigorous cross-sectional analysis of all 600 churn events revealed critical ambiguities:
1. **First-Event Flagging**: Out of 55 unique accounts with an `is_reactivation = True` churn record, **26 accounts (47.3%) have `is_reactivation = True` on their very first churn event** in the dataset.
2. **Single-Event Flagging**: **13 accounts** have `is_reactivation = True` on their **sole churn event** in the entire 2-year dataset.
3. **Repeated Churns Without the Flag**: For accounts experiencing multiple churn events, **213 out of 248 second-or-later churn events (85.9%) have `is_reactivation = False`**.

### C. Architectural Assessment & Conclusion
`is_reactivation` in RavenStack is **not a deterministic state machine transition indicator**. It was simulated as an independent Bernoulli random trial ($\sim 10\%$ probability) attached to churn event records rather than a relational state tracker.

**Guiding Rule**:
> **`is_reactivation` must NOT be used to dictate account eligibility or state transitions.**
> 
> Instead, true commercial activity at snapshot date $T_{\text{obs}}$ must be determined by whether the customer holds an active, un-terminated subscription record at that date (`start_date <= T_obs` and `end_date IS NULL` or `end_date > T_obs`).

---

## 2. Snapshot Eligibility Policies

To support both realistic SaaS operations and conservative cohort modeling, the engine implements two deterministic eligibility policies:

### Policy A: `subscription_active` (Default / Production Recommended)
An account $A$ is eligible for snapshot $T_{\text{obs}}$ if:
1. `A.signup_date <= T_obs` (account created on or before observation date).
2. At date $T_{\text{obs}}$, account $A$ holds at least one active subscription:
   $$\exists s \in A.\text{subscriptions} : s.\text{start\_date} \le T_{\text{obs}} \land (s.\text{end\_date} \text{ IS NULL} \lor s.\text{end\_date} > T_{\text{obs}})$$
* **Advantage**: Matches operational SaaS realities where customers hold active contracts, and allows reactivated customers (who started a new subscription after a past cancellation) to be scored.

### Policy B: `churn_free` (Conservative Alternative)
An account $A$ is eligible for snapshot $T_{\text{obs}}$ if:
1. `A.signup_date <= T_obs`.
2. Account $A$ has had **zero** churn events prior to or on $T_{\text{obs}}$:
   $$\forall c \in A.\text{churn\_events} : c.\text{churn\_date} > T_{\text{obs}}$$
* **Advantage**: Strictly isolates first-time churn risk, permanently removing churned accounts.

---

## 3. Mathematical Definition of `target_churn_60d`

For any account $A$ eligible at observation date $T_{\text{obs}}$:

$$\text{target\_churn\_60d}(A, T_{\text{obs}}) = \begin{cases}
1 & \text{if } \exists c \in A.\text{churn\_events} : c.\text{churn\_date} \in (T_{\text{obs}}, T_{\text{obs}} + 60\text{d}] \\
0 & \text{if } \not\exists c \in A.\text{churn\_events} : c.\text{churn\_date} \in (T_{\text{obs}}, T_{\text{obs}} + 60\text{d}] \quad \land \quad T_{\text{obs}} \le T_{\text{cutoff}} \\
\text{NULL} & \text{if } T_{\text{obs}} > T_{\text{cutoff}} \quad (\text{right-censored, future unobserved})
\end{cases}$$

Where:
* $T_{\text{cutoff}} = T_{\max} - 60\text{ days} = 2024\text{-}12\text{-}31 - 60\text{ days} = 2024\text{-}11\text{-}01$.

---

## 4. Empirical Cohort Census Across All 20 Monthly Snapshots

Under the default `subscription_active` policy, the engine was executed against all 5 source tables in `data/raw/ravenstack/`:

| Snapshot Date | Eligible Active Cohort | Target = 1 (Churned $\le 60\text{d}$) | Target = 0 (Retained $\le 60\text{d}$) | Censored | 60-Day Churn Rate |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `2023-04-01` | 21 | 3 | 18 | 0 | **14.3%** |
| `2023-05-01` | 37 | 5 | 32 | 0 | **13.5%** |
| `2023-06-01` | 50 | 6 | 44 | 0 | **12.0%** |
| `2023-07-01` | 71 | 5 | 66 | 0 | **7.0%** |
| `2023-08-01` | 90 | 9 | 81 | 0 | **10.0%** |
| `2023-09-01` | 117 | 12 | 105 | 0 | **10.3%** |
| `2023-10-01` | 131 | 14 | 117 | 0 | **10.7%** |
| `2023-11-01` | 149 | 15 | 134 | 0 | **10.1%** |
| `2023-12-01` | 172 | 24 | 148 | 0 | **14.0%** |
| `2024-01-01` | 191 | 19 | 172 | 0 | **9.9%** |
| `2024-02-01` | 216 | 24 | 192 | 0 | **11.1%** |
| `2024-03-01` | 235 | 33 | 202 | 0 | **14.0%** |
| `2024-04-01` | 264 | 30 | 234 | 0 | **11.4%** |
| `2024-05-01` | 282 | 44 | 238 | 0 | **15.6%** |
| `2024-06-01` | 313 | 52 | 261 | 0 | **16.6%** |
| `2024-07-01` | 338 | 59 | 279 | 0 | **17.5%** |
| `2024-08-01` | 366 | 66 | 300 | 0 | **18.0%** |
| `2024-09-01` | 385 | 76 | 309 | 0 | **19.7%** |
| `2024-10-01` | 415 | 85 | 330 | 0 | **20.5%** |
| `2024-11-01` | 441 | 99 | 342 | 0 | **22.4%** |
| **TOTAL** | **4,284** | **680** | **3,604** | **0** | **15.9% (Overall)** |

*(Note: Under `churn_free` policy, total snapshots = 2,890, Target=1: 485 (16.8%), Target=0: 2,405 (83.2%)).*

---

## 5. Anti-Leakage Safeguards Verified

1. **Exclusion of `accounts.churn_flag`**: The static source column `accounts.churn_flag` is never referenced during snapshot eligibility or target construction.
2. **Strict Right-Censoring**: Any snapshot evaluated after `2024-11-01` is marked with `target_churn_60d = None` rather than being assigned a false negative ($0$).
3. **Exact Point-in-Time Boundary**: Churn events occurring on $T_{\text{obs}}$ itself are considered historical events ($t \le T_{\text{obs}}$) and are **never** counted as future churn targets.
4. **Deterministic Uniqueness**: Generation guarantees $0$ duplicate `(account_id, snapshot_date)` pairs.

---

## 6. Approved Eligibility Policy & CLI Integration

- **Formally Approved Policy**: **`subscription_active`** is confirmed and documented as the primary approved policy for production cohort generation and Phase 5 feature engineering.
- **Default CLI Configuration**: `scripts/build_snapshots.py` defaults to `--policy subscription_active`, guaranteeing that running without flags reproduces the canonical cohort dataset.
- **Zero Silent Label Modifications**: Snapshot records and labels remain deterministic and unaltered.

---

## 7. Target Semantics & Churn Event Data Integrity (Phase 4.2 Audit)

### A. Target Meaning: Account Churn Event vs. Subscription Cancellation
- **Target Definition**: `target_churn_60d = 1` denotes that an account-level churn event log entry exists in `ravenstack_churn_events.csv` with `churn_date` strictly within $(T_{\text{obs}}, T_{\text{obs}} + 60\text{d}]$.
- **Contractual Reality**: The target **does NOT guarantee subscription contract cancellation** or that `end_date` is populated in `ravenstack_subscriptions.csv`. In RavenStack's synthetic generation, 90.3% of subscriptions have `end_date IS NULL`, and 90.8% of churn events occur while subscriptions remain open. Downstream modeling must treat the target as **Customer Success Churn Intent / Recorded Churn Incident**, not an automated contract termination.

### B. Reconciliation of Churn Event Record Counts (600 vs. 598 Narrative Typo)
- **Source CSV Truth**: `ravenstack_churn_events.csv` contains exactly **600 valid records** (601 lines including header).
- **Integrity Status**:
  - `churn_event_id`: 600 unique primary keys (0 duplicates, 0 nulls).
  - Referential integrity: 352 unique accounts, all 352 exist in `ravenstack_accounts.csv` (0 orphans).
  - Dates: All 600 churn dates are valid and fall within `[2023-01-25, 2024-12-31]`.
  - Date Multiplicity: Exactly **599 unique `(account_id, churn_date)` tuples**. Account `A-cc102e` contains two distinct churn event records on `2024-11-19` (`C-e20cc6` and `C-ce47da`). Both records are valid and contribute to the target window.
  - Zero Records Dropped: Exactly 0 records were filtered, excluded, or invalidated.
- **Clarification**: The number "598" referenced in the Phase 4 completion narrative was an erroneous typographical transcription in the narrative text. All 600 rows in the raw CSV are fully loaded, validated, and processed by the engine.

### C. Confirmed Reproducibility Census
Direct execution from the raw CSV files reproduces the canonical numbers with 100% mathematical precision:
- **Total Customer Snapshots**: **4,284**
- **Positive Targets (`target_churn_60d = 1`)**: **680** (15.87%)
- **Negative Targets (`target_churn_60d = 0`)**: **3,604** (84.13%)
- **Right-Censored Snapshots ($\le 2024\text{-}11\text{-}01$)**: **0**
- **Post-Cutoff Censoring Check ($2024\text{-}12\text{-}01$)**: 476 snapshots all evaluate to `None`.

