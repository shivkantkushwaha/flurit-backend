# Flurit Churn Model Training & Evaluation Specification (Purged Audit)

**Phase**: Phase 6.1 — Purged Temporal Evaluation Audit  
**Date**: October 14, 2026  
**Auditor & Architect**: Senior Software Architect & ML Platform Engineer  
**Dataset**: RavenStack SaaS Analytics (`data/raw/ravenstack/`)  
**Feature Matrix**: 4,284 point-in-time snapshots, 45 engineered features  
**Status**: Audited, Corrected, Benchmarked, and Verified  

---

## 1. Executive Summary & Audit Purpose

Phase 6.1 audits and corrects Flurit's machine learning evaluation protocol. In initial evaluation designs, consecutive chronological partitioning allowed the 60-day prediction windows of boundary cohorts to overlap with adjacent observation periods. 

Specifically:
- The training cohort at $T_{\text{obs}} = \text{2024-05-01}$ had a 60-day label window spanning $(2024\text{-}05\text{-}01, 2024\text{-}06\text{-}30]$. A validation partition starting on `2024-06-01` had its observation date occur while training outcomes were still actively resolving.
- The validation cohort at $T_{\text{obs}} = \text{2024-08-01}$ had a 60-day label window spanning $(2024\text{-}08\text{-}01, 2024\text{-}09\text{-}30]$, overlapping a test partition starting on `2024-09-01`.

To establish an institutional-grade, leakage-free evaluation protocol, Phase 6.1 implements **purged chronological partitioning** with 60-day embargo buffers, recalculates all partition demographics and candidate model metrics, corrects Brier-score calibration interpretation against constant-prevalence baselines, discloses earlier test inspections, and audits feature derivations and sentinels.

---

## 2. Purged Chronological Split Architecture

Partitions are segmented strictly by `snapshot_date`, with intermediate 60-day embargo buffers discarded to guarantee zero target-window overlap:

```
[------------- TRAIN -------------] [PURGED] [--- VAL ---] [PURGED] [--- TEST ---]
2023-04-01               2024-05-01 2024-06-01 2024-07-01  2024-08-01 2024-09-01 2024-10-01  2024-11-01
       (14 monthly cohorts)         (1 cohort)   (2 cohorts)    (1 cohort)   (2 cohorts)
```

### Partition Demographics & Census Accounting

| Partition / Cohort | Date Range | Cohorts | Total Samples | Churn ($Y=1$) | Retained ($Y=0$) | Baseline Prevalence | Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Train** | `2023-04-01` to `2024-05-01` | 14 | 2,026 | 243 | 1,783 | **11.99%** | Model training & preprocessing fitting |
| *Purged Buffer 1* | `2024-06-01` | 1 | 313 | 52 | 261 | 16.61% | Discarded: overlaps train 60d label window |
| **Validation** | `2024-07-01` to `2024-08-01` | 2 | 704 | 125 | 579 | **17.76%** | Model benchmarking & champion selection |
| *Purged Buffer 2* | `2024-09-01` | 1 | 385 | 76 | 309 | 19.74% | Discarded: overlaps val 60d label window |
| **Held-Out Test** | `2024-10-01` to `2024-11-01` | 2 | 856 | 184 | 672 | **21.50%** | Final generalization assessment |
| **Total Census** | `2023-04-01` to `2024-11-01` | 20 | 4,284 | 680 | 3,604 | **15.87%** | Complete audited dataset |

### Strict Target-Window Purging Proof
1. **Train $\to$ Val Isolation**:
   - Max train date: `2024-05-01`.
   - 60-day target window closes: $2024\text{-}05\text{-}01 + 60\text{ days} = 2024\text{-}06\text{-}30$.
   - Min validation date: `2024-07-01`.
   - $\text{Gap} = 2024\text{-}07\text{-}01 - 2024\text{-}06\text{-}30 = 1\text{ day} > 0$. **Zero target-window overlap.**
2. **Val $\to$ Test Isolation**:
   - Max validation date: `2024-08-01`.
   - 60-day target window closes: $2024\text{-}08\text{-}01 + 60\text{ days} = 2024\text{-}09\text{-}30$.
   - Min test date: `2024-10-01`.
   - $\text{Gap} = 2024\text{-}10\text{-}01 - 2024\text{-}09\text{-}30 = 1\text{ day} > 0$. **Zero target-window overlap.**

---

## 3. Feature Lineage & Sentinel Audit

### A. Subscription Upgrade / Downgrade Features
- **Source Field Verification**: Confirmed in `ravenstack_subscriptions.csv`:
  - `upgrade_flag`: 529 True records, 4,471 False records.
  - `downgrade_flag`: 218 True records, 4,782 False records.
- **Leakage Prevention**: Evaluated strictly on subscriptions where `start_date <= T_obs`. Future contract amendments starting after $T_{\text{obs}}$ are strictly filtered out by [`SubscriptionFeatureExtractor`](file:///c:/flurit-backend/app/ml/features/subscription_features.py#L33-L35).

### B. Sentinel Value Handling
Four features employ `-1.0` sentinels to encode unobserved history:
- `usage_days_since_last`: `-1.0` if no telemetry prior to snapshot date.
- `tickets_avg_satisfaction_90d`: `-1.0` if no rated support tickets in past 90 days.
- `tickets_avg_first_response_minutes_90d`: `-1.0` if no tickets submitted in past 90 days.
- `churn_days_since_last`: `-1.0` if customer has never previously churned.

Each sentinel is accompanied by a binary indicator (`usage_has_history`, `tickets_has_unrated_90d`, `churn_has_prior`) and ticket volume counts. Preprocessing (one-hot encoding and feature scaling) is fit **exclusively on the training partition** ($N=2,026$).

---

## 4. Calibration & Brier Score Methodology

### Mathematical Framework
The Brier score measures the mean squared error of probability forecasts:
$$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$

A naive baseline forecasting constant probability equal to partition prevalence $P = \bar{y}$ achieves:
$$\text{Brier}_{\text{baseline}} = P(1 - P)$$

The Brier Skill Score (BSS) quantifies calibration improvement over this naive baseline:
$$\text{BSS} = 1 - \frac{\text{Brier}_{\text{model}}}{\text{Brier}_{\text{baseline}}}$$
- $\text{BSS} > 0$: Probabilities are better calibrated than trivial constant prevalence.
- $\text{BSS} = 0$: Performance is identical to predicting base rate $P$ everywhere.
- $\text{BSS} < 0$: Probabilities are worse calibrated than a trivial base rate forecast.

### Calibration Finding & Correction
In Phase 6, raw Brier scores were incorrectly described as "well calibrated" because the absolute values ($~0.18$) seemed small. However, when compared against the constant-prevalence baseline $P(1-P) = 0.1460$ (validation) and $0.1687$ (test):
- All models trained with `class_weight='balanced'` exhibit **negative Brier skill scores** ($\text{BSS} \in [-0.76, -0.23]$).
- **Root Cause**: `class_weight='balanced'` artificially reweights the positive class to 50%, shifting predicted probabilities upward toward 0.50. This increases squared error $(p_i - 0)^2$ on the majority negative class.
- **Audit Decision**: We explicitly retract any claim of "good calibration." Probability calibration (e.g. Platt scaling or isotonic regression) was **not** fitted post-hoc to prevent test-label contamination or overfitting on small validation cohorts ($N=704$).

---

## 5. Candidate Models Validation Comparison

Four candidate architectures were trained on the training partition ($N=2,026$) and benchmarked on the clean, purged validation partition ($N=704$):

*Validation baseline prevalence: **17.76%** (Brier baseline = **0.1460**).*

| Model Architecture | PR-AUC | ROC-AUC | Brier Score | BSS | Precision @ 0.50 | Recall @ 0.50 | F1 @ 0.50 | Confusion Matrix (TN / FP / FN / TP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Selected Champion) | **0.2010** | **0.5347** | 0.2309 | -0.5810 | **0.1868** | **0.3840** | **0.2513** | 370 / 209 / 77 / 48 |
| **Decision Tree** | 0.1699 | 0.4870 | 0.2577 | -0.7649 | 0.1805 | 0.4000 | 0.2488 | 352 / 227 / 75 / 50 |
| **Random Forest** | 0.1993 | 0.5289 | 0.1864 | -0.2766 | 0.1296 | 0.0560 | 0.0782 | 532 / 47 / 118 / 7 |
| **HistGradientBoosting** | 0.1742 | 0.4989 | 0.1806 | -0.2368 | 0.1594 | 0.0880 | 0.1134 | 521 / 58 / 114 / 11 |

### Champion Selection Rationale
- Under strict validation PR-AUC governance, **`logistic_regression`** achieved the highest PR-AUC (**0.2010** vs **0.1776** baseline prevalence), narrowly exceeding `random_forest` (0.1993) and comfortably surpassing `hist_gradient_boosting` (0.1742) and `decision_tree` (0.1699).
- On the unpurged validation split in Phase 6, `random_forest` had scored higher (0.2437) by capitalizing on the contaminated `2024-06-01` cohort. Once the 60-day buffer is purged, regularized linear modeling demonstrates the greatest stability.
- [`LogisticRegression`](file:///c:/flurit-backend/app/ml/training/pipeline.py#L65-L70) is officially designated as Flurit's champion architecture.

---

## 6. Held-Out Test Evaluation of Champion Model

> [!IMPORTANT]
> **Scientific Integrity Disclosure**: An unpurged test partition (`2024-09-01` to `2024-11-01`) was previously inspected in Phase 6 before this audit was executed. The test split has now been strictly frozen to `2024-10-01` through `2024-11-01` to eliminate the 60-day target window overlap from validation cohort `2024-08-01`.

### Test Cohort Demographics
- **Date Range**: `2024-10-01` to `2024-11-01` (2 cohorts)
- **Total Samples**: 856 customer snapshots
- **Actual Churners ($Y=1$)**: 184
- **Retained Customers ($Y=0$)**: 672
- **Baseline Test Prevalence**: **21.50%**
- **Test Brier Baseline ($P(1-P)$)**: **0.1687**

### Generalization Performance

| Metric | Test Value | Benchmark / Baseline | Delta / Status |
| :--- | :---: | :---: | :--- |
| **PR-AUC** | **0.2949** | 0.2150 (Prevalence) | **+7.99 percentage point lift** |
| **ROC-AUC** | **0.6205** | 0.5000 (Random Guessing) | Meaningful global discrimination |
| **Brier Score** | **0.2119** | 0.1687 (Constant Prevalence) | $\text{BSS} = -0.2554$ (uncalibrated probabilities) |
| **Decision Cutoff** | **0.50** | — | **Un-tuned default threshold (not optimized)** |
| **Precision @ 0.50** | **30.28%** | 21.50% (Base Rate) | **+8.78 percentage point precision lift** |
| **Recall @ 0.50** | **46.74%** | — | Captures 86 out of 184 total churners |
| **F1 Score @ 0.50** | **0.3675** | — | Solid balanced classification |

### Confusion Matrix (@ Threshold = 0.50)
```
                       Actual Negative (0)    Actual Positive (1)
Predicted Negative (0)         474 (TN)               98 (FN)
Predicted Positive (1)         198 (FP)               86 (TP)
```

### Threshold Clarification
The 0.50 decision threshold was **not tuned or optimized**. It represents the canonical classification threshold. At this default threshold, Logistic Regression flags 284 accounts, correctly identifying **86 churners** (46.74% recall) with **30.28% precision** (1.41x population base rate).

---

## 7. Persisted Artifacts & Metadata Lineage

Model artifacts are persisted in [`artifacts/models/churn_prediction_v1/`](file:///c:/flurit-backend/artifacts/models/churn_prediction_v1/):
- [`model.joblib`](file:///c:/flurit-backend/artifacts/models/churn_prediction_v1/model.joblib): Complete serialized pipeline (`StandardScaler` + `OneHotEncoder` + `LogisticRegression`).
- [`metadata.json`](file:///c:/flurit-backend/artifacts/models/churn_prediction_v1/metadata.json): Complete run metadata including:
  - `purged_evaluation_audit`: `"Phase 6.1 Purged Temporal Partitions"`
  - `prior_evaluation_disclosure`: Explicit disclosure of earlier Phase 6 test inspection.
  - `discarded_boundary_cohorts`: Complete record of purged cohorts (`2024-06-01` with 313 samples, `2024-09-01` with 385 samples).
  - `decision_threshold_tuned`: `false`.
  - `calibration_interpretation`: Explanation of Brier baseline and BSS.
  - `validation_comparison` & `test_evaluation`: Full metric suites.

---

## 8. Verification & Test Suite Results

The unit test suite in [`tests/test_training.py`](file:///c:/flurit-backend/tests/test_training.py) validates all purged splitting invariants:
- `test_strict_temporal_isolation_no_date_overlap`: Zero date sharing between partitions.
- `test_row_count_conservation`: Row accounting across partitions plus discarded cohorts.
- `test_target_window_purging_enforcement`: Rejection of partitions violating the 60-day target buffer.
- `test_purged_boundary_cohorts_identified`: Verification of discarded boundary cohort identification.
- `test_brier_baseline_and_skill_score_calculation`: Verification of $P(1-P)$ baseline and BSS logic.
- `test_save_and_load_roundtrip`: End-to-end artifact serialization and inference parity.
- `test_predict_proba_bounds`: Output probability validity across all 4 architectures.

All 52 tests pass across the codebase.
