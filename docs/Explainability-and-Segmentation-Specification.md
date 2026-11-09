# Flurit Model Explainability & Customer Segmentation Specification

**Phase**: Phase 7 — Model Explainability & Customer Segmentation  
**Date**: October 14, 2026  
**Auditor & Architect**: Senior Software Architect & ML Platform Engineer  
**Dataset**: RavenStack SaaS Analytics (`data/raw/ravenstack/`)  
**Base Pipeline**: Phase 6.1 Approved Logistic Regression Pipeline (`artifacts/models/churn_prediction_v1/`)  
**Status**: Implemented, Evaluated, Benchmarked, and Verified  

---

## 1. Executive Summary & Architecture Overview

Phase 7 implements Flurit's interpretability and unsupervised behavioral segmentation layer on top of the approved Phase 6.1 point-in-time feature matrix and champion `LogisticRegression` pipeline.

### Core Architecture Invariants
1. **Mathematically Sound Linear Explainer**: `TreeExplainer` is specialized for decision tree ensembles and fundamentally incompatible with linear models. Flurit implements `shap.LinearExplainer` with an independent background masker, computing exact additive Shapley attributions in log-odds space.
2. **Canonical Feature Aggregation**: The 61 one-hot encoded and standardized features are analytically aggregated back into Flurit's 45 canonical point-in-time features, strictly preserving the efficiency axiom:
   $$\phi_0 + \sum_{f=1}^{45} \Phi_f = z = \text{decision\_function}(X)$$
3. **Observational Association vs. Causation**: Feature attributions reflect correlational shifts in predicted model score relative to the background training baseline. Every explanation explicitly disclaims causal guarantees or intervention outcomes.
4. **Unsupervised Segmentation Discipline**: K-Means clustering is fitted **strictly on the development training partition** ($N=2,026$). All target labels (`target_churn_60d`, `churn_flag`) and post-snapshot fields are strictly excluded.
5. **Cluster Evaluation ($K=2..6$)**: Systematic benchmarking across candidate cluster counts using silhouette score, Davies-Bouldin index, Calinski-Harabasz index, inertia, and cluster size balance.
6. **Estimated Revenue at Risk**: Customer Value (contract MRR) is combined with Model Risk Scores into an operational prioritization matrix. In accordance with Phase 6.1's negative Brier skill score findings, model scores are explicitly treated as uncalibrated risk rankings rather than literal actuarial probabilities.
7. **Test-Set Inspection Disclosure**: Carried forward from Phase 6.1, earlier inspection of test cohorts prior to temporal purging is recorded across all documentation.

---

## 2. SHAP Linear Explanations & Mathematical Lineage

### Why LinearExplainer (and Not TreeExplainer)
The approved Phase 6.1 champion is a `LogisticRegression(class_weight='balanced')` estimator. `TreeExplainer` relies on tree path traversal and is inapplicable to linear hyperplanes. `shap.LinearExplainer` computes exact, closed-form Shapley values directly from the model's weight vector $\boldsymbol{\beta}$ and background feature expectations $\mathbb{E}[\mathbf{x}]$:
$$\phi_j(x) = \beta_j \left( x_j - \mathbb{E}[x_j] \right)$$

### Aggregation to Canonical Features
Flurit's preprocessor generates 61 transformed columns from 45 input features (4 one-hot encoded categoricals, 41 standardized numericals). Because Shapley values are strictly additive across disjoint feature partitions, the attribution of a categorical feature $C$ is the exact sum of its dummy variable attributions:
$$\Phi_C = \sum_{k \in \text{categories}(C)} \phi_{C=k}$$

For any customer snapshot, the sum of canonical attributions plus the expected base value equals the log-odds decision function:
$$\text{Log-Odds } z = \phi_0 + \sum_{f=1}^{45} \Phi_f$$
$$\text{Model Risk Score } p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Numerical verification on test snapshots confirms this identity holds to machine precision ($|z - (\phi_0 + \sum \Phi_f)| \le 4.44 \times 10^{-16}$).

### Global Feature Importance (Held-Out Test Cohort)

Evaluated across the held-out test cohort ($N=856$ snapshots, dates `2024-10-01` and `2024-11-01`):

| Rank | Feature Name | Mean \|SHAP\| | Primary Direction | Domain Group | Substantive Model Meaning |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `sub_active_count` | **0.6777** | Positive (+) | `subscription` | High count of active contract lines raises model score |
| **2** | `sub_total_active_seats` | **0.5166** | Negative (-) | `subscription` | Larger seat volume lowers model score (protective) |
| **3** | `usage_count_30d` | **0.4026** | Positive (+) | `usage` | High short-term usage activity raises model score |
| **4** | `usage_active_days_90d` | **0.3320** | Negative (-) | `usage` | Consistent active days over 90d lowers model score |
| **5** | `sub_tenure_days` | **0.2874** | Negative (-) | `subscription` | Longer commercial contract tenure lowers model score |
| **6** | `usage_distinct_features_30d` | **0.2809** | Negative (-) | `usage` | Broad product module adoption lowers model score |
| **7** | `usage_count_90d` | **0.2556** | Positive (+) | `usage` | Medium-term volume attribution |
| **8** | `account_industry` | **0.2178** | Positive (+) | `account` | Industry vertical categorical attribution |
| **9** | `usage_count_60d` | **0.2176** | Positive (+) | `usage` | 60-day telemetry volume attribution |
| **10** | `churn_prior_count` | **0.2168** | Negative (-) | `churn_history` | Historical churn frequency attribution |
| **11** | `account_age_days` | **0.2077** | Negative (-) | `account` | Overall customer lifecycle age |
| **12** | `account_initial_seats` | **0.1923** | Positive (+) | `account` | Initial seat count at signup |
| **13** | `churn_has_prior` | **0.1644** | Positive (+) | `churn_history` | Binary indicator of prior churn occurrence |
| **14** | `account_country` | **0.1614** | Negative (-) | `account` | Geographic location attribution |
| **15** | `tickets_urgent_count_90d` | **0.1579** | Negative (-) | `support` | Recent urgent support escalations |

### Sample Local Explanations Across Risk Tiers

#### High-Risk Customer Snapshot: `A-00cac8` (Snapshot Date: `2024-10-01`)
- **Model Churn Score**: **0.5628** (Classified: High Risk, $\ge 0.50$)
- **Base Log-Odds**: $+0.1038$ | **Predicted Log-Odds**: $+0.2524$
- **Top Factors Raising Score**:
  1. `sub_active_count` (value = 7): $\text{SHAP} = \mathbf{+0.6192}$
  2. `usage_distinct_features_30d` (value = 1): $\text{SHAP} = \mathbf{+0.2437}$
  3. `account_country` (value = AU): $\text{SHAP} = \mathbf{+0.1873}$
- **Top Factors Lowering Score**:
  1. `sub_tenure_days` (value = 381 days): $\text{SHAP} = \mathbf{-0.3919}$
  2. `usage_active_days_90d` (value = 4 days): $\text{SHAP} = \mathbf{-0.2826}$
- **Plain-Language Summary**: *"Estimated model churn score is 0.56. Main factors associated with higher score: sub active count (7), usage distinct features 30d (1), account country (AU). Mitigating factors associated with lower score: sub tenure days (381), usage active days 90d (4)."*

#### Moderate-Risk Customer Snapshot: `A-00bed1` (Snapshot Date: `2024-10-01`)
- **Model Churn Score**: **0.4982** (Classified: Moderate Risk, $[0.35, 0.50)$)
- **Top Factors Raising Score**: `sub_active_count` (10, $+1.1304$), `churn_days_since_last` (272d, $+0.4410$)
- **Top Factors Lowering Score**: `sub_total_active_seats` (342 seats, $-1.0117$), `usage_count_90d` (100 events, $-0.3385$)

---

## 3. Unsupervised Customer Segmentation (K-Means)

### Feature Preprocessing & Leakage Prevention
- **Feature Space**: All 45 point-in-time behavioral, commercial, usage, and ticket features.
- **Target Exclusion**: `target_churn_60d`, `churn_flag`, and post-snapshot fields are strictly barred from the pipeline by [`CustomerSegmenter._verify_no_leakage()`](file:///c:/flurit-backend/app/ml/segmentation/clustering.py#L134-L142).
- **Partition Isolation**: Preprocessing (`StandardScaler` on numericals, `OneHotEncoder` on categoricals) and K-Means cluster centroids are fit **exclusively on the training development partition** ($N=2,026$).

### Candidate Cluster Evaluation ($K=2$ to $K=6$)

Evaluated on the training development partition ($N=2,026$, 61 transformed dimensions):

| $K$ | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index | Inertia (WCSS) | Cluster Sample Sizes | Cluster Size % Distribution |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$K=2$** | **0.1117** | 2.7201 | **243.4** | 77,343.5 | [1,195, 831] | 59.0% / 41.0% |
| **$K=3$** (Champion) | **0.0985** | 2.6909 | 200.5 | 72,310.8 | [885, 601, 540] | **43.7% / 29.7% / 26.7%** |
| **$K=4$** | 0.1038 | 2.1855 | 177.2 | 68,608.4 | [17, 584, 515, 910] | 0.8% / 28.8% / 25.4% / 44.9% |
| **$K=5$** | 0.0960 | 2.1522 | 172.1 | 64,633.2 | [402, 441, 715, 451, 17] | 19.8% / 21.8% / 35.3% / 22.3% / 0.8% |
| **$K=6$** | 0.0627 | 2.1585 | 157.0 | 62,393.4 | [426, 539, 359, 400, 17, 285] | 21.0% / 26.6% / 17.7% / 19.7% / 0.8% / 14.1% |

### Champion Selection Rationale ($K=3$)
- **Degenerate Micro-Cluster Rejection**: For $K=4$, $K=5$, and $K=6$, the clustering algorithm isolates an anomalous micro-cluster of exactly **17 accounts** ($0.8\%$ of the population), leaving the remaining accounts partitioned sub-optimally.
- **Balanced Operational Partitioning**: $K=3$ achieves a well-distributed split (43.7%, 29.7%, 26.7%) without micro-clusters, providing operational utility for customer success workflows.
- While $K=2$ achieves a slightly higher silhouette score (0.1117 vs 0.0985), a binary split provides insufficient segmentation granularity for tiered customer interventions. $K=3$ is selected as Flurit's champion segmentation model.

### Empirical Cluster Profiles ($K=3$)

Centroids computed strictly from observed development data ($N=2,026$):

| Metric | Cluster 0: Standard Accounts | Cluster 1: Enterprise Scale | Cluster 2: Established Growth |
| :--- | :---: | :---: | :---: |
| **Sample Size ($N$)** | 885 (43.7%) | 601 (29.7%) | 540 (26.7%) |
| **Avg Monthly Recurring Revenue (MRR)** | **$5,255.98** | **$9,622.05** | **$5,974.99** |
| **Avg Annual Recurring Revenue (ARR)** | $63,071.74 | $115,464.63 | $71,699.88 |
| **Avg Licensed Seats** | 78.5 seats | **125.3 seats** | 90.1 seats |
| **Avg Commercial Tenure** | 143.0 days | **164.2 days** | 132.4 days |
| **Avg 30-Day Product Usage** | 12.3 actions | **39.3 actions** | 15.8 actions |
| **Avg Active Days (Past 30d)** | 1.2 days | **3.7 days** | 1.6 days |
| **Avg 90-Day Support Tickets** | 0.1 tickets | 0.3 tickets | **1.3 tickets** |
| **Avg Prior Churn Events** | 0.39 events | 0.33 events | 0.36 events |
| **Dominant Billing Frequency** | `monthly` | `annual` | `monthly` |

### Methodological Limitations of K-Means
1. **Spherical Variance Assumption**: K-Means assumes isotropic spherical clusters and equal variance across dimensions.
2. **Standardization Sensitivity**: Standardizing all 61 dimensions places equal geometric weight on sparse telemetry indicators as on core commercial fields (MRR).
3. **Telemetry Density**: Low baseline telemetry activity in RavenStack creates high clustering density near zero usage.

---

## 4. Customer Value × Model Risk Prioritization Analysis

### Mathematical Metric Definition
$$\text{Estimated Revenue at Risk} = \text{Current MRR} \times \text{Model Churn Score}$$

> [!CAUTION]
> **Probabilistic Disclaimer**: As established in Phase 6.1, raw Logistic Regression probabilities have negative Brier skill scores ($\text{BSS} = -0.2554$). Consequently, model scores represent **uncalibrated ranking/risk scores**, NOT literal actuarial probabilities. Estimated Revenue at Risk is an operational prioritization heuristic, NOT guaranteed financial loss.

### Prioritization Quadrants (Held-Out Test Cohort: $N=856$ Accounts)

| Quadrant Name | Priority | Accounts | % Share | Total Portfolio MRR ($) | Estimated Revenue at Risk ($) | Recommended Retention Action |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Priority 1: High Value - High Risk** | **1** | **164** | 19.2% | **$2,659,452.00** | **$1,580,905.21** | Immediate executive outreach, contract review, and dedicated technical audit. |
| **Priority 2: Mid Value - High Risk** | **2** | **60** | 7.0% | **$368,092.00** | **$221,605.15** | Proactive CS contact, satisfaction review, and concession modeling. |
| **Priority 3: High Value - Moderate Risk** | **3** | **173** | 20.2% | **$3,177,590.00** | **$1,320,750.72** | Account health monitoring and executive sponsor check-in. |
| **Priority 4: Core Value - High Risk** | **4** | **60** | 7.0% | **$112,427.00** | **$67,726.40** | Automated retention workflows, product surveys, self-service nurture. |
| **Priority 5: Standard Monitoring** | **5** | **399** | 46.6% | **$6,881,701.00** | **$1,456,828.66** | Ongoing lifecycle engagement, standard health check-ins, product updates. |
| **Total Test Portfolio** | — | **856** | 100.0% | **$13,199,262.00** | **$4,647,816.13** | Portfolio-wide risk allocation. |

---

## 5. Persisted Artifacts & Lineage

The explainability and segmentation pipelines generate serializable artifacts adhering to the Phase 3 database model structures:

```
artifacts/
├── explainability/
│   ├── global_feature_importance.json    # Ranked 45 canonical features with mean |SHAP|
│   └── sample_local_explanations.json    # Detailed local factor attributions & summaries
└── segmentation/
    ├── segmentation_model.joblib         # Fitted preprocessor + KMeans(n_clusters=3)
    ├── cluster_evaluation_metrics.json   # Full K=2..6 evaluation table
    ├── cluster_profiles.json             # Empirical centroids & descriptions for K=3
    └── risk_value_matrix.json            # 5-tier Value x Risk quadrant accounting
```

---

## 6. Verification Suite Results

### Automated Tests
- **Explainability Suite** ([`tests/test_explainability.py`](file:///c:/flurit-backend/tests/test_explainability.py)):
  - `test_additive_shapley_efficiency_identity`: Proof that $\sum \Phi_f + \phi_0 = z$.
  - `test_deterministic_explanations`: Verified reproducibility across runs.
  - `test_local_explanation_structure_and_disclaimer`: Verified factors, summaries, and disclaimers.
  - `test_global_feature_importance_completeness`: Verified ranking of all 45 features.
  - `test_rejection_of_incompatible_classifier`: Verified rejection of non-linear models.
- **Segmentation Suite** ([`tests/test_segmentation.py`](file:///c:/flurit-backend/tests/test_segmentation.py)):
  - `test_prohibited_target_labels_rejection`: Verified rejection of `target_churn_60d`.
  - `test_k_range_evaluation`: Verified metrics across $K=2..6$.
  - `test_deterministic_clustering_and_profiling`: Verified cluster reproducibility.
  - `test_feature_order_consistency`: Verified invariance to feature column shuffling.
  - `test_artifact_save_and_load_roundtrip`: Verified artifact persistence and loading parity.
  - `test_value_risk_quadrant_analysis`: Verified Value $\times$ Risk matrix calculations.

```powershell
.\.venv\Scripts\pytest.exe -v
# 63 passed in 7.58s (100% passing across all 10 test modules)
```

### Code Quality & Typing
- **Linter (`ruff check .`)**: All checks passed.
- **Formatter (`ruff format --check .`)**: 71 files formatted.
- **Type Checker (`mypy app tests scripts alembic`)**: Success: no issues found in 60 source files.

---

## Restrictions Verified
- Zero modifications to the approved champion model (`LogisticRegression`).
- Zero modifications to approved snapshot labels or source CSVs.
- Zero connections to PostgreSQL, Supabase, or Bynalt infrastructure.
- Zero Docker / container files created.
- Zero Git commits or pushes executed.

Phase 7 is complete and ready for technical lead review.
