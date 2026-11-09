# Backend & ML Model Implementation Blueprint (Technical Architecture)

# Flurit — Technical Build Plan

## 1. The architecture I recommend

Use four main pieces:

```
                    FLURIT
                       │
        ┌──────────────┴──────────────┐
        │                             │
   Next.js Frontend              FastAPI Backend
   TypeScript                    Python
        │                             │
        └──────────────┬──────────────┘
                       │
                    PostgreSQL
                       │
          ┌────────────┴─────────────┐
          │                          │
     Application Data             ML Layer
          │                          │
          │                scikit-learn / SHAP
          │                          │
          └────────────┬─────────────┘
                       │
                  Model Artifacts
```

### Stack

**Frontend**

- Next.js
- TypeScript
- Tailwind CSS
- Recharts or similar chart library

**Backend**

- Python
- FastAPI
- SQLAlchemy
- Pydantic

**Database**

- PostgreSQL

**ML**

- Pandas
- NumPy
- Scikit-learn
- SHAP
- Joblib

**Development**

- Docker / Docker Compose
- Git / GitHub

I would **not** use Kafka, Spark, Kubernetes, microservices, etc. for this version. They add infrastructure without improving the actual Flurit project.

---

# 2. First: freeze the prediction problem

This is more important than choosing the algorithm.

We should not define Flurit as:

> “Predict whether an account has ever churned.”
> 

That would make the project retrospective.

Instead:

> **At a particular point in time, predict whether an active customer will churn during the following prediction window.**
> 

For the first version, I recommend a **60-day prediction window**.

So each training example looks like:

```
Customer A
Snapshot date: 2024-06-01

Information available up to 2024-06-01
        ↓
Model
        ↓
Did this customer churn between
2024-06-01 and 2024-07-31?
        ↓
YES / NO
```

This makes Flurit genuinely predictive.

A recent independent implementation of RavenStack used a similar time-aware framing and explicitly avoided random splits and post-hoc information leakage. [GitHub](https://github.com/Himanshu-2678/early-churn-intelligence?utm_source=chatgpt.com)

---

# 3. Backend development — exact order

Do it in this order.

## Phase 1 — Create the repository

```
flurit/
│
├── frontend/
├── backend/
├── ml/
├── data/
├── docs/
├── docker-compose.yml
└── README.md
```

Don't start with UI.

First build the data/backend foundation.

---

# 4. Phase 2 — Download and inspect the raw dataset

Put the five source files into:

```
data/raw/
```

Expected source tables:

```
ravenstack_accounts.csv
ravenstack_subscriptions.csv
ravenstack_feature_usage.csv
ravenstack_support_tickets.csv
ravenstack_churn_events.csv
```

The published schema includes fields such as account metadata, MRR and plan information, usage activity, support activity, and churn events. [GitHub](https://github.com/WizardNox/End-to-End-SaaS-Analytics-Engineering-Platform?utm_source=chatgpt.com)

### First scripts

Create:

```
backend/
  scripts/
    inspect_dataset.py
    validate_dataset.py
```

`inspect_dataset.py` should report:

```
rows
columns
dtypes
nulls
unique IDs
date ranges
duplicate IDs
```

Do this **before writing any ML code**.

---

# 5. Phase 3 — PostgreSQL schema

Don't train directly from CSVs.

Load the source data into PostgreSQL.

### Raw/source tables

```
accounts
subscriptions
feature_usage
support_tickets
churn_events
```

These should roughly mirror the source files.

### Then create derived tables

```
customer_snapshots
customer_features
customer_predictions
customer_explanations
customer_segments
model_runs
```

This separation matters.

Raw data stays raw.

Derived ML/business data can change.

---

# 6. Phase 4 — Build the relationships

The source structure is essentially:

```
accounts
   │
   ├───────────────┐
   │               │
   ▼               ▼
subscriptions   support_tickets
   │
   ▼
feature_usage

accounts
   │
   ▼
churn_events
```

The published schema confirms the key relationships: subscriptions reference accounts, feature usage references subscriptions, while support tickets and churn events reference accounts. [GitHub](https://github.com/WizardNox/End-to-End-SaaS-Analytics-Engineering-Platform?utm_source=chatgpt.com)

---

# 7. Phase 5 — Create the customer snapshot table

This is **the most important backend/data-engineering part of Flurit**.

Instead of training on raw events directly, we create a time-based customer snapshot.

Conceptually:

```
account_id
snapshot_date
```

For example:

```
A001 | 2024-03-01
A001 | 2024-04-01
A001 | 2024-05-01
A001 | 2024-06-01
```

Each snapshot represents:

> **What did we know about this customer at this point in time?**
> 

Then we build features from information available **before or on that snapshot date**.

This is what keeps the project realistic.

---

# 8. Phase 6 — Create the target label

For every snapshot:

```
target_churn_60d
```

Logic:

```
IF churn occurs within next 60 days
    → 1
ELSE
    → 0
```

But there is an important rule:

### Don't create negative labels where the future is unknown.

Suppose:

```
Snapshot = Dec 15
Dataset ends = Dec 31
```

We cannot know whether the customer churned during the next 60 days because the dataset doesn't go that far.

Those snapshots should not be used as normal negative training examples.

This is one of the things that will make the ML methodology much more defensible.

---

# 9. Phase 7 — Feature engineering

Now we turn the raw tables into **customer features**.

This is where Flurit becomes interesting.

## Account features

From `accounts`:

```
industry
country
seats
is_trial
plan_tier
tenure_days
tenure_months
```

Don't automatically use every field. We'll test usefulness.

---

## Subscription features

From `subscriptions`:

```
current_mrr
current_arr
billing_frequency
auto_renew
upgrade_count
downgrade_count
subscription_count
current_plan
days_until_end
```

The source dataset contains MRR, ARR, billing frequency, upgrade/downgrade and auto-renew fields. [GitHub](https://github.com/WizardNox/End-to-End-SaaS-Analytics-Engineering-Platform?utm_source=chatgpt.com)

---

## Product usage features

From `feature_usage`:

### Recent activity

```
usage_count_7d
usage_count_30d
usage_duration_7d
usage_duration_30d
active_usage_days_30d
feature_count_30d
error_count_30d
```

### Recency

```
days_since_last_usage
```

### Trend

This one is especially valuable:

```
usage_last_30d
vs
usage_previous_30d
```

Then:

```
usage_change_pct
```

For example:

```
Previous 30 days: 1,000
Recent 30 days:     600

Change = -40%
```

That is much more meaningful for churn than simply using total usage.

---

# 10. Support features

From `support_tickets`:

```
tickets_30d
tickets_90d
high_priority_tickets_30d
escalations_30d
avg_resolution_hours
avg_first_response_minutes
avg_satisfaction
```

And potentially:

```
recent_ticket_increase
```

Again, we're interested in **changes**, not just totals.

The source table includes priority, resolution time, first response time, satisfaction and escalation fields. [GitHub](https://github.com/WizardNox/End-to-End-SaaS-Analytics-Engineering-Platform?utm_source=chatgpt.com)

---

# 11. Features we must NOT use for prediction

This is extremely important.

Some fields tell us about the churn **after it happened**.

Using them would be data leakage.

For example:

```
churn_date
reason_code
feedback_text
refund_amount
```

And potentially:

```
churn_flag
```

depending on where it is stored and how it was generated.

These can be used for **analysis after the fact**, but not as predictive input.

For example:

```
BAD:

churn_date
     ↓
predict
     ↓
churn
```

That's cheating.

Instead:

```
past behaviour
past support
past subscription data
       ↓
predict future churn
```

---

# 12. Final ML dataset

After feature engineering, we want something like:

| account_id | snapshot_date | mrr | usage_30d | usage_change | tickets_30d | satisfaction | tenure | auto_renew | target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A001 | 2024-05-01 | 800 | 120 | -35% | 3 | 4.1 | 14 | 0 | 1 |
| A002 | 2024-05-01 | 2400 | 680 | +8% | 0 | 4.8 | 26 | 1 | 0 |

This is the dataset that actually goes into ML.

---

# 13. Phase 8 — Train/validation/test split

Do **not** randomly split the complete dataset.

For a time-based churn problem, we want:

```
PAST                    FUTURE
─────────────────────────────────────
TRAIN       VALIDATION       TEST
```

For example:

```
earliest 70% → train
next 15%     → validation
latest 15%   → test
```

The exact dates will be decided after we inspect the dataset.

Inside the training period, we can use **TimeSeriesSplit** for cross-validation.

This avoids training on the future and evaluating on the past.

---

# 14. Phase 9 — Preprocessing pipeline

We build preprocessing as part of the ML pipeline.

### Numeric columns

Examples:

```
mrr
tenure
usage_30d
usage_change
ticket_count
satisfaction
```

Actions:

```
missing value handling
optional outlier handling
scaling for linear model
```

### Categorical columns

Examples:

```
industry
country
plan_tier
billing_frequency
```

Actions:

```
missing value handling
One-Hot Encoding
```

Use `ColumnTransformer`.

Then the model and preprocessing become one reproducible pipeline.

---

# 15. Phase 10 — Models

I recommend four main models.

## Model 1 — Logistic Regression

This is our **baseline**.

Why?

It's simple, interpretable and directly covered by your syllabus.

It answers:

> Can a relatively simple model already predict churn reasonably well?
> 

---

## Model 2 — Decision Tree

Useful because:

- easy to understand
- naturally handles non-linear relationships
- produces human-readable decision rules

But we will control:

```
max_depth
min_samples_split
min_samples_leaf
```

to prevent overfitting.

---

## Model 3 — Random Forest

This should be one of our major models.

Why?

It handles:

- non-linear relationships
- interactions between features
- mixed feature types
- noisy data

And it fits directly into the ensemble-learning part of your syllabus.

We'll tune:

```
n_estimators
max_depth
min_samples_leaf
max_features
```

---

## Model 4 — Gradient Boosting

This is our second major ensemble model.

We'll compare it against Random Forest.

Parameters will include things such as:

```
n_estimators
learning_rate
max_depth
min_samples_leaf
```

For this project, I would initially use **scikit-learn GradientBoostingClassifier or HistGradientBoostingClassifier** rather than immediately bringing XGBoost into the project. It keeps the methodology closer to your course.

---

# 16. Do we use deep learning?

### No — not initially.

There is no good reason to throw an ANN into this project just to say "deep learning."

Our data isn't large enough to justify making that the center of the project, and the course already gives us strong classical models.

The goal is:

> **Good problem formulation + correct features + correct evaluation**
> 

rather than:

> “We used a neural network.”
> 

Later, we can experiment with an ANN as an optional comparison if there's a genuine reason.

---

# 17. Class imbalance

Churn will probably be the minority class.

So we first inspect:

```
Churn %
Non-churn %
```

Then compare approaches.

### Baseline

Normal training.

### Cost-sensitive learning

Use:

```
class_weight="balanced"
```

where supported.

### SMOTE

Only if needed.

And **SMOTE must be applied only to the training portion**, never before train/test splitting.

I would **not automatically use SMOTE** just because churn is imbalanced.

We should let the actual data tell us whether it's necessary.

---

# 18. Model evaluation

Don't make accuracy the main metric.

Our evaluation table will look like:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | — | — | — | — | — |
| Decision Tree | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| Gradient Boosting | — | — | — | — | — |

We'll also inspect:

- confusion matrix
- ROC curve
- Precision-Recall curve

### Why Recall matters

If Flurit misses a genuinely high-risk customer, the business has no chance to intervene.

So recall is important.

But precision matters too because we don't want to tell a company:

> “800 customers are high-risk”
> 

when most are actually fine.

Therefore we'll choose a sensible classification threshold instead of blindly using `0.5`.

---

# 19. Model selection

We should **not decide today** that Random Forest is the winner.

We run the experiments.

Then:

```
Best model
        ↓
Probability calibration check
        ↓
Threshold selection
        ↓
Final test evaluation
        ↓
Save model
```

For Flurit, the model should ideally produce a **probability**, not just 0/1.

---

# 20. Model artifact

Once the selected pipeline is finalized:

```
flurit/
  ml/
    artifacts/
      churn_model.joblib
      feature_schema.json
      model_metadata.json
```

`churn_model.joblib` will contain the preprocessing + model pipeline.

`model_metadata.json` can contain:

```
{
  "model": "RandomForestClassifier",
  "version": "1.0",
  "trained_at": "...",
  "features": [...],
  "target": "target_churn_60d"
}
```

---

# 21. Generate predictions

Now we run the selected model over the **latest valid customer snapshot**.

For each customer:

```
account_id
snapshot_date
churn_probability
risk_level
model_version
```

Example:

```
A1042
2024-11-01
0.87
HIGH
v1.0
```

Store this in:

```
customer_predictions
```

---

# 22. Risk levels

Keep it simple.

Initially:

```
0.00 – 0.39 → Low
0.40 – 0.69 → Medium
0.70 – 1.00 → High
```

But eventually, even these boundaries should be based on our validation results rather than chosen arbitrarily.

So:

**Risk score = model probability**

while:

**Risk level = business threshold**

These are different concepts.

---

# 23. SHAP / explainability

Once we have a selected model, we add explainability.

For tree models:

```
TreeExplainer
```

For each customer:

```
prediction = 0.87

Top contributors:
usage_change        +0.18
tickets_30d         +0.11
satisfaction        +0.07
auto_renew          +0.04
tenure              -0.03
```

We convert that into a simple explanation.

### Flurit UI

Instead of:

> SHAP value = 0.1842
> 

show:

> **Usage has fallen significantly in the recent period and is one of the main factors increasing this customer's predicted churn risk.**
> 

But the underlying explanation remains based on the model output.

---

# 24. Global explainability

Not only individual customers.

Flurit should also answer:

> **What are the major churn drivers across the customer base?**
> 

Example:

```
1. Usage decline
2. Support activity
3. Satisfaction
4. Auto-renew status
5. Tenure
```

This can power an Insights page.

---

# 25. Customer Value

For the first version, use something simple and defensible.

### Primary customer-value measure

**Current MRR**

Then:

```
Revenue at Risk
=
MRR × Churn Probability
```

Example:

```
MRR = $2,000
Risk = 80%

Estimated Risk-Weighted Revenue = $1,600
```

Important wording:

> **Revenue at Risk is an estimated risk-weighted amount, not guaranteed lost revenue.**
> 

---

# 26. Customer Value × Churn Risk matrix

Now create:

```
X-axis → Churn Risk
Y-axis → MRR / Customer Value
```

Each dot is a customer.

Quadrants:

```
                    HIGH CHURN RISK
                          │
       Monitor            │     Priority
                          │
──────────────────────────┼────────────────
                          │
       Low priority       │     Attention
                          │
                    LOW CHURN RISK
```

This is one of the most useful parts of Flurit.

---

# 27. Customer segmentation

This is separate from churn prediction.

Do **not** train K-Means with churn as a target.

Instead use customer behaviour/value features such as:

```
MRR
tenure
usage
usage frequency
feature breadth
support activity
```

Then:

### Standardize

```
StandardScaler
```

### Test cluster counts

```
K = 2
K = 3
K = 4
K = 5
K = 6
```

Use:

- Elbow method
- Silhouette score
- business interpretability

Then select a useful K.

Only after seeing the centroids should we name clusters.

For example, a cluster might turn out to be:

> **High-value, highly engaged**
> 

Another:

> **Low usage, high support activity**
> 

But the names should be based on the actual results.

---

# 28. Backend API layer

Once the data + ML layer works, FastAPI exposes it.

## Dashboard

```
GET /api/v1/dashboard/overview
GET /api/v1/dashboard/risk-distribution
GET /api/v1/dashboard/revenue-at-risk
GET /api/v1/dashboard/high-risk-customers
GET /api/v1/dashboard/risk-trend
```

## Customers

```
GET /api/v1/customers
GET /api/v1/customers/{account_id}
GET /api/v1/customers/{account_id}/risk
GET /api/v1/customers/{account_id}/explanation
GET /api/v1/customers/{account_id}/usage
GET /api/v1/customers/{account_id}/support
GET /api/v1/customers/{account_id}/subscription
GET /api/v1/customers/{account_id}/timeline
```

## Risk / analytics

```
GET /api/v1/analytics/risk-vs-value
GET /api/v1/analytics/customer-statistics
GET /api/v1/analytics/usage
GET /api/v1/analytics/support
```

## Segmentation

```
GET /api/v1/segments
GET /api/v1/segments/{segment_id}
GET /api/v1/segments/{segment_id}/customers
```

## Models

```
GET /api/v1/models
GET /api/v1/models/{model_id}
GET /api/v1/models/comparison
GET /api/v1/models/active
```

## Dataset

```
GET /api/v1/dataset
```

## Internal ML operations

```
POST /api/v1/internal/pipeline/ingest
POST /api/v1/internal/pipeline/features
POST /api/v1/internal/pipeline/train
POST /api/v1/internal/pipeline/predict
POST /api/v1/internal/pipeline/segment
GET  /api/v1/internal/pipeline/status
```

These internal routes don't need to be publicly exposed in the deployed application.

---

# 29. Backend folder structure

I would structure it like this:

```
backend/
│
├── app/
│   ├── main.py
│   │
│   ├── api/
│   │   └── routes/
│   │       ├── dashboard.py
│   │       ├── customers.py
│   │       ├── analytics.py
│   │       ├── segments.py
│   │       ├── models.py
│   │       └── dataset.py
│   │
│   ├── models/
│   │   ├── account.py
│   │   ├── subscription.py
│   │   ├── prediction.py
│   │   └── ...
│   │
│   ├── schemas/
│   ├── services/
│   │   ├── customer_service.py
│   │   ├── analytics_service.py
│   │   └── risk_service.py
│   │
│   ├── db/
│   │   ├── session.py
│   │   └── migrations/
│   │
│   └── core/
│
└── tests/
```

And separately:

```
ml/
│
├── ingestion/
├── preprocessing/
├── features/
├── training/
├── evaluation/
├── explainability/
├── segmentation/
├── inference/
├── artifacts/
└── notebooks/
```

Keep **ML logic out of API route files**.

---

# 30. The actual development order

This is what I would literally follow.

### Step 1

**Set up repository + Docker + PostgreSQL + FastAPI**

### Step 2

**Load all five RavenStack tables**

### Step 3

**Validate IDs, dates, duplicates, nulls and relationships**

### Step 4

**Create database schema**

### Step 5

**Build customer timeline/snapshot generation**

### Step 6

**Define the 60-day churn target**

### Step 7

**Build feature engineering pipeline**

### Step 8

**Create the final ML training dataset**

### Step 9

**Create chronological train/validation/test split**

### Step 10

**Build preprocessing pipeline**

### Step 11

**Train Logistic Regression**

### Step 12

**Train Decision Tree**

### Step 13

**Train Random Forest**

### Step 14

**Train Gradient Boosting**

### Step 15

**Compare models**

### Step 16

**Select threshold + final model**

### Step 17

**Save model artifact**

### Step 18

**Generate customer predictions**

### Step 19

**Generate SHAP explanations**

### Step 20

**Build customer-value/risk calculations**

### Step 21

**Build K-Means segmentation**

### Step 22

**Build FastAPI endpoints**

### Step 23

**Build dashboard**

### Step 24

**Build customer profile**

### Step 25

**Connect risk/explanation/segmentation APIs**

### Step 26

**Add tests**

### Step 27

**Dockerize everything**

### Step 28

**Deploy**

---

# 31. What the final ML pipeline looks like

This is the core architecture you should remember:

```
                 RAVENSTACK
                     │
          ┌──────────┼───────────┐
          │          │           │
       Accounts   Usage      Support
          │          │           │
          └──────────┼───────────┘
                     │
               Subscriptions
                     │
                     ▼
             Customer Snapshots
                     │
                     ▼
             Feature Engineering
                     │
                     ▼
              Churn Target
                     │
                     ▼
             Train / Val / Test
                     │
          ┌──────────┼──────────┐
          │          │          │
       Logistic    Random     Gradient
       Regression  Forest      Boosting
          │          │          │
          └──────────┼──────────┘
                     ▼
                Evaluation
                     │
                     ▼
              Selected Model
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
       Churn      SHAP       Customer
       Risk     Explanation   Value
          │          │          │
          └──────────┼──────────┘
                     ▼
             Flurit Analytics
                     │
                     ▼
                Dashboard
```

---

# 32. One thing I would NOT build yet

You previously talked about an AI chatbot/recommendation layer.

I would **leave that out of the first technical build**.

First make this work:

> **Data → prediction → explanation → business prioritization**
> 

Once that is working properly, an optional natural-language layer becomes easy to add on top.

For example:

> “Show me high-value customers with more than 70% churn risk.”
> 

The backend can already answer that through structured data. An AI layer can come later.

---

# 33. What makes this genuinely good

The strongest part of Flurit won't be that it has four ML models.

It will be these three engineering decisions:

### 1. Time-aware prediction

We predict **future churn**, rather than using retrospective information.

### 2. Real feature engineering

We don't just feed raw columns into Random Forest. We construct customer behaviour features from the relational tables.

### 3. Prediction + explanation + business value

We don't stop at:

> **Churn = 1**
> 

We reach:

> **Churn probability = 82%**
> 
> 
> **Why = usage decline + support activity**
> 
> **MRR = $2,400**
> 
> **Estimated revenue at risk = $1,968**
> 
> **Priority = High**
> 

That is the part that makes Flurit feel like a **complete product rather than an ML notebook**.

And the RavenStack dataset is particularly suited to this because its published schema separates accounts, subscriptions, usage, support tickets and churn events, which lets us build the customer-level feature layer ourselves rather than receiving a pre-made modeling table. [GitHub](https://github.com/WizardNox/End-to-End-SaaS-Analytics-Engineering-Platform?utm_source=chatgpt.com)

### One caveat we should keep in the PRD

RavenStack is **synthetic** and relatively small. We should never claim the model is production-validated across real SaaS companies. The value of the project is that we're building the **architecture and methodology correctly**, including leakage control and time-aware evaluation, on a realistic multi-table SaaS dataset. [wikidocs.net](https://wikidocs.net/341897?utm_source=chatgpt.com)

That is the technical plan I would freeze before implementation.