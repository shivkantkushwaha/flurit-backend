# Dataset

## 1. Selected Dataset

### RavenStack — SaaS Subscription & Churn Analytics Dataset

For **Flurit**, we have selected the **SaaS Subscription & Churn Analytics Dataset** published on Kaggle by Rivalytics.

The main reason for selecting this dataset is its structure. Instead of being a single customer-level CSV, it represents a SaaS business through **five related tables** covering accounts, subscriptions, product usage, support activity, and churn events. This gives us enough information to build Flurit around customer behaviour and business context rather than only demographic information. [GitHub](https://github.com/Quratulain-qurat97/SaaS-Subscription-Churn-Analytics?utm_source=chatgpt.com)

**Dataset source:** [Kaggle — SaaS Subscription & Churn Analytics Dataset](https://www.kaggle.com/datasets/rivalytics/saas-subscription-and-churn-analytics-dataset?utm_source=chatgpt.com)

> **Note:** RavenStack is a **synthetic/simulated SaaS dataset**. It is designed to represent a realistic SaaS environment; it is not actual production data from a real company. [Free AI Tools](https://free2aitools.com/dataset/rivalytics/saas-subscription-and-churn-analytics-dataset?utm_source=chatgpt.com)
> 

---

## 2. Why We Selected RavenStack

Flurit is intended to answer more than:

> **“Will this customer churn?”**
> 

We also want to study:

> **“Why may this customer churn?”**
> 
> 
> **“What is the value of this customer?”**
> 
> **“Which customers should receive more attention?”**
> 

The RavenStack dataset supports this direction because it contains different types of customer information that can be connected together.

It allows us to combine:

**Customer profile + Subscription + Product usage + Support activity + Churn outcome**

This makes it suitable for feature engineering, churn prediction, explainability, and customer-level analysis. [GitHub](https://github.com/IrmandoKoyo/ravenstack-ai-dashboard)

---

## 3. Dataset Structure

The dataset contains five related tables:

### `accounts`

Contains basic information about the customer/account.

Examples:

- `account_id`
- `company_name`
- `industry`
- `country`
- `referral_source`
- `signup_date`

### `subscriptions`

Contains information about the customer's subscriptions.

Examples:

- `subscription_id`
- `account_id`
- `plan_tier`
- `mrr_amount`
- `start_date`
- `end_date`
- `status`

### `feature_usage`

Contains product usage information over time.

Examples:

- `usage_id`
- `subscription_id`
- `feature_name`
- `usage_count`
- `usage_date`
- `error_count`
- `usage_time_sec`
- `is_beta`

### `support_tickets`

Contains customer support activity.

Examples:

- `ticket_id`
- `account_id`
- `priority`
- `resolution_hours`
- `first_response`
- `satisfaction`
- `is_escalated`

### `churn_events`

Contains information about customers who churned.

Examples:

- `churn_id`
- `account_id`
- `churn_date`
- `reason_code`
- `feedback_text`

The published schema confirms these five tables and their account/subscription relationships. [GitHub](https://github.com/IrmandoKoyo/ravenstack-ai-dashboard)

---

## 4. Tables & Relationships

The basic relationship between the tables is:

```
                         accounts
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
      subscriptions   support_tickets   churn_events
             │
             ▼
       feature_usage
```

The important keys are:

```
accounts.account_id
        ↓
subscriptions.account_id
        ↓
feature_usage.subscription_id
```

and:

```
accounts.account_id
        ↓
support_tickets.account_id

accounts.account_id
        ↓
churn_events.account_id
```

This means Flurit can bring information from different parts of the customer lifecycle into one customer-level view. [GitHub](https://github.com/IrmandoKoyo/ravenstack-ai-dashboard)

---

## 5. Why the Multi-Table Structure Matters

A real business does not normally keep every customer detail in one giant table.

Different parts of the business generate different records:

```
Customer
   ↓
Subscription
   ↓
Product usage
   ↓
Support interactions
   ↓
Churn / Retention outcome
```

Flurit will combine these records during data preparation and create a final dataset suitable for machine learning.

For example, we may create features such as:

- Current MRR
- Customer tenure
- Recent feature usage
- Change in usage
- Number of support tickets
- Support satisfaction
- Number of escalated tickets
- Subscription status
- Churn history

The exact features will be decided after we inspect the raw data and understand which fields are suitable for prediction.

---

## 6. Dataset Scale

The commonly published version contains approximately:

| Table | Approx. Records |
| --- | --- |
| Accounts | 500 |
| Subscriptions | 4,853–5,000 |
| Feature Usage | 25,000+ |
| Support Tickets | 2,000 |
| Churn Events | 600 |

Different public analyses report slightly different row counts depending on the version or processing used, so we will verify the exact counts from the dataset files when we begin implementation. [GitHub](https://github.com/Quratulain-qurat97/SaaS-Subscription-Churn-Analytics?utm_source=chatgpt.com)

The dataset covers the period **January 2023 to December 2024** in a published analysis. [GitHub](https://github.com/Quratulain-qurat97/SaaS-Subscription-Churn-Analytics?utm_source=chatgpt.com)

---

## 7. Important Data Considerations

Before model training, we will inspect the dataset for:

**Missing values**

Identify fields with missing information and decide how they should be handled.

**Duplicate records**

Check for duplicate or repeated records where relevant.

**Data types and dates**

Convert dates and numerical fields into suitable formats.

**Relationships**

Verify that the joins between accounts, subscriptions, usage, support and churn events are correct.

**Data leakage**

Make sure that information only available after churn is not accidentally used to predict that same churn event.

**Feature quality**

Keep features that can realistically be available before the prediction point.

This last point is especially important because Flurit is intended to predict churn **before it happens**, not explain it only after the customer has already left.

---

# 8. Rejected Alternatives

## KKBox Churn Prediction Dataset

KKBox was considered because it is based on real subscription and behavioural data from a music-streaming business and contains large amounts of transaction and user-activity data. [GitHub](https://github.com/Quratulain-qurat97/SaaS-Subscription-Churn-Analytics?utm_source=chatgpt.com)

However, it is mainly focused on music-streaming behaviour. Its data is much more centred around listening activity, transactions and renewals.

For Flurit, we wanted a dataset that gives us a broader SaaS business view including:

**subscriptions + product usage + support + customer value + churn**

KKBox is also considerably larger and more complex, which would add significant data-processing work without being necessary for our current project scope.

**Decision: Not selected for the main Flurit implementation.**

---

## 1-Million-Row Customer Churn Dataset

We also considered a large 1-million-row churn dataset.

Its main advantage is scale. However, a large row count alone does not make a dataset more suitable for Flurit.

Our priority is the **quality and structure of the customer data**, not simply the number of rows.

The selected RavenStack dataset gives us multiple related business data sources that can be joined and analysed together, which better matches the type of customer-retention product we want to build.

**Decision: Not selected as the primary dataset.**

---

# 9. Final Decision

### Selected

**RavenStack — SaaS Subscription & Churn Analytics Dataset**

### Why

It gives Flurit:

**Multiple related tables**

**Subscription information**

**Customer value / MRR data**

**Product usage data**

**Support activity**

**Churn outcomes**

**Time-based information**

This makes it suitable for our planned workflow:

```
Raw SaaS Data
      ↓
Data Preparation
      ↓
Feature Engineering
      ↓
Churn Prediction
      ↓
Model Explanation
      ↓
Customer Risk & Value Analysis
      ↓
Flurit Dashboard
```

### Project principle

> **We are selecting the dataset based on how well it supports the problem we want to solve, not simply based on dataset size.**
> 

This will be the main dataset used for Flurit unless a significant data-quality issue is discovered during the initial inspection.