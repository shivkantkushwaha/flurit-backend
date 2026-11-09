# Project Documentation

---

## 1. Project Overview

### What is Flurit

The project aims to identify customers who may be likely to leave a service by studying their customer, subscription, product usage, and support data.

Instead of only predicting whether a customer will churn, Flurit will also study the factors related to the prediction and help identify which customers may need more attention.

### Main flow

**Customer Data → Data Preparation → Feature Engineering → ML Models → Churn Prediction → Explanation → Customer Risk Analysis**

---

# 2. Problem We Are Solving

Customer churn is a common business problem. A company may have many customers, but it is difficult to manually identify which customers are becoming likely to leave.

Customers also usually do not directly tell the company that they are planning to leave.

Flurit therefore uses existing customer data and historical outcomes to learn patterns associated with churn.

The main questions are:

**Who is likely to churn?**

**What factors are associated with the risk?**

**Which customers should receive more attention?**

---

# 3. Dataset Selected

## RavenStack SaaS Subscription & Churn Analytics Dataset

### Why we selected it

We selected the **RavenStack SaaS Subscription & Churn Analytics Dataset** because its structure is closer to how customer data can exist in an actual SaaS business.

Instead of putting everything into one flat CSV file, the dataset is divided into related tables:

- `accounts`
- `subscriptions`
- `feature_usage`
- `support_tickets`
- `churn_events`

The tables are connected through customer/account and subscription identifiers. The dataset contains approximately **500 accounts, 5,000 subscriptions, 25,000 feature-usage records, 2,000 support tickets, and 600 churn events** in the commonly published version. [GitHub](https://github.com/PurvaShah/saas_subscription_churn_analytics/blob/main/README.md?utm_source=chatgpt.com)

This gives us different types of information that can be combined for analysis:

**Customer profile + Subscription + Product usage + Support activity + Churn outcome**

That makes the dataset suitable for the type of project we want to build.

### Important note

RavenStack is a **synthetic dataset**, not real company data. It is designed to represent a realistic SaaS environment and is intended for learning, analytics, and portfolio projects. [GitHub](https://github.com/PurvaShah/saas_subscription_churn_analytics/blob/main/README.md?utm_source=chatgpt.com)

We will clearly state this in our project rather than presenting it as real customer data.

---

# 4. Why We Did Not Select Other Datasets

## KKBox Churn Prediction Dataset

KKBox was seriously considered because it is based on a real music-streaming business and contains customer, transaction, and daily user-behaviour data. Its churn definition is also based on whether a user renews within 30 days after membership expiry. [Kaggle](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge?utm_source=chatgpt.com)

However, it is strongly focused on a music-streaming use case. The behavioural data is mainly listening activity, while Flurit is intended to demonstrate a broader **SaaS customer-retention workflow** involving subscriptions, product usage, support interactions, and customer value.

KKBox is also much larger and more complex than what we need for this project.

**Decision:** Not selected for the main Flurit implementation.

---

## Customer Churn Prediction Dataset — 1M

We also considered a 1-million-row churn dataset.

Its main advantage is scale: it contains around **1 million customer records and more than 30 columns**, including variables related to satisfaction, complaints, service calls, payments, usage and churn. [RStudio Pubs](https://rstudio-pubs-static.s3.amazonaws.com/1438571_3863e061046746a4affdfffe491d490f.html?utm_source=chatgpt.com)

However, it is a **single large synthetic dataset**. The large row count is useful for demonstrating data processing at scale, but it does not give us the multi-table customer and business structure that we want for Flurit.

For this project, **data structure and business relevance are more important than simply having a very large number of rows.**

**Decision:** Not selected as the primary dataset.

---

# 5. Why RavenStack Fits Flurit

The dataset allows us to work with several parts of the customer relationship:

| Data | What Flurit can study |
| --- | --- |
| Accounts | Customer profile and business information |
| Subscriptions | Plans, MRR, billing and renewals |
| Feature Usage | Product engagement and usage patterns |
| Support Tickets | Support activity, resolution and satisfaction |
| Churn Events | Churn date and related information |

This allows us to build features such as:

`MRR`

`Tenure`

`Recent Usage`

`Usage Change`

`Support Ticket Count`

`Satisfaction Score`

`Auto-Renewal Status`

`Upgrade/Downgrade Activity`

These features can then be used for churn prediction and customer analysis. [GitHub](https://github.com/PurvaShah/saas_subscription_churn_analytics/blob/main/README.md?utm_source=chatgpt.com)

---

# 6. Planned ML Approach

We will treat churn prediction as a **classification problem**.

### Models to study

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

We will compare the models using appropriate classification metrics, including:

**Accuracy · Precision · Recall · F1-score · ROC-AUC**

We will also use a confusion matrix to understand the prediction results.

---

# 7. Explainability

The project will not stop at:

> **“This customer has an 82% churn probability.”**
> 

We also want to understand **what factors contributed to the prediction**.

For selected models, we plan to use **SHAP-based explanations** or suitable feature-importance methods to show the main factors behind a customer's risk.

Example:

> **Churn Risk: High — 82%**
> 
> 
> Main factors: declining usage, low satisfaction, recent support activity.
> 

This makes the prediction easier to understand.

---

# 8. Customer Value & Risk

A customer with high churn probability is not always equally important to the business.

For example:

**Customer A:** 80% churn risk, MRR = $100

**Customer B:** 65% churn risk, MRR = $2,000

Both are at risk, but the potential business impact is different.

Therefore, Flurit will also explore **customer value together with churn risk**.

This can lead to a simple:

**Customer Value × Churn Risk** analysis.

---

# 9. Expected Output

The final project is expected to provide:

### Customer Risk

Churn probability and risk level.

### Risk Explanation

Important factors associated with the prediction.

### Customer Analysis

Customer value, usage, subscription and support information.

### Business View

Summary of high-risk customers and potential revenue at risk.

### Dashboard

A simple interface for viewing these results.

---

# 10. Project Scope

### Core scope

**Data preparation**

**Feature engineering**

**Churn prediction**

**Model comparison**

**Model evaluation**

**Explainability**

**Customer risk analysis**

**Dashboard**

### Possible extensions

Customer segmentation, improved customer-health analysis, and retention suggestions can be explored after the core project is completed.

---

# 11. Important Project Principle

> **We are building Flurit as a proper ML project, not only as a model-training exercise.**
> 

The objective is to connect:

**Data → Machine Learning → Explanation → Business Analysis**

while keeping the project technically understandable and appropriate for the course.

---