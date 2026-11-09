"""Point-in-time Feature Schema Definitions.

Defines the canonical metadata, data types, lookback windows, and null-handling
rules for all features generated for Flurit customer snapshots.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class FeatureGroup(StrEnum):
    """Categorical domain grouping for customer features."""

    ACCOUNT = "account"
    SUBSCRIPTION = "subscription"
    USAGE = "usage"
    SUPPORT = "support"
    CHURN_HISTORY = "churn_history"


@dataclass(frozen=True)
class FeatureDefinition:
    """Metadata specification for an individual feature."""

    name: str
    group: FeatureGroup
    dtype: str
    lookback_days: int | None
    description: str
    missing_value_rule: str


# Canonical Feature Specifications
FEATURE_DEFINITIONS: Final[list[FeatureDefinition]] = [
    # --------------------------------------------------------------------------
    # Account Features
    # --------------------------------------------------------------------------
    FeatureDefinition(
        name="account_age_days",
        group=FeatureGroup.ACCOUNT,
        dtype="int64",
        lookback_days=None,
        description="Days between account creation (signup_date) and snapshot date.",
        missing_value_rule="Zero if signup_date equals snapshot_date; non-negative.",
    ),
    FeatureDefinition(
        name="account_industry",
        group=FeatureGroup.ACCOUNT,
        dtype="category",
        lookback_days=None,
        description="Account industry domain (e.g. DevTools, FinTech, Healthcare).",
        missing_value_rule="Categorical; 'unknown' if missing.",
    ),
    FeatureDefinition(
        name="account_country",
        group=FeatureGroup.ACCOUNT,
        dtype="category",
        lookback_days=None,
        description="Account headquarters country code.",
        missing_value_rule="Categorical; 'unknown' if missing.",
    ),
    FeatureDefinition(
        name="account_referral_source",
        group=FeatureGroup.ACCOUNT,
        dtype="category",
        lookback_days=None,
        description="Customer acquisition channel (e.g. organic, partner, paid_ad).",
        missing_value_rule="Categorical; 'unknown' if missing.",
    ),
    FeatureDefinition(
        name="account_initial_seats",
        group=FeatureGroup.ACCOUNT,
        dtype="int64",
        lookback_days=None,
        description="Licensed seat count declared at account creation.",
        missing_value_rule="Defaults to 1 if missing.",
    ),
    FeatureDefinition(
        name="account_is_trial",
        group=FeatureGroup.ACCOUNT,
        dtype="int64",
        lookback_days=None,
        description="Binary indicator (1/0) whether account was initialized on trial.",
        missing_value_rule="0 if missing.",
    ),
    # --------------------------------------------------------------------------
    # Subscription Features
    # --------------------------------------------------------------------------
    FeatureDefinition(
        name="sub_active_count",
        group=FeatureGroup.SUBSCRIPTION,
        dtype="int64",
        lookback_days=None,
        description="Count of subscriptions actively un-terminated at snapshot date.",
        missing_value_rule="0 if no active subscriptions.",
    ),
    FeatureDefinition(
        name="sub_current_mrr",
        group=FeatureGroup.SUBSCRIPTION,
        dtype="float64",
        lookback_days=None,
        description="Total Monthly Recurring Revenue across active subscriptions.",
        missing_value_rule="0.0 if no active subscriptions.",
    ),
    FeatureDefinition(
        name="sub_current_arr",
        group=FeatureGroup.SUBSCRIPTION,
        dtype="float64",
        lookback_days=None,
        description="Total Annual Recurring Revenue across active subscriptions.",
        missing_value_rule="0.0 if no active subscriptions.",
    ),
    FeatureDefinition(
        name="sub_total_active_seats",
        group=FeatureGroup.SUBSCRIPTION,
        dtype="int64",
        lookback_days=None,
        description="Total licensed seats across active subscriptions.",
        missing_value_rule="0 if no active subscriptions.",
    ),
    FeatureDefinition(
        name="sub_tenure_days",
        group=FeatureGroup.SUBSCRIPTION,
        dtype="int64",
        lookback_days=None,
        description="Days from customer's earliest subscription start date to snapshot date.",
        missing_value_rule="0 if no historical subscription.",
    ),
    FeatureDefinition(
        name="sub_billing_frequency",
        group=FeatureGroup.SUBSCRIPTION,
        dtype="category",
        lookback_days=None,
        description="Primary billing interval of active subscriptions ('monthly', 'annual', 'mixed', 'none').",
        missing_value_rule="'none' if no active subscriptions.",
    ),
    FeatureDefinition(
        name="sub_has_auto_renew",
        group=FeatureGroup.SUBSCRIPTION,
        dtype="int64",
        lookback_days=None,
        description="Binary indicator if at least one active subscription has auto-renew enabled.",
        missing_value_rule="0 if no active subscriptions.",
    ),
    FeatureDefinition(
        name="sub_has_active_trial",
        group=FeatureGroup.SUBSCRIPTION,
        dtype="int64",
        lookback_days=None,
        description="Binary indicator if at least one active subscription is in trial mode.",
        missing_value_rule="0 if no active subscriptions.",
    ),
    FeatureDefinition(
        name="sub_historical_upgrade_count",
        group=FeatureGroup.SUBSCRIPTION,
        dtype="int64",
        lookback_days=None,
        description="Historical count of subscriptions started on or before snapshot with upgrade_flag=True.",
        missing_value_rule="0 if none.",
    ),
    FeatureDefinition(
        name="sub_historical_downgrade_count",
        group=FeatureGroup.SUBSCRIPTION,
        dtype="int64",
        lookback_days=None,
        description="Historical count of subscriptions started on or before snapshot with downgrade_flag=True.",
        missing_value_rule="0 if none.",
    ),
    # --------------------------------------------------------------------------
    # Feature Usage Features
    # --------------------------------------------------------------------------
    FeatureDefinition(
        name="usage_count_30d",
        group=FeatureGroup.USAGE,
        dtype="int64",
        lookback_days=30,
        description="Total product action count in the past 30 days (T_obs - 30d, T_obs].",
        missing_value_rule="0 if no usage in window.",
    ),
    FeatureDefinition(
        name="usage_count_60d",
        group=FeatureGroup.USAGE,
        dtype="int64",
        lookback_days=60,
        description="Total product action count in the past 60 days (T_obs - 60d, T_obs].",
        missing_value_rule="0 if no usage in window.",
    ),
    FeatureDefinition(
        name="usage_count_90d",
        group=FeatureGroup.USAGE,
        dtype="int64",
        lookback_days=90,
        description="Total product action count in the past 90 days (T_obs - 90d, T_obs].",
        missing_value_rule="0 if no usage in window.",
    ),
    FeatureDefinition(
        name="usage_duration_secs_30d",
        group=FeatureGroup.USAGE,
        dtype="float64",
        lookback_days=30,
        description="Total active product engagement duration in seconds over past 30 days.",
        missing_value_rule="0.0 if no usage in window.",
    ),
    FeatureDefinition(
        name="usage_duration_secs_90d",
        group=FeatureGroup.USAGE,
        dtype="float64",
        lookback_days=90,
        description="Total active product engagement duration in seconds over past 90 days.",
        missing_value_rule="0.0 if no usage in window.",
    ),
    FeatureDefinition(
        name="usage_error_count_30d",
        group=FeatureGroup.USAGE,
        dtype="int64",
        lookback_days=30,
        description="Total feature execution error count in past 30 days.",
        missing_value_rule="0 if no errors.",
    ),
    FeatureDefinition(
        name="usage_error_rate_30d",
        group=FeatureGroup.USAGE,
        dtype="float64",
        lookback_days=30,
        description="Ratio of feature errors to total usage actions in past 30 days.",
        missing_value_rule="0.0 if usage_count_30d is 0.",
    ),
    FeatureDefinition(
        name="usage_active_days_30d",
        group=FeatureGroup.USAGE,
        dtype="int64",
        lookback_days=30,
        description="Number of distinct calendar days with feature activity in past 30 days (0-30).",
        missing_value_rule="0 if no usage.",
    ),
    FeatureDefinition(
        name="usage_active_days_90d",
        group=FeatureGroup.USAGE,
        dtype="int64",
        lookback_days=90,
        description="Number of distinct calendar days with feature activity in past 90 days (0-90).",
        missing_value_rule="0 if no usage.",
    ),
    FeatureDefinition(
        name="usage_distinct_features_30d",
        group=FeatureGroup.USAGE,
        dtype="int64",
        lookback_days=30,
        description="Count of distinct feature modules utilized in past 30 days.",
        missing_value_rule="0 if no usage.",
    ),
    FeatureDefinition(
        name="usage_beta_feature_count_30d",
        group=FeatureGroup.USAGE,
        dtype="int64",
        lookback_days=30,
        description="Total actions on beta features (is_beta_feature=True) in past 30 days.",
        missing_value_rule="0 if no beta usage.",
    ),
    FeatureDefinition(
        name="usage_days_since_last",
        group=FeatureGroup.USAGE,
        dtype="float64",
        lookback_days=None,
        description="Days since latest historical feature usage date on or before snapshot.",
        missing_value_rule="-1.0 sentinel if customer has never logged usage (paired with usage_has_history=0).",
    ),
    FeatureDefinition(
        name="usage_has_history",
        group=FeatureGroup.USAGE,
        dtype="int64",
        lookback_days=None,
        description="Binary indicator (1/0) whether customer has any recorded usage history on or before snapshot.",
        missing_value_rule="0 if no usage ever recorded.",
    ),
    FeatureDefinition(
        name="usage_trend_pct_30d",
        group=FeatureGroup.USAGE,
        dtype="float64",
        lookback_days=60,
        description="Relative velocity change: (usage_30d - usage_prev_30d) / max(usage_prev_30d, 1.0).",
        missing_value_rule="0.0 if both windows have zero usage; bounded in [-1.0, 5.0].",
    ),
    # --------------------------------------------------------------------------
    # Support Tickets Features
    # --------------------------------------------------------------------------
    FeatureDefinition(
        name="tickets_count_30d",
        group=FeatureGroup.SUPPORT,
        dtype="int64",
        lookback_days=30,
        description="Support tickets submitted in the past 30 days (T_obs - 30d, T_obs].",
        missing_value_rule="0 if no tickets.",
    ),
    FeatureDefinition(
        name="tickets_count_90d",
        group=FeatureGroup.SUPPORT,
        dtype="int64",
        lookback_days=90,
        description="Support tickets submitted in the past 90 days (T_obs - 90d, T_obs].",
        missing_value_rule="0 if no tickets.",
    ),
    FeatureDefinition(
        name="tickets_count_lifetime",
        group=FeatureGroup.SUPPORT,
        dtype="int64",
        lookback_days=None,
        description="Total lifetime support tickets submitted on or before snapshot date.",
        missing_value_rule="0 if no tickets.",
    ),
    FeatureDefinition(
        name="tickets_unresolved_count",
        group=FeatureGroup.SUPPORT,
        dtype="int64",
        lookback_days=None,
        description="Tickets submitted on or before snapshot that remain open/unresolved (closed_at is null or > T_obs).",
        missing_value_rule="0 if all closed.",
    ),
    FeatureDefinition(
        name="tickets_has_unresolved",
        group=FeatureGroup.SUPPORT,
        dtype="int64",
        lookback_days=None,
        description="Binary indicator (1/0) whether account has at least one currently unresolved ticket at snapshot.",
        missing_value_rule="0 if no unresolved tickets.",
    ),
    FeatureDefinition(
        name="tickets_urgent_count_90d",
        group=FeatureGroup.SUPPORT,
        dtype="int64",
        lookback_days=90,
        description="Count of tickets with priority='urgent' submitted in past 90 days.",
        missing_value_rule="0 if none.",
    ),
    FeatureDefinition(
        name="tickets_high_or_urgent_count_90d",
        group=FeatureGroup.SUPPORT,
        dtype="int64",
        lookback_days=90,
        description="Count of tickets with priority in ('high', 'urgent') submitted in past 90 days.",
        missing_value_rule="0 if none.",
    ),
    FeatureDefinition(
        name="tickets_escalated_count_90d",
        group=FeatureGroup.SUPPORT,
        dtype="int64",
        lookback_days=90,
        description="Count of tickets with escalation_flag=True submitted in past 90 days.",
        missing_value_rule="0 if none.",
    ),
    FeatureDefinition(
        name="tickets_avg_satisfaction_90d",
        group=FeatureGroup.SUPPORT,
        dtype="float64",
        lookback_days=90,
        description="Mean satisfaction score for rated tickets in past 90 days (1.0 to 5.0).",
        missing_value_rule="-1.0 sentinel if no rated tickets (paired with tickets_has_unrated_90d indicator).",
    ),
    FeatureDefinition(
        name="tickets_has_unrated_90d",
        group=FeatureGroup.SUPPORT,
        dtype="int64",
        lookback_days=90,
        description="Binary indicator (1/0) if customer had tickets in past 90 days where satisfaction was unrated.",
        missing_value_rule="0 if no unrated tickets or no tickets.",
    ),
    FeatureDefinition(
        name="tickets_avg_first_response_minutes_90d",
        group=FeatureGroup.SUPPORT,
        dtype="float64",
        lookback_days=90,
        description="Mean first response time in minutes for tickets submitted in past 90 days.",
        missing_value_rule="-1.0 sentinel if no tickets submitted in window.",
    ),
    # --------------------------------------------------------------------------
    # Churn History Features
    # --------------------------------------------------------------------------
    FeatureDefinition(
        name="churn_prior_count",
        group=FeatureGroup.CHURN_HISTORY,
        dtype="int64",
        lookback_days=None,
        description="Count of historical churn events logged on or before snapshot date.",
        missing_value_rule="0 if customer has never previously churned.",
    ),
    FeatureDefinition(
        name="churn_has_prior",
        group=FeatureGroup.CHURN_HISTORY,
        dtype="int64",
        lookback_days=None,
        description="Binary indicator (1/0) if customer experienced at least one prior churn event.",
        missing_value_rule="0 if no prior churn.",
    ),
    FeatureDefinition(
        name="churn_days_since_last",
        group=FeatureGroup.CHURN_HISTORY,
        dtype="float64",
        lookback_days=None,
        description="Days from most recent historical churn event (<= T_obs) to snapshot date.",
        missing_value_rule="-1.0 sentinel if no prior churn.",
    ),
    FeatureDefinition(
        name="churn_prior_refund_total",
        group=FeatureGroup.CHURN_HISTORY,
        dtype="float64",
        lookback_days=None,
        description="Cumulative refund dollars granted during historical churn events on or before snapshot.",
        missing_value_rule="0.0 if no prior churns or no refunds.",
    ),
]

FEATURE_NAMES: Final[list[str]] = [f.name for f in FEATURE_DEFINITIONS]
FEATURE_NAME_TO_DEF: Final[dict[str, FeatureDefinition]] = {f.name: f for f in FEATURE_DEFINITIONS}

# Strictly Excluded Fields (Prohibited to prevent target or post-hoc leakage)
STRICTLY_EXCLUDED_FIELDS: Final[set[str]] = {
    "churn_flag",  # Static accounts / subscriptions retrospective column
    "accounts.churn_flag",
    "subscriptions.churn_flag",
    "target_churn_60d",  # Prediction target
    "churn_date",  # Post-hoc future churn attributes
    "reason_code",
    "feedback_text",
    "refund_amount_usd",  # Prohibited when from future churn events
}
