"""Pydantic v2 Request and Response Schemas for Flurit Inference API.

Defines strict input validation schemas for the 45 canonical features, model prediction
requests, customer risk explanations, and portfolio prioritization quadrants.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.ml.features.schema import FEATURE_NAMES

# Allowed categories from the RavenStack audited dataset
ALLOWED_INDUSTRIES: set[str] = {
    "Healthcare",
    "Finance",
    "Retail",
    "Education",
    "Tech",
    "DevTools",
    "FinTech",
    "E-commerce",
    "Cybersecurity",
    "unknown",
}

ALLOWED_REFERRAL_SOURCES: set[str] = {
    "organic",
    "paid_ad",
    "referral",
    "partner",
    "direct",
    "unknown",
}

ALLOWED_BILLING_FREQUENCIES: set[str] = {
    "monthly",
    "annual",
    "quarterly",
    "unknown",
}


class CanonicalFeaturesPayload(BaseModel):
    """Strict schema enforcing all 45 canonical point-in-time features.

    Rejects extra/unknown fields and requires every documented feature
    to prevent lookahead bias or silent default substitution.
    """

    model_config = ConfigDict(extra="forbid")

    # 1. Account features (6)
    account_age_days: Annotated[int, Field(ge=0, description="Days since account creation")]
    account_industry: Annotated[str, Field(description="Account industry vertical")]
    account_country: Annotated[str, Field(description="Headquarters country code")]
    account_referral_source: Annotated[str, Field(description="Acquisition channel")]
    account_initial_seats: Annotated[int, Field(ge=0, description="Seats at creation")]
    account_is_trial: Annotated[int, Field(ge=0, le=1, description="Binary trial indicator")]

    # 2. Subscription features (10)
    sub_active_count: Annotated[int, Field(ge=0, description="Active subscription count")]
    sub_current_mrr: Annotated[
        float, Field(ge=0.0, description="Current monthly recurring revenue")
    ]
    sub_current_arr: Annotated[float, Field(ge=0.0, description="Current annual recurring revenue")]
    sub_total_active_seats: Annotated[int, Field(ge=0, description="Sum of active seats")]
    sub_tenure_days: Annotated[float, Field(ge=0.0, description="Commercial tenure in days")]
    sub_billing_frequency: Annotated[str, Field(description="Billing frequency")]
    sub_has_auto_renew: Annotated[int, Field(ge=0, le=1, description="Auto-renew active (1/0)")]
    sub_has_active_trial: Annotated[
        int, Field(ge=0, le=1, description="Active trial present (1/0)")
    ]
    sub_historical_upgrade_count: Annotated[int, Field(ge=0, description="Historical upgrades")]
    sub_historical_downgrade_count: Annotated[int, Field(ge=0, description="Historical downgrades")]

    # 3. Usage features (14)
    usage_count_30d: Annotated[int, Field(ge=0, description="Events in past 30 days")]
    usage_count_60d: Annotated[int, Field(ge=0, description="Events in past 60 days")]
    usage_count_90d: Annotated[int, Field(ge=0, description="Events in past 90 days")]
    usage_duration_secs_30d: Annotated[float, Field(ge=0.0, description="Duration in past 30 days")]
    usage_duration_secs_90d: Annotated[float, Field(ge=0.0, description="Duration in past 90 days")]
    usage_error_count_30d: Annotated[int, Field(ge=0, description="Errors in past 30 days")]
    usage_error_rate_30d: Annotated[
        float, Field(ge=0.0, le=1.0, description="Error rate in past 30 days")
    ]
    usage_active_days_30d: Annotated[
        int, Field(ge=0, le=30, description="Active days in past 30 days")
    ]
    usage_active_days_90d: Annotated[
        int, Field(ge=0, le=90, description="Active days in past 90 days")
    ]
    usage_distinct_features_30d: Annotated[
        int, Field(ge=0, description="Distinct features in past 30 days")
    ]
    usage_beta_feature_count_30d: Annotated[
        int, Field(ge=0, description="Beta features in past 30 days")
    ]
    usage_days_since_last: Annotated[
        float, Field(ge=-1.0, description="Days since last activity (-1 if none)")
    ]
    usage_has_history: Annotated[int, Field(ge=0, le=1, description="History exists (1/0)")]
    usage_trend_pct_30d: Annotated[float, Field(description="30d vs 60d usage delta pct")]

    # 4. Support features (11)
    tickets_count_30d: Annotated[int, Field(ge=0, description="Tickets in past 30 days")]
    tickets_count_90d: Annotated[int, Field(ge=0, description="Tickets in past 90 days")]
    tickets_count_lifetime: Annotated[int, Field(ge=0, description="Lifetime tickets")]
    tickets_unresolved_count: Annotated[int, Field(ge=0, description="Open/unresolved tickets")]
    tickets_has_unresolved: Annotated[int, Field(ge=0, le=1, description="Has open tickets (1/0)")]
    tickets_urgent_count_90d: Annotated[
        int, Field(ge=0, description="Urgent tickets in past 90 days")
    ]
    tickets_high_or_urgent_count_90d: Annotated[
        int, Field(ge=0, description="High/urgent tickets in past 90 days")
    ]
    tickets_escalated_count_90d: Annotated[
        int, Field(ge=0, description="Escalated tickets in past 90 days")
    ]
    tickets_avg_satisfaction_90d: Annotated[
        float, Field(ge=-1.0, le=5.0, description="Mean CSAT score (-1 if unrated)")
    ]
    tickets_has_unrated_90d: Annotated[
        int, Field(ge=0, le=1, description="Has unrated tickets (1/0)")
    ]
    tickets_avg_first_response_minutes_90d: Annotated[
        float, Field(ge=-1.0, description="First response mins (-1 if none)")
    ]

    # 5. Churn history features (4)
    churn_prior_count: Annotated[int, Field(ge=0, description="Prior churn events")]
    churn_has_prior: Annotated[int, Field(ge=0, le=1, description="Has prior churn (1/0)")]
    churn_days_since_last: Annotated[
        float, Field(ge=-1.0, description="Days since last churn (-1 if none)")
    ]
    churn_prior_refund_total: Annotated[float, Field(ge=0.0, description="Prior refund total USD")]

    @field_validator("account_industry")
    @classmethod
    def validate_industry(cls, v: str) -> str:
        """Sanitize and validate industry."""
        v_clean = v.strip()
        if not v_clean:
            return "unknown"
        return v_clean

    @field_validator("account_country")
    @classmethod
    def validate_country(cls, v: str) -> str:
        """Sanitize and validate country code."""
        v_clean = v.strip().upper()
        if not v_clean:
            return "UNKNOWN"
        return v_clean

    @field_validator("account_referral_source")
    @classmethod
    def validate_referral_source(cls, v: str) -> str:
        """Sanitize referral source."""
        v_clean = v.strip().lower()
        if not v_clean:
            return "unknown"
        return v_clean

    @field_validator("sub_billing_frequency")
    @classmethod
    def validate_billing_frequency(cls, v: str) -> str:
        """Sanitize billing frequency."""
        v_clean = v.strip().lower()
        if not v_clean:
            return "unknown"
        return v_clean

    def to_canonical_dict(self) -> dict[str, Any]:
        """Return dictionary strictly keyed by canonical FEATURE_NAMES order."""
        data = self.model_dump()
        return {col: data[col] for col in FEATURE_NAMES}


class ScorePredictionRequest(BaseModel):
    """Payload to request model scoring for a customer snapshot."""

    model_config = ConfigDict(extra="forbid")

    account_id: str | None = Field(default=None, description="Optional account identifier")
    snapshot_date: str | None = Field(
        default=None, description="Optional snapshot observation date (YYYY-MM-DD)"
    )
    features: CanonicalFeaturesPayload


class ScorePredictionResponse(BaseModel):
    """Response containing model risk score and metadata."""

    account_id: str | None = Field(default=None)
    snapshot_date: str | None = Field(default=None)
    model_name: str = Field(..., description="Canonical model identity")
    model_version: str = Field(..., description="Version of the active model artifact")
    model_score: float = Field(..., ge=0.0, le=1.0, description="Model churn score in [0.0, 1.0]")
    predicted_risk_label: int = Field(
        ..., description="Binary classification (1 for high risk, 0 otherwise)"
    )
    decision_threshold: float = Field(default=0.5, description="Decision threshold applied")
    risk_tier: Literal["High Risk", "Moderate Risk", "Low Risk"] = Field(
        ..., description="Categorical risk band"
    )
    score_interpretation: str = Field(
        ...,
        description="Notice explaining that score is an uncalibrated ranking, not a calibrated probability",
    )


class FactorAttributionResponse(BaseModel):
    """Attribution item representing a single feature's local contribution."""

    feature_name: str
    feature_value: Any
    attribution: float
    direction: Literal["increases_risk", "decreases_risk"]
    feature_description: str


class CustomerRiskRequest(BaseModel):
    """Payload to request combined churn score, SHAP explanation, and customer segment."""

    model_config = ConfigDict(extra="forbid")

    snapshot_date: str | None = Field(default=None, description="Observation date (YYYY-MM-DD)")
    features: CanonicalFeaturesPayload


class CustomerRiskResponse(BaseModel):
    """Response containing score, local SHAP factors, and segment assignment."""

    account_id: str
    snapshot_date: str
    model_version: str
    model_score: float
    predicted_risk_label: int
    decision_threshold: float
    risk_tier: Literal["High Risk", "Moderate Risk", "Low Risk"]
    segment_id: int
    segment_name: str
    segment_summary: str
    top_risk_increasing_factors: list[FactorAttributionResponse]
    top_risk_decreasing_factors: list[FactorAttributionResponse]
    plain_language_summary: str
    association_disclaimer: str
    score_interpretation: str


class PersistedCustomerSummary(BaseModel):
    """Tenant-scoped customer roster entry backed by durable onboarding artifacts."""

    account_id: str
    account_name: str | None = None
    industry: str
    country: str
    current_mrr: float | None = None
    risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_tier: str | None = None
    segment_name: str | None = None
    assessed_at: str | None = None
    snapshot_date: str | None = None
    score_interpretation: str | None = None


class PersistedCustomerListResponse(BaseModel):
    items: list[PersistedCustomerSummary]
    total: int
    page: int
    page_size: int


class PersistedCustomerDetailResponse(BaseModel):
    account_id: str
    account_name: str | None = None
    industry: str
    country: str
    signup_date: str
    plan_tier: str | None = None
    current_mrr: float | None = None
    risk_score: float | None = None
    risk_tier: str | None = None
    segment_name: str | None = None
    snapshot_date: str | None = None
    assessed_at: str | None = None
    score_interpretation: str | None = None
    explanation_summary: str | None = None
    top_risk_factors: list[dict[str, Any]] = Field(default_factory=list)
    subscriptions: list[dict[str, Any]] = Field(default_factory=list)


class PersistedCustomerHistoryItem(BaseModel):
    prediction_id: int
    snapshot_date: str | None = None
    assessed_at: str
    model_name: str
    model_version: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_tier: str
    segment_name: str | None = None
    score_interpretation: str


class PersistedCustomerHistoryResponse(BaseModel):
    account_id: str
    items: list[PersistedCustomerHistoryItem]


class ScoredCustomerRecord(BaseModel):
    """Individual customer record for portfolio prioritization."""

    model_config = ConfigDict(extra="forbid")

    account_id: str = Field(..., description="Unique customer account identifier")
    sub_current_mrr: float = Field(
        ..., ge=0.0, description="Monthly recurring revenue (contract value)"
    )
    model_score: float = Field(..., ge=0.0, le=1.0, description="Uncalibrated model churn score")


class PrioritizationQuadrantsRequest(BaseModel):
    """Collection of scored customers to group into Value x Risk priority quadrants."""

    model_config = ConfigDict(extra="forbid")

    records: list[ScoredCustomerRecord] = Field(
        ...,
        min_length=1,
        description="List of scored customer accounts to prioritize",
    )
    mrr_high_threshold: float = Field(default=8000.0, description="MRR threshold for high value")
    mrr_mid_threshold: float = Field(default=4000.0, description="MRR threshold for mid value")
    score_high_threshold: float = Field(default=0.50, description="Score threshold for high risk")
    score_mid_threshold: float = Field(
        default=0.35, description="Score threshold for moderate risk"
    )


class PrioritizationQuadrantDetail(BaseModel):
    """Quadrant aggregate metrics."""

    quadrant_name: str
    priority_level: int
    account_count: int
    account_percentage: float
    total_mrr: float
    estimated_revenue_at_risk_mrr: float
    action_recommendation: str


class PrioritizationQuadrantsResponse(BaseModel):
    """Portfolio-level prioritization response."""

    total_accounts: int
    total_mrr: float
    total_estimated_revenue_at_risk_mrr: float
    quadrants: list[PrioritizationQuadrantDetail]
    risk_metric_interpretation: str
