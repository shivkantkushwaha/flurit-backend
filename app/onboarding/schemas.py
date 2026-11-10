"""Pydantic Schemas for Dataset Onboarding, Profiling, Mapping, and Readiness.

Defines the request, response, and metadata models for secure CSV ingestion,
statistical dataset profiling, concept-to-column mappings, and readiness
evaluations against the canonical feature schema and frozen model.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class InferredDataType(StrEnum):
    """Data types inferred by the profiler."""

    INTEGER = "integer"
    FLOAT = "float"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    CATEGORY = "category"
    TEXT = "text"
    UNKNOWN = "unknown"


class ConceptType(StrEnum):
    """Flurit core entity concepts."""

    ACCOUNTS = "accounts"
    SUBSCRIPTIONS = "subscriptions"
    FEATURE_USAGE = "feature_usage"
    SUPPORT_TICKETS = "support_tickets"
    CHURN_EVENTS = "churn_events"


class ReadinessStatus(StrEnum):
    """Readiness status for target construction or model scoring."""

    READY = "READY"
    PARTIALLY_READY = "PARTIALLY_READY"
    UNSUPPORTED = "UNSUPPORTED"


class IssueSeverity(StrEnum):
    """Severity level of validation issues."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class DateRange(BaseModel):
    """Date span information for datetime columns or datasets."""

    min_date: str | None = Field(
        default=None, description="Earliest recorded ISO timestamp or date."
    )
    max_date: str | None = Field(default=None, description="Latest recorded ISO timestamp or date.")
    span_days: int | None = Field(
        default=None, description="Total days elapsed between min and max date."
    )


class ColumnProfile(BaseModel):
    """Structural and statistical profile of an individual dataset column."""

    column_name: str = Field(..., description="Column name in the uploaded dataset.")
    inferred_type: InferredDataType = Field(..., description="Inferred data type.")
    total_count: int = Field(..., description="Total row count.")
    null_count: int = Field(..., description="Count of null or missing values.")
    null_percentage: float = Field(..., description="Percentage of missing values (0.0 to 100.0).")
    unique_count: int = Field(..., description="Count of distinct values.")
    unique_ratio: float = Field(..., description="Ratio of unique values to non-null rows.")
    is_candidate_id: bool = Field(
        ..., description="Flag indicating column is likely a primary/foreign key."
    )
    min_value: str | None = Field(
        default=None, description="Minimum scalar value formatted as string."
    )
    max_value: str | None = Field(
        default=None, description="Maximum scalar value formatted as string."
    )
    sample_categories: list[str] | None = Field(
        default=None,
        description="Redacted sample unique values for low-cardinality categorical fields.",
    )
    date_range: DateRange | None = Field(
        default=None, description="Temporal range if column is datetime."
    )


class DatasetProfile(BaseModel):
    """Complete structural and statistical profile of an uploaded CSV dataset."""

    file_name: str = Field(..., description="Sanitized filename of the uploaded dataset.")
    file_size_bytes: int = Field(..., description="Size of uploaded file in bytes.")
    row_count: int = Field(..., description="Total row count.")
    column_count: int = Field(..., description="Total column count.")
    duplicate_row_count: int = Field(..., description="Count of duplicate rows.")
    candidate_id_columns: list[str] = Field(
        default_factory=list,
        description="List of column names identified as candidate identifiers.",
    )
    candidate_target_columns: list[str] = Field(
        default_factory=list,
        description="List of column names identified as candidate churn targets or status indicators.",
    )
    columns: list[ColumnProfile] = Field(
        default_factory=list,
        description="Column-level profiling metrics.",
    )
    date_range: DateRange | None = Field(
        default=None,
        description="Global date range across all datetime columns in this dataset.",
    )


class DatasetProfileResponse(BaseModel):
    """API response model for dataset upload and profiling."""

    success: bool = True
    message: str = Field(..., description="Summary status message.")
    datasets: list[DatasetProfile] = Field(
        default_factory=list,
        description="Profile results for uploaded dataset files.",
    )


# ------------------------------------------------------------------------------
# Concept Mapping Schemas
# ------------------------------------------------------------------------------


class AccountConceptMapping(BaseModel):
    """Column mappings for the Account core concept."""

    account_id: str = Field(
        ..., description="Column containing unique account/customer ID (Required)."
    )
    signup_date: str = Field(
        ..., description="Column containing account creation timestamp (Required)."
    )
    industry: str | None = Field(
        default=None, description="Column containing account industry vertical."
    )
    country: str | None = Field(default=None, description="Column containing account country code.")
    referral_source: str | None = Field(
        default=None, description="Column containing customer acquisition channel."
    )
    initial_seats: str | None = Field(
        default=None, description="Column containing initial licensed seat count."
    )
    is_trial: str | None = Field(
        default=None, description="Column containing trial account indicator."
    )


class SubscriptionConceptMapping(BaseModel):
    """Column mappings for the Subscription core concept."""

    subscription_id: str = Field(
        ..., description="Column containing unique subscription ID (Required)."
    )
    account_id: str = Field(..., description="Column containing account ID foreign key (Required).")
    start_date: str = Field(
        ..., description="Column containing subscription start timestamp (Required)."
    )
    end_date: str | None = Field(
        default=None, description="Column containing subscription termination timestamp."
    )
    mrr: str | None = Field(
        default=None, description="Column containing monthly recurring revenue."
    )
    arr: str | None = Field(default=None, description="Column containing annual recurring revenue.")
    seats: str | None = Field(default=None, description="Column containing active seat count.")
    billing_frequency: str | None = Field(
        default=None, description="Column containing billing frequency."
    )
    auto_renew: str | None = Field(
        default=None, description="Column containing auto-renew indicator."
    )
    is_trial: str | None = Field(
        default=None, description="Column containing active trial indicator."
    )
    status: str | None = Field(default=None, description="Column containing subscription status.")


class FeatureUsageConceptMapping(BaseModel):
    """Column mappings for the Feature Usage core concept."""

    event_time: str = Field(..., description="Column containing usage event timestamp (Required).")
    account_id: str | None = Field(
        default=None,
        description="Column containing account ID foreign key (Required if subscription_id not provided).",
    )
    subscription_id: str | None = Field(
        default=None,
        description="Column containing subscription ID foreign key.",
    )
    usage_id: str | None = Field(default=None, description="Column containing usage event ID.")
    usage_count: str | None = Field(default=None, description="Column containing the number of usage events represented by each row.")
    feature_name: str | None = Field(
        default=None, description="Column containing feature or action name."
    )
    duration_secs: str | None = Field(
        default=None, description="Column containing session duration in seconds."
    )
    is_error: str | None = Field(default=None, description="Column containing error indicator.")
    is_beta: str | None = Field(
        default=None, description="Column containing beta feature indicator."
    )


class SupportTicketsConceptMapping(BaseModel):
    """Column mappings for the Support Tickets core concept."""

    ticket_id: str = Field(..., description="Column containing unique ticket ID (Required).")
    account_id: str = Field(..., description="Column containing account ID foreign key (Required).")
    created_at: str = Field(
        ..., description="Column containing ticket creation timestamp (Required)."
    )
    closed_at: str | None = Field(
        default=None, description="Column containing ticket resolution timestamp."
    )
    status: str | None = Field(
        default=None, description="Column containing ticket resolution status."
    )
    priority: str | None = Field(
        default=None, description="Column containing ticket priority level."
    )
    satisfaction_score: str | None = Field(
        default=None, description="Column containing customer CSAT rating."
    )


class ChurnEventsConceptMapping(BaseModel):
    """Column mappings for the Churn Events core concept."""

    account_id: str = Field(..., description="Column containing account ID foreign key (Required).")
    churn_date: str = Field(
        ..., description="Column containing recorded churn event timestamp (Required)."
    )
    churn_event_id: str | None = Field(
        default=None, description="Column containing unique churn event ID."
    )
    churn_type: str | None = Field(
        default=None, description="Column containing churn classification."
    )
    churn_reason: str | None = Field(default=None, description="Column containing churn reason.")


class SchemaMappingRequest(BaseModel):
    """Request model submitting concept column mappings for readiness validation."""

    accounts: AccountConceptMapping | None = Field(
        default=None,
        description="Column mappings for Account concept.",
    )
    subscriptions: SubscriptionConceptMapping | None = Field(
        default=None,
        description="Column mappings for Subscription concept.",
    )
    feature_usage: FeatureUsageConceptMapping | None = Field(
        default=None,
        description="Column mappings for Feature Usage concept.",
    )
    support_tickets: SupportTicketsConceptMapping | None = Field(
        default=None,
        description="Column mappings for Support Tickets concept.",
    )
    churn_events: ChurnEventsConceptMapping | None = Field(
        default=None,
        description="Column mappings for Churn Events concept.",
    )
    dataset_profiles: list[DatasetProfile] | None = Field(
        default=None,
        description="Optional dataset profiles previously generated to validate column existence and types.",
    )
    prediction_window_days: int = Field(
        default=60,
        ge=1,
        le=365,
        description="Target prediction window in days.",
    )


# ------------------------------------------------------------------------------
# Readiness Assessment Schemas
# ------------------------------------------------------------------------------


class ValidationIssue(BaseModel):
    """Individual validation failure, warning, or informational notice."""

    severity: IssueSeverity = Field(..., description="Issue severity: ERROR, WARNING, or INFO.")
    concept: ConceptType | None = Field(
        default=None, description="Associated core concept, if applicable."
    )
    field: str | None = Field(default=None, description="Associated concept field, if applicable.")
    code: str = Field(..., description="Machine-readable issue code.")
    message: str = Field(..., description="Human-readable issue description.")


class FeatureCoverageSummary(BaseModel):
    """Summary of 45 canonical features supported by the proposed schema mapping."""

    total_features: int = Field(
        default=45, description="Total canonical features required by model (45)."
    )
    covered_count: int = Field(
        ..., description="Count of canonical features derivable from mapped columns."
    )
    missing_count: int = Field(
        ..., description="Count of canonical features without underlying data."
    )
    defaulted_count: int = Field(
        ..., description="Count of missing features that will be filled with sentinels."
    )
    coverage_pct: float = Field(
        ..., description="Percentage of canonical features covered (0.0 to 100.0)."
    )
    covered_features: list[str] = Field(
        default_factory=list, description="Derivable feature names."
    )
    missing_features: list[str] = Field(default_factory=list, description="Missing feature names.")
    defaulted_features: list[str] = Field(
        default_factory=list, description="Features receiving default sentinels."
    )


class ReadinessAssessmentResponse(BaseModel):
    """Comprehensive readiness assessment report for custom dataset onboarding."""

    model_config = ConfigDict(extra="ignore")

    overall_status: ReadinessStatus = Field(
        ...,
        description="Consolidated dataset onboarding readiness: READY, PARTIALLY_READY, or UNSUPPORTED.",
    )
    target_readiness: ReadinessStatus = Field(
        ...,
        description="Readiness for point-in-time snapshotting and 60d churn target construction.",
    )
    target_readiness_reason: str = Field(
        ..., description="Detailed rationale for target construction readiness."
    )
    scoring_readiness: ReadinessStatus = Field(
        ...,
        description="Readiness for immediate inference scoring using frozen Phase 6.1 champion pipeline.",
    )
    scoring_readiness_reason: str = Field(
        ..., description="Detailed rationale for scoring readiness."
    )
    mapping_preview: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Structured preview of mapped concept fields.",
    )
    feature_coverage: FeatureCoverageSummary = Field(
        ...,
        description="Detailed coverage breakdown across 45 canonical features.",
    )
    validation_issues: list[ValidationIssue] = Field(
        default_factory=list,
        description="List of detected validation errors, warnings, and compatibility notices.",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Actionable recommendations to resolve warnings or improve data quality.",
    )
    next_steps: list[str] = Field(
        default_factory=list,
        description="Next operational steps based on overall readiness status.",
    )


# ------------------------------------------------------------------------------
# Phase 11 Custom Dataset Processing & Analysis Schemas
# ------------------------------------------------------------------------------


class RecordRejectionSummary(BaseModel):
    """Summary of records quarantined or rejected during schema transformation."""

    total_rejected: int = Field(
        default=0, description="Total count of rejected or quarantined rows."
    )
    rejected_by_table: dict[str, int] = Field(
        default_factory=dict, description="Rejection count breakdown by source concept."
    )
    reasons: list[str] = Field(
        default_factory=list, description="Sanitized explanations for record rejections."
    )


class SnapshotGenerationSummary(BaseModel):
    """Summary of point-in-time customer snapshots generated."""

    total_snapshots: int = Field(..., description="Total point-in-time snapshot records generated.")
    observation_dates_count: int = Field(
        ..., description="Count of distinct observation dates evaluated."
    )
    observation_dates: list[str] = Field(
        default_factory=list, description="ISO strings of evaluated observation dates."
    )
    eligible_accounts_count: int = Field(
        ..., description="Count of distinct eligible accounts across snapshots."
    )
    positive_targets_count: int = Field(
        ..., description="Count of snapshots with future churn target = 1."
    )
    negative_targets_count: int = Field(
        ..., description="Count of snapshots with future churn target = 0."
    )
    censored_targets_count: int = Field(
        ..., description="Count of snapshots right-censored (target = None)."
    )
    churn_rate_pct: float | None = Field(
        default=None, description="Observed churn rate percentage among uncensored snapshots."
    )


class PrioritizationQuadrantSummary(BaseModel):
    """Aggregated portfolio Value x Risk prioritization quadrant detail."""

    quadrant_name: str = Field(..., description="Name of prioritization quadrant.")
    priority_level: str | int = Field(..., description="Urgency priority classification.")
    account_count: int = Field(..., description="Number of customer accounts in this quadrant.")
    account_percentage: float = Field(
        ..., description="Percentage of portfolio accounts in this quadrant."
    )
    total_mrr: float = Field(..., description="Total active monthly recurring revenue in quadrant.")
    estimated_revenue_at_risk_mrr: float = Field(
        ..., description="Uncalibrated MRR x model_score prioritization estimate."
    )
    action_recommendation: str = Field(
        ..., description="Recommended customer success intervention."
    )


class ScoringSummary(BaseModel):
    """Summary of model predictions, explanations, and segment assignments."""

    scored_snapshots_count: int = Field(
        ..., description="Total customer snapshots scored by model."
    )
    model_name: str = Field(default="flurit_churn_predictor", description="Name of serving model.")
    model_version: str = Field(default="1.0.0", description="Model version.")
    average_model_score: float = Field(
        ..., description="Mean uncalibrated model score across scored records."
    )
    high_risk_count: int = Field(default=0, description="Count of accounts in High Risk tier.")
    moderate_risk_count: int = Field(
        default=0, description="Count of accounts in Moderate Risk tier."
    )
    low_risk_count: int = Field(default=0, description="Count of accounts in Low Risk tier.")
    segments_distribution: dict[str, int] = Field(
        default_factory=dict, description="Account distribution across K-Means clusters."
    )
    prioritization_quadrants: list[PrioritizationQuadrantSummary] = Field(
        default_factory=list, description="Portfolio-level Value x Risk quadrant breakdown."
    )
    top_global_risk_increasing_factors: list[dict[str, Any]] = Field(
        default_factory=list, description="Features driving risk higher across scored accounts."
    )
    score_interpretation_notice: str = Field(
        ..., description="Mandatory operational score interpretation notice."
    )


class ProcessingOptions(BaseModel):
    """Configurable options for custom dataset processing execution."""

    prediction_window_days: int = Field(
        default=60, ge=1, le=365, description="Target prediction horizon in days."
    )
    observation_dates: list[str] | None = Field(
        default=None, description="Optional explicit observation dates (YYYY-MM-DD)."
    )
    allow_sentinel_scoring: bool = Field(
        default=False,
        description="Explicitly allow scoring with default sentinels when usage/support telemetry is missing.",
    )
    score_latest_only: bool = Field(
        default=False,
        description="Score only the latest observation date cohort instead of all historical snapshots.",
    )


class EndToEndAnalysisResponse(BaseModel):
    """Complete response payload for custom dataset processing and ML analysis."""

    model_config = ConfigDict(extra="ignore")

    processing_status: str = Field(
        ...,
        description="Execution status: SUCCESS, PARTIAL_SUCCESS, NOT_READY, INFERENCE_FAILED, or FAILED.",
    )
    analysis_ready: bool = Field(
        ..., description="Whether relational structure and mandatory concepts were parsed."
    )
    label_construction_ready: bool = Field(
        ...,
        description="Whether point-in-time snapshots and 60d churn targets could be constructed.",
    )
    existing_model_scoring_ready: bool = Field(
        ...,
        description="Whether dataset is semantically compatible for scoring with frozen champion model.",
    )
    feature_coverage: FeatureCoverageSummary | None = Field(
        default=None,
        description="Canonical 45-feature coverage from the submitted schema mapping.",
    )
    sentinel_scoring_used: bool = Field(
        default=False,
        description="True only when the caller explicitly enabled sentinel-based scoring.",
    )
    persisted_summary: dict[str, Any] | None = Field(
        default=None,
        description="Counts and model run identifier after durable persistence completes.",
    )
    scoring_blockers: list[str] = Field(
        default_factory=list,
        description="Specific reasons why model scoring was blocked or omitted.",
    )
    rejection_summary: RecordRejectionSummary = Field(
        default_factory=RecordRejectionSummary,
        description="Summary of rejected/quarantined input records.",
    )
    snapshot_summary: SnapshotGenerationSummary | None = Field(
        default=None, description="Point-in-time snapshot and label distribution metrics."
    )
    scoring_summary: ScoringSummary | None = Field(
        default=None, description="Model inference, segmentation, and prioritization results."
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings regarding distribution shift or data quality.",
    )
    next_steps: list[str] = Field(
        default_factory=list, description="Actionable next operational steps."
    )


class ProcessingPreviewResponse(BaseModel):
    """Dry-run preview response checking transformation and readiness without executing heavy ML."""

    analysis_ready: bool = Field(..., description="Whether relational structure is valid.")
    label_construction_ready: bool = Field(
        ..., description="Whether 60d target labeling is feasible."
    )
    existing_model_scoring_ready: bool = Field(
        ..., description="Whether full 45-feature scoring is supported."
    )
    total_input_rows: int = Field(..., description="Total rows parsed across all uploaded files.")
    rejection_summary: RecordRejectionSummary = Field(
        default_factory=RecordRejectionSummary, description="Quarantined records."
    )
    scoring_blockers: list[str] = Field(
        default_factory=list, description="Blockers preventing direct model scoring."
    )
    estimated_eligible_accounts: int = Field(
        default=0, description="Estimated number of currently active accounts."
    )
    potential_observation_dates: list[str] = Field(
        default_factory=list,
        description="Feasible monthly observation dates within date span.",
    )
    warnings: list[str] = Field(default_factory=list, description="Data quality notices.")
    next_steps: list[str] = Field(default_factory=list, description="Recommended next steps.")


class JobSubmissionResponse(BaseModel):
    """Response returned upon successfully enqueuing an asynchronous processing job (HTTP 202)."""

    model_config = ConfigDict(extra="ignore")

    job_id: str = Field(..., description="Unique processing job identifier.")
    status: str = Field(..., description="Initial job status (typically 'queued').")
    workspace_id: str = Field(..., description="Authorized workspace identifier.")
    created_at: str = Field(..., description="Job creation timestamp in ISO 8601 format.")
    message: str = Field(..., description="Human-readable submission status confirmation.")


class JobStatusResponse(BaseModel):
    """Detailed status, progress, and result payload for an asynchronous processing job."""

    model_config = ConfigDict(extra="ignore")

    job_id: str = Field(..., description="Unique processing job identifier.")
    workspace_id: str = Field(..., description="Authorized workspace identifier.")
    dataset_name: str | None = Field(
        default=None, description="Optional dataset descriptor or name."
    )
    status: str = Field(
        ...,
        description="Current job status: queued, running, succeeded, failed, or cancelled.",
    )
    progress_pct: int = Field(
        default=0,
        description="Execution progress percentage (0 to 100).",
    )
    stage_message: str | None = Field(
        default=None,
        description="Current processing pipeline stage or execution status.",
    )
    created_at: str = Field(..., description="Creation ISO timestamp.")
    started_at: str | None = Field(default=None, description="Execution start ISO timestamp.")
    completed_at: str | None = Field(default=None, description="Completion ISO timestamp.")
    error_message: str | None = Field(
        default=None,
        description="Sanitized failure description if job failed (no stack traces or secrets).",
    )
    result_summary: dict[str, Any] | None = Field(
        default=None,
        description="Structured processing and analysis results when succeeded.",
    )


class JobListResponse(BaseModel):
    """Paginated list of processing jobs for the caller workspace."""

    model_config = ConfigDict(extra="ignore")

    total: int = Field(..., description="Total count of processing jobs matching criteria.")
    jobs: list[JobStatusResponse] = Field(
        default_factory=list,
        description="List of workspace processing jobs.",
    )


class JobCancelResponse(BaseModel):
    """Response returned upon cancelling an asynchronous job."""

    model_config = ConfigDict(extra="ignore")

    job_id: str = Field(..., description="Cancelled processing job identifier.")
    status: str = Field(..., description="Updated status after cancellation.")
    message: str = Field(..., description="Cancellation confirmation message.")
