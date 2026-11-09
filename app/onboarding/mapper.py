"""Schema Mapping and Readiness Assessment Engine for Flurit Onboarding.

Evaluates user-provided concept-to-column mappings against Flurit's 5 core SaaS
concepts (Accounts, Subscriptions, Feature Usage, Support Tickets, Churn Events),
validates data types and temporal spans, and determines readiness for:
  1. Historical point-in-time snapshotting & 60-day churn target construction.
  2. Direct inference scoring against the frozen Phase 6.1 champion pipeline.
"""

from datetime import datetime
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ml.features.schema import FEATURE_DEFINITIONS, FeatureGroup
from app.onboarding.schemas import (
    ColumnProfile,
    ConceptType,
    FeatureCoverageSummary,
    InferredDataType,
    IssueSeverity,
    ReadinessAssessmentResponse,
    ReadinessStatus,
    SchemaMappingRequest,
    ValidationIssue,
)

logger = get_logger("flurit.onboarding.mapper")

# Canonical features grouped by domain
CANONICAL_FEATURES_BY_GROUP: dict[FeatureGroup, list[str]] = {
    group: [f.name for f in FEATURE_DEFINITIONS if f.group == group] for group in FeatureGroup
}
ALL_CANONICAL_FEATURES: list[str] = [f.name for f in FEATURE_DEFINITIONS]


class SchemaMapper:
    """Evaluates concept mappings, validates column semantics, and assesses onboarding readiness."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def assess_readiness(self, request: SchemaMappingRequest) -> ReadinessAssessmentResponse:
        """Run comprehensive validation and generate onboarding readiness report."""
        issues: list[ValidationIssue] = []
        mapping_preview: dict[str, dict[str, Any]] = {}

        # Build column metadata lookup if dataset profiles were provided
        column_lookup: dict[str, ColumnProfile] = {}
        all_dataset_span_days: int | None = None
        global_min_dt: datetime | None = None
        global_max_dt: datetime | None = None

        if request.dataset_profiles:
            for ds in request.dataset_profiles:
                for col in ds.columns:
                    # Store by column_name (case-insensitive key mapping)
                    column_lookup[col.column_name.lower()] = col
                if ds.date_range and ds.date_range.min_date and ds.date_range.max_date:
                    try:
                        d_min = datetime.fromisoformat(ds.date_range.min_date)
                        d_max = datetime.fromisoformat(ds.date_range.max_date)
                        if global_min_dt is None or d_min < global_min_dt:
                            global_min_dt = d_min
                        if global_max_dt is None or d_max > global_max_dt:
                            global_max_dt = d_max
                    except Exception:
                        pass

        if global_min_dt is not None and global_max_dt is not None:
            all_dataset_span_days = max(0, (global_max_dt.date() - global_min_dt.date()).days)

        # ----------------------------------------------------------------------
        # 1. Validate Concept Mappings & Column Existence
        # ----------------------------------------------------------------------
        has_accounts = request.accounts is not None
        has_subscriptions = request.subscriptions is not None
        has_usage = request.feature_usage is not None
        has_support = request.support_tickets is not None
        has_churn = request.churn_events is not None

        # Accounts validation
        if has_accounts and request.accounts:
            mapping_preview["accounts"] = request.accounts.model_dump()
            self._validate_column(
                concept=ConceptType.ACCOUNTS,
                field="account_id",
                col_name=request.accounts.account_id,
                column_lookup=column_lookup,
                expected_types=[InferredDataType.TEXT, InferredDataType.INTEGER],
                issues=issues,
                is_mandatory=True,
            )
            self._validate_column(
                concept=ConceptType.ACCOUNTS,
                field="signup_date",
                col_name=request.accounts.signup_date,
                column_lookup=column_lookup,
                expected_types=[InferredDataType.DATETIME],
                issues=issues,
                is_mandatory=True,
            )
            if request.accounts.industry:
                self._validate_column(
                    concept=ConceptType.ACCOUNTS,
                    field="industry",
                    col_name=request.accounts.industry,
                    column_lookup=column_lookup,
                    expected_types=[InferredDataType.CATEGORY, InferredDataType.TEXT],
                    issues=issues,
                )
            if request.accounts.country:
                self._validate_column(
                    concept=ConceptType.ACCOUNTS,
                    field="country",
                    col_name=request.accounts.country,
                    column_lookup=column_lookup,
                    expected_types=[InferredDataType.CATEGORY, InferredDataType.TEXT],
                    issues=issues,
                )
        else:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    concept=ConceptType.ACCOUNTS,
                    code="MISSING_CONCEPT_MAPPING",
                    message="Accounts concept mapping is missing. Mandatory fields 'account_id' and 'signup_date' are required.",
                )
            )

        # Subscriptions validation
        if has_subscriptions and request.subscriptions:
            mapping_preview["subscriptions"] = request.subscriptions.model_dump()
            self._validate_column(
                concept=ConceptType.SUBSCRIPTIONS,
                field="subscription_id",
                col_name=request.subscriptions.subscription_id,
                column_lookup=column_lookup,
                expected_types=[InferredDataType.TEXT, InferredDataType.INTEGER],
                issues=issues,
                is_mandatory=True,
            )
            self._validate_column(
                concept=ConceptType.SUBSCRIPTIONS,
                field="account_id",
                col_name=request.subscriptions.account_id,
                column_lookup=column_lookup,
                expected_types=[InferredDataType.TEXT, InferredDataType.INTEGER],
                issues=issues,
                is_mandatory=True,
            )
            self._validate_column(
                concept=ConceptType.SUBSCRIPTIONS,
                field="start_date",
                col_name=request.subscriptions.start_date,
                column_lookup=column_lookup,
                expected_types=[InferredDataType.DATETIME],
                issues=issues,
                is_mandatory=True,
            )
            if request.subscriptions.end_date:
                self._validate_column(
                    concept=ConceptType.SUBSCRIPTIONS,
                    field="end_date",
                    col_name=request.subscriptions.end_date,
                    column_lookup=column_lookup,
                    expected_types=[InferredDataType.DATETIME],
                    issues=issues,
                )
            if request.subscriptions.mrr:
                self._validate_column(
                    concept=ConceptType.SUBSCRIPTIONS,
                    field="mrr",
                    col_name=request.subscriptions.mrr,
                    column_lookup=column_lookup,
                    expected_types=[InferredDataType.FLOAT, InferredDataType.INTEGER],
                    issues=issues,
                )
        else:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    concept=ConceptType.SUBSCRIPTIONS,
                    code="MISSING_CONCEPT_MAPPING",
                    message="Subscriptions concept mapping is missing. Mandatory fields 'subscription_id', 'account_id', and 'start_date' are required.",
                )
            )

        # Feature Usage validation
        if has_usage and request.feature_usage:
            mapping_preview["feature_usage"] = request.feature_usage.model_dump()
            self._validate_column(
                concept=ConceptType.FEATURE_USAGE,
                field="event_time",
                col_name=request.feature_usage.event_time,
                column_lookup=column_lookup,
                expected_types=[InferredDataType.DATETIME],
                issues=issues,
                is_mandatory=True,
            )
            if not request.feature_usage.account_id and not request.feature_usage.subscription_id:
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.ERROR,
                        concept=ConceptType.FEATURE_USAGE,
                        code="MISSING_RELATIONSHIP_KEY",
                        message="Feature usage requires either 'account_id' or 'subscription_id' to associate activity with customer.",
                    )
                )
        else:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    concept=ConceptType.FEATURE_USAGE,
                    code="OPTIONAL_CONCEPT_OMITTED",
                    message="Feature usage concept mapping not provided. All 14 usage-based features will receive default sentinels (zeros).",
                )
            )

        # Support Tickets validation
        if has_support and request.support_tickets:
            mapping_preview["support_tickets"] = request.support_tickets.model_dump()
            self._validate_column(
                concept=ConceptType.SUPPORT_TICKETS,
                field="ticket_id",
                col_name=request.support_tickets.ticket_id,
                column_lookup=column_lookup,
                expected_types=[InferredDataType.TEXT, InferredDataType.INTEGER],
                issues=issues,
                is_mandatory=True,
            )
            self._validate_column(
                concept=ConceptType.SUPPORT_TICKETS,
                field="account_id",
                col_name=request.support_tickets.account_id,
                column_lookup=column_lookup,
                expected_types=[InferredDataType.TEXT, InferredDataType.INTEGER],
                issues=issues,
                is_mandatory=True,
            )
            self._validate_column(
                concept=ConceptType.SUPPORT_TICKETS,
                field="created_at",
                col_name=request.support_tickets.created_at,
                column_lookup=column_lookup,
                expected_types=[InferredDataType.DATETIME],
                issues=issues,
                is_mandatory=True,
            )
        else:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    concept=ConceptType.SUPPORT_TICKETS,
                    code="OPTIONAL_CONCEPT_OMITTED",
                    message="Support tickets concept mapping not provided. All 11 support-related features will receive default sentinels.",
                )
            )

        # Churn Events validation
        has_subscription_end_date = bool(request.subscriptions and request.subscriptions.end_date)
        if has_churn and request.churn_events:
            mapping_preview["churn_events"] = request.churn_events.model_dump()
            self._validate_column(
                concept=ConceptType.CHURN_EVENTS,
                field="account_id",
                col_name=request.churn_events.account_id,
                column_lookup=column_lookup,
                expected_types=[InferredDataType.TEXT, InferredDataType.INTEGER],
                issues=issues,
                is_mandatory=True,
            )
            self._validate_column(
                concept=ConceptType.CHURN_EVENTS,
                field="churn_date",
                col_name=request.churn_events.churn_date,
                column_lookup=column_lookup,
                expected_types=[InferredDataType.DATETIME],
                issues=issues,
                is_mandatory=True,
            )
        else:
            if not has_subscription_end_date:
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.ERROR,
                        concept=ConceptType.CHURN_EVENTS,
                        code="NO_CHURN_GROUND_TRUTH",
                        message="Neither 'churn_events' table nor 'subscriptions.end_date' is mapped. Cannot construct historical churn ground truth.",
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.INFO,
                        concept=ConceptType.CHURN_EVENTS,
                        code="DERIVED_CHURN_GROUND_TRUTH",
                        message="No dedicated churn_events table mapped; churn ground-truth will be derived from subscription termination dates (end_date).",
                    )
                )

        # ----------------------------------------------------------------------
        # 2. Evaluate Temporal Span & Target Construction Readiness
        # ----------------------------------------------------------------------
        target_readiness, target_reason = self._evaluate_target_readiness(
            has_accounts=has_accounts,
            has_subscriptions=has_subscriptions,
            has_churn_or_end_date=(has_churn or has_subscription_end_date),
            span_days=all_dataset_span_days,
            prediction_window_days=request.prediction_window_days,
            issues=issues,
        )

        # ----------------------------------------------------------------------
        # 3. Evaluate Canonical Feature Coverage & Scoring Readiness
        # ----------------------------------------------------------------------
        coverage_summary = self._compute_feature_coverage(
            has_accounts=has_accounts,
            has_subscriptions=has_subscriptions,
            has_usage=has_usage,
            has_support=has_support,
            has_churn=has_churn,
            request=request,
        )

        scoring_readiness, scoring_reason = self._evaluate_scoring_readiness(
            has_accounts=has_accounts,
            has_subscriptions=has_subscriptions,
            has_usage=has_usage,
            has_support=has_support,
            coverage=coverage_summary,
            issues=issues,
        )

        # ----------------------------------------------------------------------
        # 4. Synthesize Overall Status, Recommendations & Next Steps
        # ----------------------------------------------------------------------
        has_critical_errors = any(i.severity == IssueSeverity.ERROR for i in issues)

        if (
            has_critical_errors
            or target_readiness == ReadinessStatus.UNSUPPORTED
            or scoring_readiness == ReadinessStatus.UNSUPPORTED
        ):
            overall_status = ReadinessStatus.UNSUPPORTED
        elif (
            target_readiness == ReadinessStatus.READY and scoring_readiness == ReadinessStatus.READY
        ):
            overall_status = ReadinessStatus.READY
        else:
            overall_status = ReadinessStatus.PARTIALLY_READY

        recommendations = self._generate_recommendations(
            issues=issues,
            coverage=coverage_summary,
            span_days=all_dataset_span_days,
            target_readiness=target_readiness,
            scoring_readiness=scoring_readiness,
        )

        next_steps = self._generate_next_steps(
            overall_status=overall_status,
            target_readiness=target_readiness,
            scoring_readiness=scoring_readiness,
        )

        return ReadinessAssessmentResponse(
            overall_status=overall_status,
            target_readiness=target_readiness,
            target_readiness_reason=target_reason,
            scoring_readiness=scoring_readiness,
            scoring_readiness_reason=scoring_reason,
            mapping_preview=mapping_preview,
            feature_coverage=coverage_summary,
            validation_issues=issues,
            recommendations=recommendations,
            next_steps=next_steps,
        )

    def _validate_column(
        self,
        concept: ConceptType,
        field: str,
        col_name: str,
        column_lookup: dict[str, ColumnProfile],
        expected_types: list[InferredDataType],
        issues: list[ValidationIssue],
        is_mandatory: bool = False,
    ) -> None:
        """Validate column existence, nullity, and type against profiling metadata."""
        col_clean = col_name.strip()
        if not col_clean:
            if is_mandatory:
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.ERROR,
                        concept=concept,
                        field=field,
                        code="EMPTY_COLUMN_MAPPING",
                        message=f"Mandatory field '{concept.value}.{field}' mapped to empty column name.",
                    )
                )
            return

        if not column_lookup:
            # No profiles provided to check against
            return

        col_key = col_clean.lower()
        if col_key not in column_lookup:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.ERROR if is_mandatory else IssueSeverity.WARNING,
                    concept=concept,
                    field=field,
                    code="COLUMN_NOT_FOUND",
                    message=f"Mapped column '{col_clean}' for '{concept.value}.{field}' does not exist in the provided dataset profiles.",
                )
            )
            return

        col_prof = column_lookup[col_key]

        # Check null rate for identifier fields
        if is_mandatory and field.endswith("_id") and col_prof.null_count > 0:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    concept=concept,
                    field=field,
                    code="NULL_IDENTIFIER_VALUES",
                    message=f"Column '{col_clean}' mapped to ID field '{field}' contains {col_prof.null_count:,} missing values ({col_prof.null_percentage}%). Rows with null IDs will be excluded.",
                )
            )

        # Check type compatibility
        if expected_types and col_prof.inferred_type not in expected_types:
            # Soften datetime vs text: if inferred as text, datetime parsing might still succeed
            if (
                InferredDataType.DATETIME in expected_types
                and col_prof.inferred_type == InferredDataType.TEXT
            ):
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.WARNING,
                        concept=concept,
                        field=field,
                        code="POSSIBLE_DATETIME_MISMATCH",
                        message=f"Column '{col_clean}' mapped to timestamp field '{field}' was inferred as text, not datetime. Ensure timestamps are in ISO 8601 or YYYY-MM-DD format.",
                    )
                )
            else:
                expected_str = "/".join(t.value for t in expected_types)
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.WARNING,
                        concept=concept,
                        field=field,
                        code="DATA_TYPE_MISMATCH",
                        message=f"Column '{col_clean}' inferred as '{col_prof.inferred_type.value}', expected {expected_str}.",
                    )
                )

    def _evaluate_target_readiness(
        self,
        has_accounts: bool,
        has_subscriptions: bool,
        has_churn_or_end_date: bool,
        span_days: int | None,
        prediction_window_days: int,
        issues: list[ValidationIssue],
    ) -> tuple[ReadinessStatus, str]:
        """Assess feasibility of leak-free point-in-time snapshot and target construction."""
        if not has_accounts or not has_subscriptions:
            return (
                ReadinessStatus.UNSUPPORTED,
                "Both Account and Subscription concepts are required to establish point-in-time customer snapshot eligibility.",
            )

        if not has_churn_or_end_date:
            return (
                ReadinessStatus.UNSUPPORTED,
                "No churn ground truth available. Either churn_events table or subscriptions.end_date is required to construct target labels.",
            )

        if span_days is not None:
            if span_days < prediction_window_days:
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.ERROR,
                        code="INSUFFICIENT_TEMPORAL_SPAN",
                        message=f"Dataset temporal span is {span_days} days, which is less than the required prediction window ({prediction_window_days} days).",
                    )
                )
                return (
                    ReadinessStatus.UNSUPPORTED,
                    f"Temporal span ({span_days} days) is less than the {prediction_window_days}-day prediction horizon. Cannot label future churn targets.",
                )

            recommended_span = self.settings.ONBOARDING_RECOMMENDED_HISTORY_DAYS
            if span_days < recommended_span:
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.WARNING,
                        code="LIMITED_LOOKBACK_WINDOW",
                        message=f"Dataset temporal span is {span_days} days (recommended: {recommended_span} days). Early snapshots will have truncated 90-day lookback windows.",
                    )
                )
                return (
                    ReadinessStatus.PARTIALLY_READY,
                    f"Dataset temporal span ({span_days} days) is sufficient for {prediction_window_days}d target labeling, but below recommended {recommended_span}d for full historical lookbacks.",
                )

        return (
            ReadinessStatus.READY,
            "Dataset provides all required temporal and relational concepts with sufficient date span for leak-free point-in-time snapshotting.",
        )

    def _compute_feature_coverage(
        self,
        has_accounts: bool,
        has_subscriptions: bool,
        has_usage: bool,
        has_support: bool,
        has_churn: bool,
        request: SchemaMappingRequest,
    ) -> FeatureCoverageSummary:
        """Calculate coverage and sentinel defaulting across all 45 canonical features."""
        covered: list[str] = []
        missing: list[str] = []
        defaulted: list[str] = []

        for feat in FEATURE_DEFINITIONS:
            name = feat.name
            group = feat.group

            if group == FeatureGroup.ACCOUNT:
                if has_accounts:
                    covered.append(name)
                else:
                    missing.append(name)
                    defaulted.append(name)

            elif group == FeatureGroup.SUBSCRIPTION:
                if has_subscriptions:
                    covered.append(name)
                else:
                    missing.append(name)
                    defaulted.append(name)

            elif group == FeatureGroup.USAGE:
                if has_usage:
                    covered.append(name)
                else:
                    missing.append(name)
                    defaulted.append(name)

            elif group == FeatureGroup.SUPPORT:
                if has_support:
                    covered.append(name)
                else:
                    missing.append(name)
                    defaulted.append(name)

            elif group == FeatureGroup.CHURN_HISTORY:
                if has_churn:
                    covered.append(name)
                else:
                    missing.append(name)
                    defaulted.append(name)

        covered_count = len(covered)
        total_count = len(ALL_CANONICAL_FEATURES)
        coverage_pct = round((covered_count / total_count * 100.0), 1)

        return FeatureCoverageSummary(
            total_features=total_count,
            covered_count=covered_count,
            missing_count=len(missing),
            defaulted_count=len(defaulted),
            covered_features=covered,
            missing_features=missing,
            defaulted_features=defaulted,
            coverage_pct=coverage_pct,
        )

    def _evaluate_scoring_readiness(
        self,
        has_accounts: bool,
        has_subscriptions: bool,
        has_usage: bool,
        has_support: bool,
        coverage: FeatureCoverageSummary,
        issues: list[ValidationIssue],
    ) -> tuple[ReadinessStatus, str]:
        """Assess compatibility for immediate inference with frozen Phase 6.1 Logistic Regression."""
        if not has_accounts or not has_subscriptions:
            return (
                ReadinessStatus.UNSUPPORTED,
                "Frozen champion model requires primary customer profile and financial subscription attributes to compute inference scores.",
            )

        if not has_usage or not has_support:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    code="MODEL_DISTRIBUTION_SHIFT",
                    message="Missing usage or support ticket data. While inference scoring is technically executable by sentinel imputation, model accuracy will be compromised due to severe feature distribution shift.",
                )
            )
            return (
                ReadinessStatus.PARTIALLY_READY,
                f"Scoring is executable via sentinel imputation ({coverage.covered_count}/45 features mapped, {coverage.coverage_pct}% coverage), but model predictions will exhibit significant distribution shift. Custom retraining is advised.",
            )

        return (
            ReadinessStatus.READY,
            "All canonical feature domains (Account, Subscription, Usage, Support, Churn History) are mapped. Dataset is fully compatible with frozen inference pipeline.",
        )

    def _generate_recommendations(
        self,
        issues: list[ValidationIssue],
        coverage: FeatureCoverageSummary,
        span_days: int | None,
        target_readiness: ReadinessStatus,
        scoring_readiness: ReadinessStatus,
    ) -> list[str]:
        """Generate tailored, actionable recommendations for resolving data deficiencies."""
        recs: list[str] = []

        if target_readiness == ReadinessStatus.UNSUPPORTED:
            recs.append(
                "Upload a historical churn events dataset or ensure subscription end dates are provided to define ground-truth churn labels."
            )

        if span_days is not None and span_days < 150:
            recs.append(
                f"Extend historical observation span from {span_days} days to at least 150 days to enable complete 90-day feature lookback windows prior to 60-day prediction intervals."
            )

        if coverage.missing_count > 0:
            missing_groups = set()
            for feat_name in coverage.missing_features:
                for f_def in FEATURE_DEFINITIONS:
                    if f_def.name == feat_name:
                        missing_groups.add(f_def.group.value)
            recs.append(
                f"Provide dataset mappings for missing domain groups ({', '.join(sorted(missing_groups))}) to improve feature coverage above current {coverage.coverage_pct}%."
            )

        # Check for null identifier warnings
        if any(i.code == "NULL_IDENTIFIER_VALUES" for i in issues):
            recs.append(
                "Cleanse source tables to eliminate null values in primary and foreign key columns before initiating snapshot generation."
            )

        if scoring_readiness == ReadinessStatus.PARTIALLY_READY:
            recs.append(
                "Because of missing domain features, consider using Flurit's model training pipeline to train a dedicated model on your mapped features rather than relying on the frozen RavenStack model."
            )

        if not recs:
            recs.append(
                "All schema requirements and temporal thresholds satisfied. Ready for pipeline execution."
            )

        return recs

    def _generate_next_steps(
        self,
        overall_status: ReadinessStatus,
        target_readiness: ReadinessStatus,
        scoring_readiness: ReadinessStatus,
    ) -> list[str]:
        """Generate clear sequential operational next steps."""
        if overall_status == ReadinessStatus.READY:
            return [
                "1. Confirm approved concept mappings in Flurit configuration.",
                "2. Trigger point-in-time customer snapshot generation across approved observation dates.",
                "3. Compute canonical 45-feature vectors using feature engineering engine.",
                "4. Execute batch scoring and SHAP explainability via /api/v1/predictions endpoints.",
            ]
        elif overall_status == ReadinessStatus.PARTIALLY_READY:
            steps = ["1. Review schema validation warnings and feature coverage report."]
            if scoring_readiness == ReadinessStatus.PARTIALLY_READY:
                steps.append(
                    "2. Decide whether to run inference with default sentinel imputation or upload missing domain datasets (usage, support)."
                )
            if target_readiness == ReadinessStatus.PARTIALLY_READY:
                steps.append(
                    "3. Verify that truncated lookback windows for early customer cohorts are acceptable for your analysis."
                )
            steps.append("4. Proceed with restricted snapshotting or upload enriched datasets.")
            return steps
        else:
            return [
                "1. Address critical schema validation errors identified in this report.",
                "2. Ensure mandatory Account and Subscription datasets with required timestamp fields are uploaded.",
                "3. Re-run schema mapping readiness assessment.",
            ]
