"""Canonical Schema Transformer for Flurit Onboarding.

Transforms validated user-supplied CSV DataFrames and concept mappings into
Flurit's canonical intermediate relational representation, enforcing referential
integrity, date parsing, and record quarantine logging without leaking PII.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from app.core.logging import get_logger
from app.onboarding.schemas import (
    RecordRejectionSummary,
    SchemaMappingRequest,
)

logger = get_logger("flurit.onboarding.transformer")


@dataclass
class CanonicalIntermediateDataset:
    """Normalized intermediate tables ready for snapshotting and feature pipelines."""

    accounts_df: pd.DataFrame
    subscriptions_df: pd.DataFrame
    usage_df: pd.DataFrame
    tickets_df: pd.DataFrame
    churn_events_df: pd.DataFrame
    rejection_summary: RecordRejectionSummary
    warnings: list[str] = field(default_factory=list)
    min_date: date | None = None
    max_date: date | None = None
    span_days: int = 0
    derived_churn_from_subscriptions: bool = False
    workspace_id: str = "ws_default"


class SchemaTransformer:
    """Transforms disparate source schemas into Flurit's canonical intermediate structures."""

    def transform(
        self,
        raw_dfs: dict[str, pd.DataFrame],
        mapping: SchemaMappingRequest,
        workspace_id: str = "ws_default",
    ) -> CanonicalIntermediateDataset:
        """Execute relational extraction, normalization, and referential integrity checks."""
        rejected_by_table: dict[str, int] = {
            "accounts": 0,
            "subscriptions": 0,
            "feature_usage": 0,
            "support_tickets": 0,
            "churn_events": 0,
        }
        rejection_reasons: list[str] = []
        warnings: list[str] = []

        # ----------------------------------------------------------------------
        # 1. Accounts Transformation
        # ----------------------------------------------------------------------
        if not mapping.accounts:
            raise ValueError("Mandatory concept mapping for 'accounts' is missing.")

        acc_raw = self._resolve_dataframe_for_concept(
            raw_dfs,
            concept_name="accounts",
            required_cols=[mapping.accounts.account_id, mapping.accounts.signup_date],
        )

        acc_clean, acc_rejections = self._transform_accounts(acc_raw, mapping.accounts)
        rejected_by_table["accounts"] += acc_rejections["count"]
        rejection_reasons.extend(acc_rejections["reasons"])

        if acc_clean.empty:
            raise ValueError("All account records were rejected due to invalid IDs or dates.")

        valid_account_ids: set[str] = set(acc_clean["account_id"].unique())

        # ----------------------------------------------------------------------
        # 2. Subscriptions Transformation
        # ----------------------------------------------------------------------
        if not mapping.subscriptions:
            raise ValueError("Mandatory concept mapping for 'subscriptions' is missing.")

        sub_raw = self._resolve_dataframe_for_concept(
            raw_dfs,
            concept_name="subscriptions",
            required_cols=[
                mapping.subscriptions.subscription_id,
                mapping.subscriptions.account_id,
                mapping.subscriptions.start_date,
            ],
        )

        sub_clean, sub_rejections = self._transform_subscriptions(
            sub_raw, mapping.subscriptions, valid_account_ids
        )
        rejected_by_table["subscriptions"] += sub_rejections["count"]
        rejection_reasons.extend(sub_rejections["reasons"])

        if sub_clean.empty:
            raise ValueError(
                "All subscription records were rejected due to missing keys or orphans."
            )

        # ----------------------------------------------------------------------
        # 3. Feature Usage Transformation (Optional)
        # ----------------------------------------------------------------------
        usage_clean: pd.DataFrame
        if mapping.feature_usage:
            req_cols = [mapping.feature_usage.event_time]
            if mapping.feature_usage.subscription_id:
                req_cols.append(mapping.feature_usage.subscription_id)
            elif mapping.feature_usage.account_id:
                req_cols.append(mapping.feature_usage.account_id)

            usage_raw = self._resolve_dataframe_for_concept(
                raw_dfs,
                concept_name="feature_usage",
                required_cols=req_cols,
            )
            usage_clean, usage_rejections = self._transform_usage(
                usage_raw, mapping.feature_usage, sub_clean, valid_account_ids
            )
            rejected_by_table["feature_usage"] += usage_rejections["count"]
            rejection_reasons.extend(usage_rejections["reasons"])
        else:
            usage_clean = self._empty_usage_df()

        # ----------------------------------------------------------------------
        # 4. Support Tickets Transformation (Optional)
        # ----------------------------------------------------------------------
        tickets_clean: pd.DataFrame
        if mapping.support_tickets:
            req_cols = [
                mapping.support_tickets.ticket_id,
                mapping.support_tickets.account_id,
                mapping.support_tickets.created_at,
            ]
            tickets_raw = self._resolve_dataframe_for_concept(
                raw_dfs,
                concept_name="support_tickets",
                required_cols=req_cols,
            )
            tickets_clean, ticket_rejections = self._transform_tickets(
                tickets_raw, mapping.support_tickets, valid_account_ids
            )
            rejected_by_table["support_tickets"] += ticket_rejections["count"]
            rejection_reasons.extend(ticket_rejections["reasons"])
        else:
            tickets_clean = self._empty_tickets_df()

        # ----------------------------------------------------------------------
        # 5. Churn Events Transformation (Optional / Derived)
        # ----------------------------------------------------------------------
        churn_clean: pd.DataFrame
        derived_churn = False
        if mapping.churn_events:
            req_cols = [mapping.churn_events.account_id, mapping.churn_events.churn_date]
            churn_raw = self._resolve_dataframe_for_concept(
                raw_dfs,
                concept_name="churn_events",
                required_cols=req_cols,
            )
            churn_clean, churn_rejections = self._transform_churn(
                churn_raw, mapping.churn_events, valid_account_ids
            )
            rejected_by_table["churn_events"] += churn_rejections["count"]
            rejection_reasons.extend(churn_rejections["reasons"])
        elif mapping.subscriptions and mapping.subscriptions.end_date:
            # Derive churn events from terminated subscriptions
            churn_clean = self._derive_churn_from_subscriptions(sub_clean)
            derived_churn = True
            warnings.append(
                "Churn events table not supplied. Churn ground-truth was derived from subscription end dates."
            )
        else:
            churn_clean = self._empty_churn_df()

        # ----------------------------------------------------------------------
        # 6. Global Temporal Bounds Computation
        # ----------------------------------------------------------------------
        dates: list[date] = []
        if not acc_clean.empty:
            dates.extend(pd.to_datetime(acc_clean["signup_date"]).dt.date.dropna().tolist())
        if not sub_clean.empty:
            dates.extend(pd.to_datetime(sub_clean["start_date"]).dt.date.dropna().tolist())
            valid_ends = sub_clean["end_date"].dropna()
            if not valid_ends.empty:
                dates.extend(pd.to_datetime(valid_ends).dt.date.tolist())
        if not churn_clean.empty:
            dates.extend(pd.to_datetime(churn_clean["churn_date"]).dt.date.dropna().tolist())

        min_dt = min(dates) if dates else None
        max_dt = max(dates) if dates else None
        span = (max_dt - min_dt).days if (min_dt and max_dt) else 0

        total_rejected = sum(rejected_by_table.values())
        rejection_summary = RecordRejectionSummary(
            total_rejected=total_rejected,
            rejected_by_table=rejected_by_table,
            reasons=rejection_reasons,
        )

        # Propagate workspace_id to all canonical intermediate DataFrames
        acc_clean["workspace_id"] = workspace_id
        sub_clean["workspace_id"] = workspace_id
        if not usage_clean.empty:
            usage_clean["workspace_id"] = workspace_id
        if not tickets_clean.empty:
            tickets_clean["workspace_id"] = workspace_id
        if not churn_clean.empty:
            churn_clean["workspace_id"] = workspace_id

        return CanonicalIntermediateDataset(
            accounts_df=acc_clean,
            subscriptions_df=sub_clean,
            usage_df=usage_clean,
            tickets_df=tickets_clean,
            churn_events_df=churn_clean,
            rejection_summary=rejection_summary,
            warnings=warnings,
            min_date=min_dt,
            max_date=max_dt,
            span_days=span,
            derived_churn_from_subscriptions=derived_churn,
            workspace_id=workspace_id,
        )

    def _resolve_dataframe_for_concept(
        self,
        raw_dfs: dict[str, pd.DataFrame],
        concept_name: str,
        required_cols: list[str],
    ) -> pd.DataFrame:
        """Find the matching raw DataFrame by concept name or required column containment."""
        # 1. Exact or lowercase concept match
        for key, df in raw_dfs.items():
            if key.lower() == concept_name.lower():
                return df

        # 2. Match by columns
        for _key, df in raw_dfs.items():
            if all(col in df.columns for col in required_cols):
                return df

        # 3. Fallback: single DataFrame provided
        if len(raw_dfs) == 1:
            df = next(iter(raw_dfs.values()))
            missing = [c for c in required_cols if c not in df.columns]
            if not missing:
                return df

        raise ValueError(
            f"Could not find a dataset containing required columns for '{concept_name}': {required_cols}"
        )

    def _transform_accounts(self, df: pd.DataFrame, m: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Normalize Accounts table and prune invalid records."""
        reasons: list[str] = []
        initial_len = len(df)

        data = pd.DataFrame()
        data["account_id"] = df[m.account_id].astype(str).str.strip()
        data["signup_date"] = pd.to_datetime(df[m.signup_date], errors="coerce")

        # Attribute extraction
        data["industry"] = (
            df[m.industry].astype(str).str.strip() if m.industry and m.industry in df else "unknown"
        )
        data["country"] = (
            df[m.country].astype(str).str.strip() if m.country and m.country in df else "unknown"
        )
        data["referral_source"] = (
            df[m.referral_source].astype(str).str.strip()
            if m.referral_source and m.referral_source in df
            else "unknown"
        )
        data["seats"] = (
            pd.to_numeric(df[m.initial_seats], errors="coerce").fillna(1).astype(int)
            if m.initial_seats and m.initial_seats in df
            else 1
        )
        data["is_trial"] = (
            pd.to_numeric(df[m.is_trial], errors="coerce").fillna(0).astype(int)
            if m.is_trial and m.is_trial in df
            else 0
        )

        # Drop invalid accounts (empty ID or unparseable signup_date)
        valid_mask = (
            (data["account_id"] != "") & (data["account_id"] != "nan") & data["signup_date"].notna()
        )
        clean = data[valid_mask].copy()

        # Deduplicate on account_id
        clean = clean.drop_duplicates(subset=["account_id"], keep="first")
        rejected_count = initial_len - len(clean)

        if rejected_count > 0:
            reasons.append(
                f"{rejected_count:,} account rows were rejected due to empty account_id, duplicate account_id, or invalid signup date."
            )

        return clean, {"count": rejected_count, "reasons": reasons}

    def _transform_subscriptions(
        self, df: pd.DataFrame, m: Any, valid_account_ids: set[str]
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Normalize Subscriptions table and prune invalid/orphan contracts."""
        reasons: list[str] = []
        initial_len = len(df)

        data = pd.DataFrame()
        data["subscription_id"] = df[m.subscription_id].astype(str).str.strip()
        data["account_id"] = df[m.account_id].astype(str).str.strip()
        data["start_date"] = pd.to_datetime(df[m.start_date], errors="coerce")
        data["end_date"] = (
            pd.to_datetime(df[m.end_date], errors="coerce")
            if m.end_date and m.end_date in df
            else pd.NaT
        )

        # Financials and terms
        mrr_series = (
            pd.to_numeric(df[m.mrr], errors="coerce").fillna(0.0)
            if m.mrr and m.mrr in df
            else pd.Series(0.0, index=df.index)
        )
        data["mrr_amount"] = mrr_series.clip(lower=0.0)

        if m.arr and m.arr in df:
            arr_series = pd.to_numeric(df[m.arr], errors="coerce").fillna(0.0)
            data["arr_amount"] = arr_series.clip(lower=0.0)
        else:
            data["arr_amount"] = data["mrr_amount"] * 12.0

        data["seats"] = (
            pd.to_numeric(df[m.seats], errors="coerce").fillna(1).astype(int)
            if m.seats and m.seats in df
            else 1
        )
        data["billing_frequency"] = (
            df[m.billing_frequency].astype(str).str.lower().str.strip()
            if m.billing_frequency and m.billing_frequency in df
            else "monthly"
        )
        data["auto_renew_flag"] = (
            df[m.auto_renew].astype(str).str.lower().isin(["1", "true", "yes", "t"])
            if m.auto_renew and m.auto_renew in df
            else True
        )
        data["is_trial"] = (
            df[m.is_trial].astype(str).str.lower().isin(["1", "true", "yes", "t"])
            if m.is_trial and m.is_trial in df
            else False
        )
        data["upgrade_flag"] = False
        data["downgrade_flag"] = False

        # Drop invalid subscriptions and orphans
        valid_mask = (
            (data["subscription_id"] != "")
            & (data["subscription_id"] != "nan")
            & (data["account_id"] != "")
            & data["start_date"].notna()
            & data["account_id"].isin(valid_account_ids)
        )
        clean = data[valid_mask].copy()
        clean = clean.drop_duplicates(subset=["subscription_id"], keep="first")
        rejected_count = initial_len - len(clean)

        if rejected_count > 0:
            reasons.append(
                f"{rejected_count:,} subscription rows were rejected due to missing keys, unparseable start_date, or orphaned account_ids."
            )

        return clean, {"count": rejected_count, "reasons": reasons}

    def _transform_usage(
        self,
        df: pd.DataFrame,
        m: Any,
        subs_clean: pd.DataFrame,
        valid_account_ids: set[str],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Normalize Feature Usage table, linking to subscriptions and accounts."""
        reasons: list[str] = []
        initial_len = len(df)

        data = pd.DataFrame()
        data["usage_date"] = pd.to_datetime(df[m.event_time], errors="coerce")

        if m.subscription_id and m.subscription_id in df:
            data["subscription_id"] = df[m.subscription_id].astype(str).str.strip()
        elif m.account_id and m.account_id in df:
            # Synthetic link: map account to first active subscription of that account
            acc_to_sub = subs_clean.groupby("account_id")["subscription_id"].first().to_dict()
            raw_accs = df[m.account_id].astype(str).str.strip()
            data["subscription_id"] = raw_accs.map(acc_to_sub).fillna("")
        else:
            data["subscription_id"] = ""

        data["usage_id"] = (
            df[m.usage_id].astype(str).str.strip()
            if m.usage_id and m.usage_id in df
            else [f"use_{i}" for i in range(len(df))]
        )
        data["feature_name"] = (
            df[m.feature_name].astype(str).str.strip()
            if m.feature_name and m.feature_name in df
            else "core_feature"
        )
        data["usage_count"] = (
            pd.to_numeric(df[m.usage_count], errors="coerce").fillna(0).clip(lower=0).astype(int)
            if m.usage_count and m.usage_count in df
            else 1
        )
        data["usage_duration_secs"] = (
            pd.to_numeric(df[m.duration_secs], errors="coerce").fillna(0.0)
            if m.duration_secs and m.duration_secs in df
            else 0.0
        )
        data["error_count"] = (
            pd.to_numeric(df[m.is_error], errors="coerce").fillna(0).astype(int)
            if m.is_error and m.is_error in df
            else 0
        )
        data["active_users"] = 1
        data["is_beta_feature"] = (
            df[m.is_beta].astype(str).str.lower().isin(["1", "true", "yes", "t"])
            if m.is_beta and m.is_beta in df
            else False
        )

        valid_sub_ids = set(subs_clean["subscription_id"].unique())
        valid_mask = data["usage_date"].notna() & data["subscription_id"].isin(valid_sub_ids)
        clean = data[valid_mask].copy()
        rejected_count = initial_len - len(clean)

        if rejected_count > 0:
            reasons.append(
                f"{rejected_count:,} feature usage rows were rejected due to invalid dates or orphaned subscription references."
            )

        return clean, {"count": rejected_count, "reasons": reasons}

    def _transform_tickets(
        self, df: pd.DataFrame, m: Any, valid_account_ids: set[str]
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Normalize Support Tickets table."""
        reasons: list[str] = []
        initial_len = len(df)

        data = pd.DataFrame()
        data["ticket_id"] = df[m.ticket_id].astype(str).str.strip()
        data["account_id"] = df[m.account_id].astype(str).str.strip()
        data["submitted_at"] = pd.to_datetime(df[m.created_at], errors="coerce")
        data["closed_at"] = (
            pd.to_datetime(df[m.closed_at], errors="coerce")
            if m.closed_at and m.closed_at in df
            else pd.NaT
        )
        data["priority"] = (
            df[m.priority].astype(str).str.lower().str.strip()
            if m.priority and m.priority in df
            else "low"
        )
        data["satisfaction_score"] = (
            pd.to_numeric(df[m.satisfaction_score], errors="coerce")
            if m.satisfaction_score and m.satisfaction_score in df
            else None
        )
        data["escalation_flag"] = False
        data["first_response_time_minutes"] = -1.0

        valid_mask = (
            (data["ticket_id"] != "")
            & data["submitted_at"].notna()
            & data["account_id"].isin(valid_account_ids)
        )
        clean = data[valid_mask].copy()
        clean = clean.drop_duplicates(subset=["ticket_id"], keep="first")
        rejected_count = initial_len - len(clean)

        if rejected_count > 0:
            reasons.append(
                f"{rejected_count:,} support ticket rows were rejected due to missing keys, unparseable dates, or orphaned account references."
            )

        return clean, {"count": rejected_count, "reasons": reasons}

    def _transform_churn(
        self, df: pd.DataFrame, m: Any, valid_account_ids: set[str]
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Normalize Churn Events table."""
        reasons: list[str] = []
        initial_len = len(df)

        data = pd.DataFrame()
        data["account_id"] = df[m.account_id].astype(str).str.strip()
        data["churn_date"] = pd.to_datetime(df[m.churn_date], errors="coerce")
        data["churn_event_id"] = (
            df[m.churn_event_id].astype(str).str.strip()
            if m.churn_event_id and m.churn_event_id in df
            else [f"churn_{i}" for i in range(len(df))]
        )
        data["refund_amount_usd"] = 0.0

        valid_mask = (
            (data["account_id"] != "")
            & data["churn_date"].notna()
            & data["account_id"].isin(valid_account_ids)
        )
        clean = data[valid_mask].copy()
        rejected_count = initial_len - len(clean)

        if rejected_count > 0:
            reasons.append(
                f"{rejected_count:,} churn event rows were rejected due to missing IDs, invalid dates, or orphaned account references."
            )

        return clean, {"count": rejected_count, "reasons": reasons}

    @staticmethod
    def _derive_churn_from_subscriptions(subs_clean: pd.DataFrame) -> pd.DataFrame:
        """Derive canonical churn records from terminated subscriptions."""
        ended = subs_clean[subs_clean["end_date"].notna()].copy()
        if ended.empty:
            return pd.DataFrame(
                columns=["churn_event_id", "account_id", "churn_date", "refund_amount_usd"]
            )

        churn_df = pd.DataFrame()
        churn_df["churn_event_id"] = [f"derived_churn_{sid}" for sid in ended["subscription_id"]]
        churn_df["account_id"] = ended["account_id"].values
        churn_df["churn_date"] = ended["end_date"].values
        churn_df["refund_amount_usd"] = 0.0
        return churn_df

    @staticmethod
    def _empty_usage_df() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "usage_id",
                "subscription_id",
                "usage_date",
                "feature_name",
                "usage_count",
                "usage_duration_secs",
                "error_count",
                "active_users",
                "is_beta_feature",
            ]
        )

    @staticmethod
    def _empty_tickets_df() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "ticket_id",
                "account_id",
                "submitted_at",
                "closed_at",
                "priority",
                "satisfaction_score",
                "escalation_flag",
                "first_response_time_minutes",
            ]
        )

    @staticmethod
    def _empty_churn_df() -> pd.DataFrame:
        return pd.DataFrame(
            columns=["churn_event_id", "account_id", "churn_date", "refund_amount_usd"]
        )
