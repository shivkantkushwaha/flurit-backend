"""Point-in-time Feature Pipeline Orchestrator.

Combines modular feature extractors across account, subscription, usage,
support, and churn domains to produce deterministic, leakage-safe feature matrices.
"""

from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.features.account_features import AccountFeatureExtractor
from app.ml.features.churn_features import ChurnFeatureExtractor
from app.ml.features.schema import (
    FEATURE_DEFINITIONS,
    FEATURE_NAMES,
    STRICTLY_EXCLUDED_FIELDS,
)
from app.ml.features.subscription_features import SubscriptionFeatureExtractor
from app.ml.features.support_features import SupportFeatureExtractor
from app.ml.features.usage_features import UsageFeatureExtractor


class FeaturePipeline:
    """Orchestrates point-in-time feature engineering for Flurit customer snapshots."""

    def __init__(
        self,
        accounts_df: pd.DataFrame,
        subscriptions_df: pd.DataFrame,
        usage_df: pd.DataFrame,
        tickets_df: pd.DataFrame,
        churn_events_df: pd.DataFrame,
    ) -> None:
        self.account_extractor = AccountFeatureExtractor(accounts_df)
        self.subscription_extractor = SubscriptionFeatureExtractor(subscriptions_df)
        self.usage_extractor = UsageFeatureExtractor(usage_df, subscriptions_df)
        self.support_extractor = SupportFeatureExtractor(tickets_df)
        self.churn_extractor = ChurnFeatureExtractor(churn_events_df)

    @classmethod
    def from_csv_directory(cls, data_dir: Path) -> "FeaturePipeline":
        """Instantiate pipeline directly by reading all raw CSV tables from directory."""
        acc_df = pd.read_csv(data_dir / "ravenstack_accounts.csv")
        sub_df = pd.read_csv(data_dir / "ravenstack_subscriptions.csv")
        usage_df = pd.read_csv(data_dir / "ravenstack_feature_usage.csv")
        tickets_df = pd.read_csv(data_dir / "ravenstack_support_tickets.csv")
        churn_df = pd.read_csv(data_dir / "ravenstack_churn_events.csv")
        return cls(
            accounts_df=acc_df,
            subscriptions_df=sub_df,
            usage_df=usage_df,
            tickets_df=tickets_df,
            churn_events_df=churn_df,
        )

    def extract_snapshot_features(
        self,
        account_id: str,
        snapshot_date: date,
    ) -> dict[str, Any]:
        """Extract a single complete point-in-time feature vector for an account."""
        features: dict[str, Any] = {}
        features.update(self.account_extractor.extract_features(account_id, snapshot_date))
        features.update(self.subscription_extractor.extract_features(account_id, snapshot_date))
        features.update(self.usage_extractor.extract_features(account_id, snapshot_date))
        features.update(self.support_extractor.extract_features(account_id, snapshot_date))
        features.update(self.churn_extractor.extract_features(account_id, snapshot_date))
        return features

    def build_feature_matrix(
        self,
        snapshots: pd.DataFrame | Sequence[tuple[str, date]],
        include_metadata: bool = True,
    ) -> pd.DataFrame:
        """Build a deterministic point-in-time feature matrix for snapshot records.

        Args:
            snapshots: DataFrame containing ['account_id', 'snapshot_date'] (and optional target),
                       or a sequence of (account_id, snapshot_date) tuples.
            include_metadata: If True, prepends account_id and snapshot_date, and appends
                              target_churn_60d if present in input DataFrame.

        Returns:
            pd.DataFrame containing feature matrix aligned to canonical FEATURE_NAMES schema.
        """
        rows: list[dict[str, Any]] = []

        if isinstance(snapshots, pd.DataFrame):
            records = snapshots.to_dict(orient="records")
            for rec in records:
                acc_id = str(rec["account_id"])
                snap_dt = (
                    rec["snapshot_date"]
                    if isinstance(rec["snapshot_date"], date)
                    else pd.to_datetime(rec["snapshot_date"]).date()
                )
                feat = self.extract_snapshot_features(acc_id, snap_dt)

                if include_metadata:
                    row: dict[str, Any] = {
                        "account_id": acc_id,
                        "snapshot_date": snap_dt,
                    }
                    row.update(feat)
                    if "target_churn_60d" in rec:
                        row["target_churn_60d"] = rec["target_churn_60d"]
                    rows.append(row)
                else:
                    rows.append(feat)
        else:
            for acc_id, snap_dt in snapshots:
                feat = self.extract_snapshot_features(acc_id, snap_dt)
                if include_metadata:
                    row = {
                        "account_id": acc_id,
                        "snapshot_date": snap_dt,
                    }
                    row.update(feat)
                    rows.append(row)
                else:
                    rows.append(feat)

        df = pd.DataFrame(rows)

        # Enforce canonical column ordering and data types
        metadata_cols = ["account_id", "snapshot_date"] if include_metadata else []
        target_cols = (
            ["target_churn_60d"] if include_metadata and "target_churn_60d" in df.columns else []
        )
        ordered_cols = metadata_cols + FEATURE_NAMES + target_cols
        df = df[ordered_cols]

        # Apply schema data types
        for f_def in FEATURE_DEFINITIONS:
            if f_def.name in df.columns:
                if f_def.dtype == "category":
                    df[f_def.name] = df[f_def.name].astype("category")
                elif f_def.dtype == "int64":
                    df[f_def.name] = df[f_def.name].astype("int64")
                elif f_def.dtype == "float64":
                    df[f_def.name] = df[f_def.name].astype("float64")

        return df

    @staticmethod
    def verify_no_leakage(features_df: pd.DataFrame) -> None:
        """Validate feature matrix against prohibited fields and schema invariants."""
        # Check against strictly excluded field names
        cols_lower = {c.lower() for c in features_df.columns if c != "target_churn_60d"}
        for prohibited in STRICTLY_EXCLUDED_FIELDS:
            if prohibited.lower() in cols_lower:
                raise ValueError(
                    f"DATA LEAKAGE DETECTED: Prohibited column '{prohibited}' found in feature matrix!"
                )

        # Ensure all 45 feature columns exist
        missing_features = [f for f in FEATURE_NAMES if f not in features_df.columns]
        if missing_features:
            raise ValueError(f"Missing required feature columns: {missing_features}")
