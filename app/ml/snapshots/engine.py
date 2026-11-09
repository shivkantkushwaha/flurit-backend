"""Point-in-Time Customer Snapshot Generation Engine.

Produces deterministic customer snapshots across approved observation dates,
verifies point-in-time eligibility, and labels future 60-day churn targets
without data leakage.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd

from app.ml.snapshots.targets import TargetConstructor


@dataclass(frozen=True)
class SnapshotRecord:
    """Immutable representation of a point-in-time customer snapshot."""

    account_id: str
    snapshot_date: date
    is_active: bool
    target_churn_60d: int | None


class SnapshotEngine:
    """Generates time-aware customer snapshots with leakage-safe target labeling."""

    def __init__(
        self,
        accounts_df: pd.DataFrame,
        subscriptions_df: pd.DataFrame,
        churn_events_df: pd.DataFrame,
        prediction_window_days: int = 60,
        eligibility_policy: Literal["subscription_active", "churn_free"] = "subscription_active",
        global_max_date: date | None = None,
    ) -> None:
        self.accounts = accounts_df.copy()
        self.subscriptions = subscriptions_df.copy()
        self.churn_events = churn_events_df.copy()
        self.prediction_window_days = prediction_window_days
        self.policy = eligibility_policy

        # Parse date columns
        self.accounts["parsed_signup_date"] = pd.to_datetime(self.accounts["signup_date"])
        self.subscriptions["parsed_start_date"] = pd.to_datetime(self.subscriptions["start_date"])
        self.subscriptions["parsed_end_date"] = pd.to_datetime(self.subscriptions["end_date"])
        self.churn_events["parsed_churn_date"] = pd.to_datetime(self.churn_events["churn_date"])

        # Determine global max date across relevant tables
        if global_max_date is not None:
            self.global_max_date = global_max_date
        else:
            candidates = []
            if (
                not self.subscriptions.empty
                and self.subscriptions["parsed_start_date"].notnull().any()
            ):
                candidates.append(self.subscriptions["parsed_start_date"].max())
            if (
                not self.churn_events.empty
                and self.churn_events["parsed_churn_date"].notnull().any()
            ):
                candidates.append(self.churn_events["parsed_churn_date"].max())
            if not self.accounts.empty and self.accounts["parsed_signup_date"].notnull().any():
                candidates.append(self.accounts["parsed_signup_date"].max())
            self.global_max_date = max(candidates).date() if candidates else date.today()

        # Initialize target constructor
        self.target_constructor = TargetConstructor(
            prediction_window_days=prediction_window_days,
            global_max_date=self.global_max_date,
        )

    @classmethod
    def from_csv_directory(
        cls,
        data_dir: Path,
        prediction_window_days: int = 60,
        eligibility_policy: Literal["subscription_active", "churn_free"] = "subscription_active",
        global_max_date: date | None = None,
    ) -> "SnapshotEngine":
        """Instantiate engine by reading raw CSV files from directory."""
        acc_df = pd.read_csv(data_dir / "ravenstack_accounts.csv")
        sub_df = pd.read_csv(data_dir / "ravenstack_subscriptions.csv")
        churn_df = pd.read_csv(data_dir / "ravenstack_churn_events.csv")
        return cls(
            accounts_df=acc_df,
            subscriptions_df=sub_df,
            churn_events_df=churn_df,
            prediction_window_days=prediction_window_days,
            eligibility_policy=eligibility_policy,
            global_max_date=global_max_date,
        )

    def get_eligible_accounts(self, snapshot_date: date) -> list[str]:
        """Determine accounts eligible for scoring at snapshot date using point-in-time info only."""
        snap_dt = pd.to_datetime(snapshot_date)

        # 1. Account must exist on or before snapshot date
        signed_up = set(
            self.accounts[self.accounts["parsed_signup_date"] <= snap_dt]["account_id"].unique()
        )

        if self.policy == "subscription_active":
            # Account must have at least one active subscription at snapshot date
            # start_date <= snapshot_date AND (end_date IS NULL OR end_date > snapshot_date)
            active_subs = self.subscriptions[
                (self.subscriptions["parsed_start_date"] <= snap_dt)
                & (
                    self.subscriptions["parsed_end_date"].isnull()
                    | (self.subscriptions["parsed_end_date"] > snap_dt)
                )
            ]
            active_sub_accounts = set(active_subs["account_id"].unique())
            eligible = sorted(signed_up & active_sub_accounts)

        elif self.policy == "churn_free":
            # Account signed up and has never had a churn event prior to or on snapshot date
            prior_churns = set(
                self.churn_events[self.churn_events["parsed_churn_date"] <= snap_dt][
                    "account_id"
                ].unique()
            )
            eligible = sorted(signed_up - prior_churns)

        else:
            raise ValueError(f"Unknown eligibility policy: {self.policy}")

        return eligible

    def generate_snapshots_for_date(self, snapshot_date: date) -> list[SnapshotRecord]:
        """Generate snapshot records for a single observation date."""
        eligible_accounts = self.get_eligible_accounts(snapshot_date)
        records: list[SnapshotRecord] = []

        for acc_id in eligible_accounts:
            target = self.target_constructor.compute_target(
                account_id=acc_id,
                snapshot_date=snapshot_date,
                churn_events_df=self.churn_events,
            )
            record = SnapshotRecord(
                account_id=acc_id,
                snapshot_date=snapshot_date,
                is_active=True,
                target_churn_60d=target,
            )
            records.append(record)

        return records

    def generate_monthly_snapshots(
        self,
        start_date: date = date(2023, 4, 1),
        end_date: date | None = None,
        observation_dates: Sequence[date] | None = None,
    ) -> pd.DataFrame:
        """Generate deterministic customer snapshots across all observation dates.

        Args:
            start_date: First monthly observation date (default: 2023-04-01).
            end_date: Last observation date (default: cutoff date T_max - 60d).
            observation_dates: Optional explicit list of dates to evaluate.

        Returns:
            pd.DataFrame containing generated customer snapshots.
        """
        if observation_dates is not None:
            dates_to_process = list(observation_dates)
        else:
            target_end = end_date or self.target_constructor.cutoff_date or self.global_max_date
            ts_dates = pd.date_range(start=start_date, end=target_end, freq="MS")
            dates_to_process = [ts.date() for ts in ts_dates]

        all_records: list[SnapshotRecord] = []
        for obs_date in dates_to_process:
            cohort_records = self.generate_snapshots_for_date(obs_date)
            all_records.extend(cohort_records)

        # Convert to DataFrame
        data = [
            {
                "account_id": r.account_id,
                "snapshot_date": r.snapshot_date,
                "is_active": r.is_active,
                "target_churn_60d": r.target_churn_60d,
            }
            for r in all_records
        ]
        df = pd.DataFrame(data)

        if not df.empty:
            # Sort deterministically
            df = df.sort_values(["snapshot_date", "account_id"]).reset_index(drop=True)

            # Assert uniqueness constraint (account_id, snapshot_date)
            dups = df.duplicated(subset=["account_id", "snapshot_date"]).sum()
            if dups > 0:
                raise ValueError(
                    f"Integrity violation: {dups} duplicate (account_id, snapshot_date) snapshot pairs generated."
                )

        return df
