"""Feature usage point-in-time feature extraction.

Computes multi-window activity volumes (30d, 60d, 90d), durations, error rates,
distinct module usage, recency, and velocity trends strictly at snapshot date.
"""

from datetime import date
from typing import Any

import pandas as pd


class UsageFeatureExtractor:
    """Extracts product engagement and usage telemetry without lookahead leakage."""

    def __init__(
        self,
        usage_df: pd.DataFrame,
        subscriptions_df: pd.DataFrame,
    ) -> None:
        usage = usage_df.copy()
        if "account_id" in usage.columns:
            usage = usage.drop(columns=["account_id"])
        subs = subscriptions_df[["subscription_id", "account_id"]].drop_duplicates()

        # Link usage to accounts via subscriptions
        merged = usage.merge(subs, on="subscription_id", how="inner")
        merged["parsed_date"] = pd.to_datetime(merged["usage_date"])

        # Group by account_id for fast retrieval
        self._grouped = dict(iter(merged.groupby("account_id")))

    def extract_features(self, account_id: str, snapshot_date: date) -> dict[str, Any]:
        """Extract usage features for an account at snapshot date."""
        acc_usage = self._grouped.get(account_id)
        if acc_usage is None or acc_usage.empty:
            return self._empty_features()

        snap_ts = pd.to_datetime(snapshot_date)

        # Filter strictly to historical telemetry on or before snapshot date
        hist = acc_usage[acc_usage["parsed_date"] <= snap_ts]
        if hist.empty:
            return self._empty_features()

        # Recency calculation
        latest_date = hist["parsed_date"].max().date()
        days_since_last = float(max(0, (snapshot_date - latest_date).days))
        has_history = 1

        # Lookback window filters
        d30 = snap_ts - pd.Timedelta(days=30)
        d60 = snap_ts - pd.Timedelta(days=60)
        d90 = snap_ts - pd.Timedelta(days=90)

        w30 = hist[hist["parsed_date"] > d30]
        w60 = hist[hist["parsed_date"] > d60]
        w90 = hist[hist["parsed_date"] > d90]
        w_prev30 = hist[(hist["parsed_date"] > d60) & (hist["parsed_date"] <= d30)]

        # Window aggregates
        u_count_30d = int(w30["usage_count"].sum())
        u_count_60d = int(w60["usage_count"].sum())
        u_count_90d = int(w90["usage_count"].sum())

        dur_30d = float(w30["usage_duration_secs"].sum())
        dur_90d = float(w90["usage_duration_secs"].sum())

        err_30d = int(w30["error_count"].sum())
        err_rate_30d = float(err_30d / u_count_30d) if u_count_30d > 0 else 0.0

        active_days_30d = int(w30["parsed_date"].dt.date.nunique())
        active_days_90d = int(w90["parsed_date"].dt.date.nunique())

        distinct_features_30d = int(w30["feature_name"].nunique())
        beta_count_30d = int(
            w30[w30["is_beta_feature"] == True]["usage_count"].sum()  # noqa: E712
        )

        # Velocity / trend calculation
        prev_count = float(w_prev30["usage_count"].sum())
        if u_count_30d == 0 and prev_count == 0:
            trend_pct = 0.0
        else:
            raw_trend = (float(u_count_30d) - prev_count) / max(prev_count, 1.0)
            trend_pct = float(max(-1.0, min(raw_trend, 5.0)))

        return {
            "usage_count_30d": u_count_30d,
            "usage_count_60d": u_count_60d,
            "usage_count_90d": u_count_90d,
            "usage_duration_secs_30d": dur_30d,
            "usage_duration_secs_90d": dur_90d,
            "usage_error_count_30d": err_30d,
            "usage_error_rate_30d": err_rate_30d,
            "usage_active_days_30d": active_days_30d,
            "usage_active_days_90d": active_days_90d,
            "usage_distinct_features_30d": distinct_features_30d,
            "usage_beta_feature_count_30d": beta_count_30d,
            "usage_days_since_last": days_since_last,
            "usage_has_history": has_history,
            "usage_trend_pct_30d": trend_pct,
        }

    @staticmethod
    def _empty_features() -> dict[str, Any]:
        return {
            "usage_count_30d": 0,
            "usage_count_60d": 0,
            "usage_count_90d": 0,
            "usage_duration_secs_30d": 0.0,
            "usage_duration_secs_90d": 0.0,
            "usage_error_count_30d": 0,
            "usage_error_rate_30d": 0.0,
            "usage_active_days_30d": 0,
            "usage_active_days_90d": 0,
            "usage_distinct_features_30d": 0,
            "usage_beta_feature_count_30d": 0,
            "usage_days_since_last": -1.0,
            "usage_has_history": 0,
            "usage_trend_pct_30d": 0.0,
        }
