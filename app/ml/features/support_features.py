"""Support ticket point-in-time feature extraction.

Computes multi-window ticket volumes (30d, 90d, lifetime), open/unresolved ticket counts,
priority breakdown, escalations, response times, and satisfaction scores at snapshot date.
"""

from datetime import date
from typing import Any

import pandas as pd


class SupportFeatureExtractor:
    """Extracts support engagement and SLA telemetry without lookahead leakage."""

    def __init__(self, tickets_df: pd.DataFrame) -> None:
        tickets = tickets_df.copy()
        tickets["parsed_submitted"] = pd.to_datetime(
            tickets["submitted_at"], format="mixed", errors="coerce"
        )
        tickets["parsed_closed"] = pd.to_datetime(
            tickets["closed_at"], format="mixed", errors="coerce"
        )
        tickets["priority_clean"] = tickets["priority"].astype(str).str.lower()

        # Group by account_id for fast retrieval
        self._grouped = dict(iter(tickets.groupby("account_id")))

    def extract_features(self, account_id: str, snapshot_date: date) -> dict[str, Any]:
        """Extract support features for an account at snapshot date."""
        acc_tickets = self._grouped.get(account_id)
        if acc_tickets is None or acc_tickets.empty:
            return self._empty_features()

        snap_ts = pd.to_datetime(snapshot_date)

        # Filter strictly to tickets submitted on or before snapshot date
        hist = acc_tickets[acc_tickets["parsed_submitted"] <= snap_ts]
        if hist.empty:
            return self._empty_features()

        tickets_lifetime = len(hist)

        # Unresolved tickets at snapshot date:
        # submitted <= snapshot_date AND (closed_at IS NULL OR closed_at > snapshot_date)
        unresolved = hist[hist["parsed_closed"].isnull() | (hist["parsed_closed"] > snap_ts)]
        unresolved_count = len(unresolved)
        has_unresolved = 1 if unresolved_count > 0 else 0

        # Lookback window filters
        d30 = snap_ts - pd.Timedelta(days=30)
        d90 = snap_ts - pd.Timedelta(days=90)

        w30 = hist[hist["parsed_submitted"] > d30]
        w90 = hist[hist["parsed_submitted"] > d90]

        t_count_30d = len(w30)
        t_count_90d = len(w90)

        # Priority and escalation metrics in 90d window
        urgent_90d = int((w90["priority_clean"] == "urgent").sum())
        high_or_urgent_90d = int(w90["priority_clean"].isin(["high", "urgent"]).sum())
        escalated_90d = int((w90["escalation_flag"] == True).sum())  # noqa: E712

        # Satisfaction scores in 90d window
        has_unrated_90d = 1 if (w90["satisfaction_score"].isnull()).any() else 0
        rated_90d = w90[w90["satisfaction_score"].notnull()]
        if not rated_90d.empty:
            avg_satisfaction_90d = float(rated_90d["satisfaction_score"].mean())
        else:
            avg_satisfaction_90d = -1.0  # Sentinel paired with has_unrated indicator

        # Response times in 90d window
        if not w90.empty:
            avg_first_resp_90d = float(w90["first_response_time_minutes"].mean())
        else:
            avg_first_resp_90d = -1.0

        return {
            "tickets_count_30d": t_count_30d,
            "tickets_count_90d": t_count_90d,
            "tickets_count_lifetime": tickets_lifetime,
            "tickets_unresolved_count": unresolved_count,
            "tickets_has_unresolved": has_unresolved,
            "tickets_urgent_count_90d": urgent_90d,
            "tickets_high_or_urgent_count_90d": high_or_urgent_90d,
            "tickets_escalated_count_90d": escalated_90d,
            "tickets_avg_satisfaction_90d": avg_satisfaction_90d,
            "tickets_has_unrated_90d": has_unrated_90d,
            "tickets_avg_first_response_minutes_90d": avg_first_resp_90d,
        }

    @staticmethod
    def _empty_features() -> dict[str, Any]:
        return {
            "tickets_count_30d": 0,
            "tickets_count_90d": 0,
            "tickets_count_lifetime": 0,
            "tickets_unresolved_count": 0,
            "tickets_has_unresolved": 0,
            "tickets_urgent_count_90d": 0,
            "tickets_high_or_urgent_count_90d": 0,
            "tickets_escalated_count_90d": 0,
            "tickets_avg_satisfaction_90d": -1.0,
            "tickets_has_unrated_90d": 0,
            "tickets_avg_first_response_minutes_90d": -1.0,
        }
