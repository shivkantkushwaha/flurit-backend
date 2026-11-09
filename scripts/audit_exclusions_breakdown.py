"""Detailed audit script to compute:
1. Exact exclusion reasons by snapshot date for subscription_active.
2. Exact exclusion reasons by snapshot date for churn_free.
3. Cases of:
   - Accounts with churn events but active subscriptions.
   - Accounts without active subscriptions but no qualifying churn event.
4. Churners receiving target 1 under both policies.
"""

from datetime import timedelta
from pathlib import Path

import pandas as pd


def audit_exclusions():
    data_dir = Path("data/raw/ravenstack")
    accounts = pd.read_csv(data_dir / "ravenstack_accounts.csv")
    subscriptions = pd.read_csv(data_dir / "ravenstack_subscriptions.csv")
    churn_events = pd.read_csv(data_dir / "ravenstack_churn_events.csv")

    accounts["parsed_signup"] = pd.to_datetime(accounts["signup_date"]).dt.date
    subscriptions["parsed_start"] = pd.to_datetime(subscriptions["start_date"]).dt.date
    subscriptions["parsed_end"] = pd.to_datetime(subscriptions["end_date"]).dt.date
    churn_events["parsed_churn"] = pd.to_datetime(churn_events["churn_date"]).dt.date

    dates = [d.date() for d in pd.date_range(start="2023-04-01", end="2024-11-01", freq="MS")]
    all_accounts = set(accounts["account_id"].unique())
    total_acc_count = len(all_accounts)

    print("=" * 110)
    print("POLICY 1: subscription_active DETAILED CENSUS & EXCLUSION REASONS")
    print("=" * 110)
    print(
        f"{'Date':<11} | {'Eligible':<8} {'Tgt=1':<6} {'Tgt=0':<6} | {'Total Excl':<10} {'NotSignedUp':<12} {'NoSubStarted':<13} {'AllSubsEnded':<12}"
    )
    print("-" * 110)

    for snap in dates:
        # Eligible under subscription_active
        signed_up = set(accounts[accounts["parsed_signup"] <= snap]["account_id"])
        not_signed_up = all_accounts - signed_up

        # Subscriptions active at snap:
        active_subs = subscriptions[
            (subscriptions["parsed_start"] <= snap)
            & (subscriptions["parsed_end"].isnull() | (subscriptions["parsed_end"] > snap))
        ]
        active_sub_accounts = set(active_subs["account_id"].unique())

        eligible = signed_up & active_sub_accounts

        # Reasons for signed_up accounts that are NOT eligible:
        signed_up_not_eligible = signed_up - active_sub_accounts

        # 1. Has any subscription started <= snap?
        started_subs = subscriptions[subscriptions["parsed_start"] <= snap]
        started_sub_accounts = set(started_subs["account_id"].unique())

        no_sub_started = signed_up - started_sub_accounts
        all_subs_ended = signed_up_not_eligible - no_sub_started

        # Targets
        window_end = snap + timedelta(days=60)
        in_win_churners = set(
            churn_events[
                (churn_events["parsed_churn"] > snap) & (churn_events["parsed_churn"] <= window_end)
            ]["account_id"]
        )
        tgt1 = len(eligible & in_win_churners)
        tgt0 = len(eligible) - tgt1
        total_excl = total_acc_count - len(eligible)

        print(
            f"{str(snap):<11} | {len(eligible):<8} {tgt1:<6} {tgt0:<6} | {total_excl:<10} {len(not_signed_up):<12} {len(no_sub_started):<13} {len(all_subs_ended):<12}"
        )

    print("\n" + "=" * 110)
    print("POLICY 2: churn_free DETAILED CENSUS & EXCLUSION REASONS")
    print("=" * 110)
    print(
        f"{'Date':<11} | {'Eligible':<8} {'Tgt=1':<6} {'Tgt=0':<6} | {'Total Excl':<10} {'NotSignedUp':<12} {'PriorChurn':<12}"
    )
    print("-" * 110)

    for snap in dates:
        signed_up = set(accounts[accounts["parsed_signup"] <= snap]["account_id"])
        not_signed_up = all_accounts - signed_up

        prior_churns = set(churn_events[churn_events["parsed_churn"] <= snap]["account_id"])

        eligible = signed_up - prior_churns
        total_excl = total_acc_count - len(eligible)

        window_end = snap + timedelta(days=60)
        in_win_churners = set(
            churn_events[
                (churn_events["parsed_churn"] > snap) & (churn_events["parsed_churn"] <= window_end)
            ]["account_id"]
        )
        tgt1 = len(eligible & in_win_churners)
        tgt0 = len(eligible) - tgt1

        # Accounts excluded due to prior churn (who were signed up)
        excl_prior_churn = len(signed_up & prior_churns)

        print(
            f"{str(snap):<11} | {len(eligible):<8} {tgt1:<6} {tgt0:<6} | {total_excl:<10} {len(not_signed_up):<12} {excl_prior_churn:<12}"
        )

    print("\n" + "=" * 110)
    print("TASK 4 AUDIT: ACCOUNTS WITH CHURN EVENTS VS SUBSCRIPTIONS")
    print("=" * 110)
    # How many accounts in subscription_active have had a PRIOR churn event?
    prior_churn_in_sub_active = 0
    total_sa_snapshots = 0
    for snap in dates:
        signed_up = set(accounts[accounts["parsed_signup"] <= snap]["account_id"])
        active_subs = subscriptions[
            (subscriptions["parsed_start"] <= snap)
            & (subscriptions["parsed_end"].isnull() | (subscriptions["parsed_end"] > snap))
        ]
        eligible = signed_up & set(active_subs["account_id"].unique())
        prior_churns = set(churn_events[churn_events["parsed_churn"] <= snap]["account_id"])

        overlap = eligible & prior_churns
        prior_churn_in_sub_active += len(overlap)
        total_sa_snapshots += len(eligible)

    print(f"Total snapshots under subscription_active: {total_sa_snapshots}")
    print(
        f"Snapshots where account had a PRIOR churn event: {prior_churn_in_sub_active} ({prior_churn_in_sub_active / total_sa_snapshots * 100:.1f}%)"
    )

    # How many accounts in churn_free DO NOT have an active subscription at snapshot date?
    no_active_sub_in_churn_free = 0
    total_cf_snapshots = 0
    for snap in dates:
        signed_up = set(accounts[accounts["parsed_signup"] <= snap]["account_id"])
        prior_churns = set(churn_events[churn_events["parsed_churn"] <= snap]["account_id"])
        eligible = signed_up - prior_churns

        active_subs = subscriptions[
            (subscriptions["parsed_start"] <= snap)
            & (subscriptions["parsed_end"].isnull() | (subscriptions["parsed_end"] > snap))
        ]
        active_sub_accs = set(active_subs["account_id"].unique())

        overlap_no_sub = eligible - active_sub_accs
        no_active_sub_in_churn_free += len(overlap_no_sub)
        total_cf_snapshots += len(eligible)

    print(f"\nTotal snapshots under churn_free: {total_cf_snapshots}")
    print(
        f"Snapshots where account had NO active subscription at snapshot date: {no_active_sub_in_churn_free} ({no_active_sub_in_churn_free / total_cf_snapshots * 100:.1f}%)"
    )


if __name__ == "__main__":
    audit_exclusions()
