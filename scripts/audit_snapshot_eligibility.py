"""Script to comprehensively audit snapshot eligibility policies, cohort counts,
and label reconciliation between Phase 2 and Phase 4.
"""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd


def audit_dataset():
    data_dir = Path("data/raw/ravenstack")
    accounts = pd.read_csv(data_dir / "ravenstack_accounts.csv")
    subscriptions = pd.read_csv(data_dir / "ravenstack_subscriptions.csv")
    churn_events = pd.read_csv(data_dir / "ravenstack_churn_events.csv")

    accounts["parsed_signup"] = pd.to_datetime(accounts["signup_date"]).dt.date
    subscriptions["parsed_start"] = pd.to_datetime(subscriptions["start_date"]).dt.date
    subscriptions["parsed_end"] = pd.to_datetime(subscriptions["end_date"]).dt.date
    churn_events["parsed_churn"] = pd.to_datetime(churn_events["churn_date"]).dt.date

    print(f"Total Accounts: {len(accounts)}")
    print(f"Total Subscriptions: {len(subscriptions)}")
    print(f"Total Churn Events: {len(churn_events)}")

    # Dates
    dates = [d.date() for d in pd.date_range(start="2023-04-01", end="2024-11-01", freq="MS")]
    print(f"Total snapshot dates: {len(dates)} (from {dates[0]} to {dates[-1]})")

    print("\n" + "=" * 90)
    print("RECONCILING PHASE 2 VS PHASE 4 ON EACH DATE")
    print("=" * 90)

    for snap in dates:
        # Phase 2 simulation logic:
        # 1. Accounts signed up <= snap
        p2_signed_up = set(accounts[accounts["parsed_signup"] <= snap]["account_id"])
        # 2. Prior churns <= snap
        p2_prior_churns = set(churn_events[churn_events["parsed_churn"] <= snap]["account_id"])
        # 3. Active in Phase 2
        p2_active = p2_signed_up - p2_prior_churns

        # Phase 4 subscription_active logic:
        # 1. Accounts signed up <= snap
        # 2. Subscriptions active at snap: start_date <= snap and (end_date is null or end_date > snap)
        active_subs = subscriptions[
            (subscriptions["parsed_start"] <= snap)
            & (subscriptions["parsed_end"].isnull() | (subscriptions["parsed_end"] > snap))
        ]
        p4_sub_active = p2_signed_up & set(active_subs["account_id"].unique())

        # Phase 4 churn_free logic:
        # 1. Accounts signed up <= snap
        # 2. Account has never churned <= snap
        p4_churn_free = p2_signed_up - p2_prior_churns

        # Target 60d for p4_sub_active:
        window_end = snap + timedelta(days=60)
        p4_sub_churners = set()
        for acc in p4_sub_active:
            acc_c = churn_events[
                (churn_events["account_id"] == acc)
                & (churn_events["parsed_churn"] > snap)
                & (churn_events["parsed_churn"] <= window_end)
            ]
            if len(acc_c) > 0:
                p4_sub_churners.add(acc)

        # Target 60d for p4_churn_free:
        p4_cf_churners = set()
        for acc in p4_churn_free:
            acc_c = churn_events[
                (churn_events["account_id"] == acc)
                & (churn_events["parsed_churn"] > snap)
                & (churn_events["parsed_churn"] <= window_end)
            ]
            if len(acc_c) > 0:
                p4_cf_churners.add(acc)

        # Discrepancy between p2_active and p4_sub_active on 2023-04-01:
        if snap == date(2023, 4, 1):
            print(f"\nDetailed Breakdown for {snap}:")
            print(f"  P2 signed up: {len(p2_signed_up)}")
            print(f"  P2 prior churns: {len(p2_prior_churns)}")
            print(f"  P2 active (signed up - prior churns): {len(p2_active)}")
            print(f"  P4 sub_active: {len(p4_sub_active)}")
            print(f"  In P2 active but NOT in P4 sub_active: {len(p2_active - p4_sub_active)}")
            print(f"  In P4 sub_active but NOT in P2 active: {len(p4_sub_active - p2_active)}")

            # Inspect those in P2 active but NOT in P4 sub_active
            diff_accs = list(p2_active - p4_sub_active)
            print(f"  Sample accounts in P2 active but not in P4 sub_active: {diff_accs[:5]}")
            for d_acc in diff_accs[:3]:
                d_acc_subs = subscriptions[subscriptions["account_id"] == d_acc]
                print(
                    f"    Account {d_acc} signup: {accounts[accounts['account_id'] == d_acc]['signup_date'].values[0]}"
                )
                print(f"    Subscriptions for {d_acc}:")
                for _, srow in d_acc_subs.iterrows():
                    print(
                        f"      Sub {srow['subscription_id']}: start={srow['start_date']}, end={srow['end_date']}"
                    )

    # Let's print table comparing all dates
    print("\n" + "=" * 90)
    print(
        f"{'Date':<12} | {'P2 Active':<10} {'P2 Churn':<10} | {'SubActive':<10} {'SA Churn':<10} | {'ChurnFree':<10} {'CF Churn':<10}"
    )
    print("-" * 90)
    total_sa = 0
    total_sa_churn = 0
    total_cf = 0
    total_cf_churn = 0
    for snap in dates:
        p2_signed_up = set(accounts[accounts["parsed_signup"] <= snap]["account_id"])
        p2_prior_churns = set(churn_events[churn_events["parsed_churn"] <= snap]["account_id"])
        p2_active = p2_signed_up - p2_prior_churns

        window_end = snap + timedelta(days=60)
        p2_churners = set(
            churn_events[
                (churn_events["parsed_churn"] > snap) & (churn_events["parsed_churn"] <= window_end)
            ]["account_id"]
        )
        p2_churn_count = len(p2_active & p2_churners)

        active_subs = subscriptions[
            (subscriptions["parsed_start"] <= snap)
            & (subscriptions["parsed_end"].isnull() | (subscriptions["parsed_end"] > snap))
        ]
        p4_sub_active = p2_signed_up & set(active_subs["account_id"].unique())
        p4_sa_churn_count = len(p4_sub_active & p2_churners)

        p4_churn_free = p2_signed_up - p2_prior_churns
        p4_cf_churn_count = len(p4_churn_free & p2_churners)

        total_sa += len(p4_sub_active)
        total_sa_churn += p4_sa_churn_count
        total_cf += len(p4_churn_free)
        total_cf_churn += p4_cf_churn_count

        print(
            f"{str(snap):<12} | {len(p2_active):<10} {p2_churn_count:<10} | {len(p4_sub_active):<10} {p4_sa_churn_count:<10} | {len(p4_churn_free):<10} {p4_cf_churn_count:<10}"
        )

    print("-" * 90)
    print(
        f"{'TOTAL':<12} | {'-':<10} {'-':<10} | {total_sa:<10} {total_sa_churn:<10} | {total_cf:<10} {total_cf_churn:<10}"
    )


if __name__ == "__main__":
    audit_dataset()
