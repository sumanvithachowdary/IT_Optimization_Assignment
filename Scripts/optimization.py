import pandas as pd

INPUT_PATH = "IT_Optimization_Assignment/Data/cleaned_tickets.csv"
OUTPUT_PATH = "IT_Optimization_Assignment/Data/optimization_summary.txt"

ANNUAL_TICKETS = 2_000_000
AUTOMATION_COVERAGE = 0.30
AUTOMATION_COST_REDUCTION = 0.50
NETWORK_TIME_REDUCTION = 0.20
ESCALATION_REDUCTION = 0.25
TARGET_COST_REDUCTION = 0.20

def main():
    df = pd.read_csv(INPUT_PATH)
    sample_tickets = len(df)

    if sample_tickets == 0:
        raise ValueError("Dataset contains no records.")

    # Annualize development dataset
    scale = ANNUAL_TICKETS / sample_tickets
    current_cost = df["total_cost"].sum() * scale
    target_savings = current_cost * TARGET_COST_REDUCTION

    # Automation savings for simple software/access tickets
    automation = (
        df["issue_category"].isin(["Software", "Account/Access"])
        & df["priority"].isin(["Low", "Medium"])
    )
    automation_cost = (
        df.loc[automation, "resolution_time"]
        * df.loc[automation, "cost_per_hour"]
    ).sum()
    automation_savings = (
        automation_cost * AUTOMATION_COVERAGE
        * AUTOMATION_COST_REDUCTION * scale
    )

    # Network optimization for critical tickets
    network = (
        (df["issue_category"] == "Network")
        & (df["priority"] == "Critical")
    )
    network_cost = (
        df.loc[network, "resolution_time"]
        * df.loc[network, "cost_per_hour"]
    ).sum()
    network_savings = network_cost * NETWORK_TIME_REDUCTION * scale

    # Reduce escalation-related costs
    escalation_savings = (
        df["escalation_count"].sum()
        * 150 * ESCALATION_REDUCTION * scale
    )

    total_savings = automation_savings + network_savings + escalation_savings
    reduction = total_savings / current_cost
    gap = max(target_savings - total_savings, 0)

    # Engineer allocation by client type
    hours = df.groupby("client_type")["resolution_time"].sum()
    allocation = hours / hours.sum() * 100

    # Customer satisfaction target
    satisfaction = df["satisfaction_score"].mean()
    satisfaction_target = satisfaction > 4.2

    # Save detailed results
    with open(OUTPUT_PATH, "w") as file:
        file.write(
            f"""IT SUPPORT COST OPTIMIZATION SUMMARY
====================================

Development tickets: {sample_tickets:,}
Annualized tickets: {ANNUAL_TICKETS:,}
Current annual cost: ${current_cost:,.2f}
20% savings target: ${target_savings:,.2f}

Savings Scenarios
-----------------
Automation: ${automation_savings:,.2f}
Network optimization: ${network_savings:,.2f}
Escalation reduction: ${escalation_savings:,.2f}

Total savings: ${total_savings:,.2f}
Cost reduction: {reduction * 100:.2f}%
Remaining gap: ${gap:,.2f}

Customer Satisfaction
---------------------
Average satisfaction: {satisfaction:.2f}
Above 4.2 target: {'Yes' if satisfaction_target else 'No'}

Engineer Allocation
-------------------
"""
        )

        for client, pct in allocation.items():
            file.write(f"{client}: {pct:.2f}%\n")

    # Minimal console output
    print(f"Cost reduction: {reduction * 100:.2f}%")
    print(f"Customer satisfaction: {satisfaction:.2f}")
    print(f"Total savings: ${total_savings:,.2f}")
    print(f"Results saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
