import pandas as pd

INPUT_PATH = "/IT_Optimization_Assignment/Data/cleaned_tickets.csv"
OUTPUT_PATH = "/IT_Optimization_Assignment/Data/optimization_summary.txt"

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

    # Annualize development dataset results
    scale_factor = ANNUAL_TICKETS / sample_tickets
    current_annual_cost = df["total_cost"].sum() * scale_factor
    target_annual_savings = current_annual_cost * TARGET_COST_REDUCTION

    # Estimate savings from automating simple software/access tickets
    automation_mask = (
        df["issue_category"].isin(["Software", "Account/Access"])
        & df["priority"].isin(["Low", "Medium"])
    )
    automation_cost = (
        df.loc[automation_mask, "resolution_time"]
        * df.loc[automation_mask, "cost_per_hour"]
    ).sum()
    automation_savings = (
        automation_cost * AUTOMATION_COVERAGE
        * AUTOMATION_COST_REDUCTION * scale_factor
    )

    # Estimate savings from reducing critical network resolution time
    network_mask = (
        (df["issue_category"] == "Network")
        & (df["priority"] == "Critical")
    )
    network_cost = (
        df.loc[network_mask, "resolution_time"]
        * df.loc[network_mask, "cost_per_hour"]
    ).sum()
    network_savings = network_cost * NETWORK_TIME_REDUCTION * scale_factor

    # Estimate savings from reducing escalation costs
    escalation_savings = (
        df["escalation_count"].sum() * 150
        * ESCALATION_REDUCTION * scale_factor
    )

    total_savings = automation_savings + network_savings + escalation_savings
    projected_reduction = total_savings / current_annual_cost
    remaining_gap = max(target_annual_savings - total_savings, 0)

    # Allocate engineer effort based on resolution hours
    engineer_hours = df.groupby("client_type")["resolution_time"].sum()
    allocation = pd.DataFrame({
        "Resolution Hours": engineer_hours,
        "Allocation %": engineer_hours / engineer_hours.sum() * 100
    })

    # Check customer satisfaction target
    average_satisfaction = df["satisfaction_score"].mean()
    satisfaction_target = average_satisfaction > 4.2

    print("\n======================================")
    print("IT SUPPORT COST OPTIMIZATION")
    print("======================================")
    print(f"\nDevelopment tickets: {sample_tickets:,}")
    print(f"Annualized tickets: {ANNUAL_TICKETS:,}")
    print(f"Annualized current support cost: ${current_annual_cost:,.2f}")
    print(f"\n20% target savings: ${target_annual_savings:,.2f}")

    print("\nSavings scenarios:")
    print(f"Automation savings: ${automation_savings:,.2f}")
    print(f"Network optimization savings: ${network_savings:,.2f}")
    print(f"Escalation reduction savings: ${escalation_savings:,.2f}")
    print(f"\nTotal projected savings: ${total_savings:,.2f}")
    print(f"Projected cost reduction: {projected_reduction * 100:.2f}%")
    print(f"Remaining gap to 20% target: ${remaining_gap:,.2f}")

    print(f"\nAverage customer satisfaction: {average_satisfaction:.2f}")
    print(f"Above 4.2 target: {'Yes' if satisfaction_target else 'No'}")

    print("\nEngineer allocation by client type:")
    print(allocation.round(2))

    # Save optimization summary
    with open(OUTPUT_PATH, "w") as file:
        file.write(
            f"""IT SUPPORT COST OPTIMIZATION SUMMARY
====================================

Development tickets: {sample_tickets:,}
Annualized tickets: {ANNUAL_TICKETS:,}
Annualized current support cost: ${current_annual_cost:,.2f}

20% target savings: ${target_annual_savings:,.2f}

Savings Scenarios
----------------
Automation savings: ${automation_savings:,.2f}
Network optimization savings: ${network_savings:,.2f}
Escalation reduction savings: ${escalation_savings:,.2f}

Total projected savings: ${total_savings:,.2f}
Projected cost reduction: {projected_reduction * 100:.2f}%
Remaining gap to 20% target: ${remaining_gap:,.2f}

Customer Satisfaction
---------------------
Average satisfaction: {average_satisfaction:.2f}
Above 4.2 target: {'Yes' if satisfaction_target else 'No'}

Engineer Allocation
-------------------
"""
        )

        for client_type, row in allocation.iterrows():
            file.write(
                f"{client_type}: {row['Resolution Hours']:.2f} hours "
                f"({row['Allocation %']:.2f}%)\n"
            )

    print(f"\nSaved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
