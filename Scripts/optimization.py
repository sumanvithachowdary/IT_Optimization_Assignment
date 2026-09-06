"""
IT Support Cost Optimization.

The assignment requires:
1. Reduce support cost by 20%
2. Improve customer satisfaction above 4.2
3. Reduce unnecessary escalations
4. Optimize engineer allocation

The development dataset contains 100,000 tickets.
Results are annualized to the assignment scale of 2 million
tickets/year.
"""

import pandas as pd


INPUT_PATH = "/content/Optimization_Assignment/assignment-2/Data/cleaned_tickets.csv"
OUTPUT_PATH = "/content/Optimization_Assignment/assignment-2/Data/optimization_summary.txt"

ANNUAL_TICKETS = 2_000_000

# Automation scenario
AUTOMATION_COVERAGE = 0.30
AUTOMATION_COST_REDUCTION = 0.50

# Network optimization scenario
NETWORK_TIME_REDUCTION = 0.20

# Escalation optimization
ESCALATION_REDUCTION = 0.25

# Target
TARGET_COST_REDUCTION = 0.20


def main():

    df = pd.read_csv(INPUT_PATH)

    sample_tickets = len(df)

    if sample_tickets == 0:
        raise ValueError(
            "Dataset contains no records."
        )

    # ------------------------------------------------
    # Annualization
    # ------------------------------------------------

    scale_factor = (
        ANNUAL_TICKETS / sample_tickets
    )

    current_sample_cost = (
        df["total_cost"].sum()
    )

    current_annual_cost = (
        current_sample_cost
        * scale_factor
    )

    target_annual_savings = (
        current_annual_cost
        * TARGET_COST_REDUCTION
    )

    # ------------------------------------------------
    # Scenario 1: Automation
    # ------------------------------------------------

    automation_mask = (
        df["issue_category"].isin(
            ["Software", "Account/Access"]
        )
        & df["priority"].isin(
            ["Low", "Medium"]
        )
    )

    automation_engineer_cost = (
        df.loc[
            automation_mask,
            "resolution_time"
        ]
        * df.loc[
            automation_mask,
            "cost_per_hour"
        ]
    ).sum()

    automation_savings_sample = (
        automation_engineer_cost
        * AUTOMATION_COVERAGE
        * AUTOMATION_COST_REDUCTION
    )

    automation_savings_annual = (
        automation_savings_sample
        * scale_factor
    )

    # ------------------------------------------------
    # Scenario 2: Network critical optimization
    # ------------------------------------------------

    network_mask = (
        (df["issue_category"] == "Network")
        & (df["priority"] == "Critical")
    )

    network_engineer_cost = (
        df.loc[
            network_mask,
            "resolution_time"
        ]
        * df.loc[
            network_mask,
            "cost_per_hour"
        ]
    ).sum()

    network_savings_sample = (
        network_engineer_cost
        * NETWORK_TIME_REDUCTION
    )

    network_savings_annual = (
        network_savings_sample
        * scale_factor
    )

    # ------------------------------------------------
    # Scenario 3: Reduce escalation costs
    # ------------------------------------------------

    escalation_penalty_sample = (
        df["escalation_count"].sum()
        * 150
    )

    escalation_savings_sample = (
        escalation_penalty_sample
        * ESCALATION_REDUCTION
    )

    escalation_savings_annual = (
        escalation_savings_sample
        * scale_factor
    )

    # ------------------------------------------------
    # Total savings
    # ------------------------------------------------

    total_projected_savings = (
        automation_savings_annual
        + network_savings_annual
        + escalation_savings_annual
    )

    projected_reduction = (
        total_projected_savings
        / current_annual_cost
    )

    remaining_gap = max(
        target_annual_savings
        - total_projected_savings,
        0
    )

    # ------------------------------------------------
    # Engineer allocation
    # ------------------------------------------------

    engineer_allocation = (
        df.groupby("client_type")[
            "resolution_time"
        ]
        .sum()
        .sort_values(ascending=False)
    )

    engineer_allocation_percent = (
        engineer_allocation
        / engineer_allocation.sum()
        * 100
    )

    # ------------------------------------------------
    # Customer satisfaction
    # ------------------------------------------------

    average_satisfaction = (
        df["satisfaction_score"].mean()
    )

    satisfaction_above_target = (
        average_satisfaction > 4.2
    )

    # ------------------------------------------------
    # Print results
    # ------------------------------------------------

    print("\n======================================")
    print("IT SUPPORT COST OPTIMIZATION")
    print("======================================")

    print(
        f"\nDevelopment tickets: "
        f"{sample_tickets:,}"
    )

    print(
        f"Annualized tickets: "
        f"{ANNUAL_TICKETS:,}"
    )

    print(
        f"Annualized current support cost: "
        f"${current_annual_cost:,.2f}"
    )

    print(
        f"\n20% target savings: "
        f"${target_annual_savings:,.2f}"
    )

    print("\nSavings scenarios:")

    print(
        f"Automation savings: "
        f"${automation_savings_annual:,.2f}"
    )

    print(
        f"Network optimization savings: "
        f"${network_savings_annual:,.2f}"
    )

    print(
        f"Escalation reduction savings: "
        f"${escalation_savings_annual:,.2f}"
    )

    print(
        f"\nTotal projected savings: "
        f"${total_projected_savings:,.2f}"
    )

    print(
        f"Projected cost reduction: "
        f"{projected_reduction * 100:.2f}%"
    )

    print(
        f"Remaining gap to 20% target: "
        f"${remaining_gap:,.2f}"
    )

    print(
        f"\nAverage customer satisfaction: "
        f"{average_satisfaction:.2f}"
    )

    print(
        f"Above 4.2 target: "
        f"{'Yes' if satisfaction_above_target else 'No'}"
    )

    print("\nEngineer allocation by client type:")

    allocation_table = pd.DataFrame({
        "Resolution Hours": engineer_allocation,
        "Allocation %": engineer_allocation_percent
    })

    print(
        allocation_table.round(2)
    )

    # ------------------------------------------------
    # Save summary
    # ------------------------------------------------

    with open(
        OUTPUT_PATH,
        "w"
    ) as file:

        file.write(
            "IT SUPPORT COST OPTIMIZATION SUMMARY\n"
        )

        file.write(
            "====================================\n\n"
        )

        file.write(
            f"Development tickets: "
            f"{sample_tickets:,}\n"
        )

        file.write(
            f"Annualized tickets: "
            f"{ANNUAL_TICKETS:,}\n"
        )

        file.write(
            f"Annualized current support cost: "
            f"${current_annual_cost:,.2f}\n\n"
        )

        file.write(
            f"20% target savings: "
            f"${target_annual_savings:,.2f}\n\n"
        )

        file.write(
            "Savings Scenarios\n"
        )

        file.write(
            "----------------\n"
        )

        file.write(
            f"Automation savings: "
            f"${automation_savings_annual:,.2f}\n"
        )

        file.write(
            f"Network optimization savings: "
            f"${network_savings_annual:,.2f}\n"
        )

        file.write(
            f"Escalation reduction savings: "
            f"${escalation_savings_annual:,.2f}\n\n"
        )

        file.write(
            f"Total projected savings: "
            f"${total_projected_savings:,.2f}\n"
        )

        file.write(
            f"Projected cost reduction: "
            f"{projected_reduction * 100:.2f}%\n"
        )

        file.write(
            f"Remaining gap to 20% target: "
            f"${remaining_gap:,.2f}\n\n"
        )

        file.write(
            "Customer Satisfaction\n"
        )

        file.write(
            "---------------------\n"
        )

        file.write(
            f"Average satisfaction: "
            f"{average_satisfaction:.2f}\n"
        )

        file.write(
            f"Above 4.2 target: "
            f"{'Yes' if satisfaction_above_target else 'No'}\n\n"
        )

        file.write(
            "Engineer Allocation\n"
        )

        file.write(
            "-------------------\n"
        )

        for client_type, row in allocation_table.iterrows():

            file.write(
                f"{client_type}: "
                f"{row['Resolution Hours']:.2f} hours "
                f"({row['Allocation %']:.2f}%)\n"
            )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()