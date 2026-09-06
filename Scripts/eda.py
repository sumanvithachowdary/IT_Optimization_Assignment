"""
Exploratory Data Analysis for IT support tickets.

Generates visualizations for:
- Cost by category
- Resolution time by priority
- Escalations by client type
- Satisfaction vs resolution time
- Cost by priority
- Escalation rate by priority
"""

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


INPUT_PATH = "/content/Optimization_Assignment/assignment-2/Data/cleaned_tickets.csv"
OUTPUT_DIR = "."


def save_plot(filename):
    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Saved: {path}")


def main():

    global OUTPUT_DIR

    df = pd.read_csv(INPUT_PATH)

    print(f"Records available for EDA: {len(df):,}")

    # ------------------------------------------------
    # 1. Total cost by issue category
    # ------------------------------------------------

    category_cost = (
        df.groupby("issue_category")["total_cost"]
        .sum()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(9, 5))

    category_cost.plot(
        kind="bar"
    )

    plt.title("Total Support Cost by Issue Category")
    plt.xlabel("Issue Category")
    plt.ylabel("Total Cost")

    save_plot("/content/Optimization_Assignment/assignment-2/Visuals/cost_by_category.png")

    # ------------------------------------------------
    # 2. Average resolution time by priority
    # ------------------------------------------------

    priority_resolution = (
        df.groupby("priority")["resolution_time"]
        .mean()
        .reindex(
            ["Low", "Medium", "High", "Critical"]
        )
    )

    plt.figure(figsize=(8, 5))

    priority_resolution.plot(
        kind="bar"
    )

    plt.title("Average Resolution Time by Priority")
    plt.xlabel("Priority")
    plt.ylabel("Resolution Time (hours)")

    save_plot("/content/Optimization_Assignment/assignment-2/Visuals/res_time_by_priority.png")

    # ------------------------------------------------
    # 3. Average escalations by client type
    # ------------------------------------------------

    client_escalation = (
        df.groupby("client_type")["escalation_count"]
        .mean()
        .reindex(
            ["Small", "Medium", "Enterprise"]
        )
    )

    plt.figure(figsize=(8, 5))

    client_escalation.plot(
        kind="bar"
    )

    plt.title("Average Escalations by Client Type")
    plt.xlabel("Client Type")
    plt.ylabel("Average Escalation Count")

    save_plot("/content/Optimization_Assignment/assignment-2/Visuals/escalation_by_client.png")

    # ------------------------------------------------
    # 4. Satisfaction vs resolution time
    # ------------------------------------------------

    sample_size = min(1000, len(df))

    if sample_size > 0:

        sample_df = df.sample(
            n=sample_size,
            random_state=42
        )

        plt.figure(figsize=(9, 6))

        sns.scatterplot(
            data=sample_df,
            x="resolution_time",
            y="satisfaction_score",
            alpha=0.5
        )

        plt.title(
            "Customer Satisfaction vs Resolution Time"
        )

        plt.xlabel("Resolution Time (hours)")
        plt.ylabel("Satisfaction Score")

        save_plot("/content/Optimization_Assignment/assignment-2/Visuals/sat_vs_res_time.png")

    # ------------------------------------------------
    # 5. Total cost by priority
    # ------------------------------------------------

    priority_cost = (
        df.groupby("priority")["total_cost"]
        .sum()
        .reindex(
            ["Low", "Medium", "High", "Critical"]
        )
    )

    plt.figure(figsize=(8, 5))

    priority_cost.plot(
        kind="bar"
    )

    plt.title("Total Support Cost by Priority")
    plt.xlabel("Priority")
    plt.ylabel("Total Cost")

    save_plot("/content/Optimization_Assignment/assignment-2/Visuals/cost_by_priority.png")

    # ------------------------------------------------
    # 6. Escalation rate by priority
    # ------------------------------------------------

    escalation_rate = (
        df.assign(
            escalated=df["escalation_count"] > 0
        )
        .groupby("priority")["escalated"]
        .mean()
        .reindex(
            ["Low", "Medium", "High", "Critical"]
        )
        * 100
    )

    plt.figure(figsize=(8, 5))

    escalation_rate.plot(
        kind="bar"
    )

    plt.title("Escalation Rate by Priority")
    plt.xlabel("Priority")
    plt.ylabel("Escalation Rate (%)")

    save_plot("/content/Optimization_Assignment/assignment-2/Visuals/escalation_rate_by_priority.png")

    # ------------------------------------------------
    # Summary table
    # ------------------------------------------------

    category_summary = (
        df.groupby("issue_category")
        .agg(
            ticket_count=("ticket_id", "count"),
            total_cost=("total_cost", "sum"),
            avg_resolution_time=("resolution_time", "mean"),
            avg_escalations=("escalation_count", "mean"),
            avg_satisfaction=("satisfaction_score", "mean")
        )
        .sort_values(
            "total_cost",
            ascending=False
        )
    )

    category_summary.to_csv(
        "/content/Optimization_Assignment/assignment-2/Data/category_summary.csv"
    )

    print("\nCategory summary:")
    print(category_summary)


if __name__ == "__main__":
    main()