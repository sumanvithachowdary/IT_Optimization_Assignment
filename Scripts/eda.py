import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_PATH = "/IT_Optimization_Assignment/Data/cleaned_tickets.csv"
OUTPUT_DIR = "/IT_Optimization_Assignment/Visuals"
SUMMARY_PATH = "/IT_Optimization_Assignment/Data/category_summary.csv"

def save_plot(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Records available for EDA: {len(df):,}")

    # Total cost by issue category
    df.groupby("issue_category")["total_cost"].sum().sort_values().plot(
        kind="bar", figsize=(9, 5), title="Total Support Cost by Issue Category"
    )
    plt.xlabel("Issue Category")
    plt.ylabel("Total Cost")
    save_plot("cost_by_category.png")

    # Average resolution time by priority
    order = ["Low", "Medium", "High", "Critical"]
    df.groupby("priority")["resolution_time"].mean().reindex(order).plot(
        kind="bar", figsize=(8, 5), title="Average Resolution Time by Priority"
    )
    plt.xlabel("Priority")
    plt.ylabel("Resolution Time (hours)")
    save_plot("res_time_by_priority.png")

    # Average escalations by client type
    df.groupby("client_type")["escalation_count"].mean().reindex(
        ["Small", "Medium", "Enterprise"]
    ).plot(kind="bar", figsize=(8, 5), title="Average Escalations by Client Type")
    plt.xlabel("Client Type")
    plt.ylabel("Average Escalation Count")
    save_plot("escalation_by_client.png")

    # Satisfaction vs resolution time
    sample = df.sample(min(1000, len(df)), random_state=42)
    sns.scatterplot(data=sample, x="resolution_time", y="satisfaction_score", alpha=0.5)
    plt.title("Customer Satisfaction vs Resolution Time")
    plt.xlabel("Resolution Time (hours)")
    plt.ylabel("Satisfaction Score")
    save_plot("sat_vs_res_time.png")

    # Total cost by priority
    df.groupby("priority")["total_cost"].sum().reindex(order).plot(
        kind="bar", figsize=(8, 5), title="Total Support Cost by Priority"
    )
    plt.xlabel("Priority")
    plt.ylabel("Total Cost")
    save_plot("cost_by_priority.png")

    # Escalation rate by priority
    escalation_rate = (
        df.assign(escalated=df["escalation_count"] > 0)
        .groupby("priority")["escalated"].mean()
        .reindex(order) * 100
    )
    escalation_rate.plot(
        kind="bar", figsize=(8, 5), title="Escalation Rate by Priority"
    )
    plt.xlabel("Priority")
    plt.ylabel("Escalation Rate (%)")
    save_plot("escalation_rate_by_priority.png")

    # Category-level summary
    summary = df.groupby("issue_category").agg(
        ticket_count=("ticket_id", "count"),
        total_cost=("total_cost", "sum"),
        avg_resolution_time=("resolution_time", "mean"),
        avg_escalations=("escalation_count", "mean"),
        avg_satisfaction=("satisfaction_score", "mean")
    ).sort_values("total_cost", ascending=False)

    summary.to_csv(SUMMARY_PATH)
    print("\nCategory summary:")
    print(summary)

if __name__ == "__main__":
    main()
