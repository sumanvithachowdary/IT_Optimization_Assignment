"""
Generate synthetic IT support ticket data.

The assignment describes a production scale of 2 million tickets/year.
For development and Colab execution, this script generates 100,000 tickets
and the optimization script annualizes the results to 2 million tickets.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
DEFAULT_ROWS = 100_000
OUTPUT_PATH = "/content/Optimization_Assignment/assignment-2/Data/raw_tickets.csv"


def generate_data(n_rows=DEFAULT_ROWS):
    rng = np.random.default_rng(RANDOM_SEED)

    issue_categories = [
        "Hardware",
        "Software",
        "Network",
        "Account/Access",
        "Other"
    ]

    priorities = [
        "Low",
        "Medium",
        "High",
        "Critical"
    ]

    client_types = [
        "Small",
        "Medium",
        "Enterprise"
    ]

    # Generate categorical fields
    issue_category = rng.choice(
        issue_categories,
        size=n_rows,
        p=[0.20, 0.30, 0.20, 0.20, 0.10]
    )

    priority = rng.choice(
        priorities,
        size=n_rows,
        p=[0.30, 0.40, 0.23, 0.07]
    )

    client_type = rng.choice(
        client_types,
        size=n_rows,
        p=[0.50, 0.30, 0.20]
    )

    # Engineer cost based on client type
    cost_map = {
        "Small": 40,
        "Medium": 60,
        "Enterprise": 90
    }

    cost_per_hour = np.array([
        cost_map[x] for x in client_type
    ], dtype=float)

    # Priority multiplier
    priority_multiplier = {
        "Low": 0.8,
        "Medium": 1.0,
        "High": 1.4,
        "Critical": 2.0
    }

    # Category multiplier
    category_multiplier = {
        "Hardware": 1.15,
        "Software": 1.00,
        "Network": 1.30,
        "Account/Access": 0.80,
        "Other": 1.05
    }

    priority_factor = np.array([
        priority_multiplier[x] for x in priority
    ])

    category_factor = np.array([
        category_multiplier[x] for x in issue_category
    ])

    # Resolution time in hours
    resolution_time = (
        rng.gamma(
            shape=2.0,
            scale=1.0,
            size=n_rows
        )
        * priority_factor
        * category_factor
    )

    resolution_time = np.clip(
        resolution_time,
        0.25,
        24.0
    )

    # Escalation probability
    escalation_probability = (
        0.08
        + 0.025 * (resolution_time > 3)
        + 0.05 * np.isin(priority, ["High", "Critical"])
        + 0.03 * np.isin(issue_category, ["Network", "Hardware"])
    )

    escalation_probability = np.clip(
        escalation_probability,
        0.02,
        0.70
    )

    escalation_count = rng.binomial(
        n=3,
        p=escalation_probability
    )

    # Total support cost
    engineer_cost = resolution_time * cost_per_hour

    escalation_cost = escalation_count * 150

    total_cost = engineer_cost + escalation_cost

    # Satisfaction score
    satisfaction_score = (
        5.0
        - 0.22 * resolution_time
        - 0.18 * escalation_count
        + rng.normal(0, 0.35, n_rows)
    )

    satisfaction_score = np.clip(
        satisfaction_score,
        1.0,
        5.0
    )

    df = pd.DataFrame({
        "ticket_id": np.arange(1, n_rows + 1),
        "issue_category": issue_category,
        "priority": priority,
        "resolution_time": resolution_time,
        "cost_per_hour": cost_per_hour,
        "client_type": client_type,
        "escalation_count": escalation_count,
        "satisfaction_score": satisfaction_score,
        "total_cost": total_cost
    })

    # Introduce approximately 20% random missing data
    missing_columns = [
        "issue_category",
        "priority",
        "resolution_time",
        "satisfaction_score"
    ]

    for column in missing_columns:
        mask = rng.random(n_rows) < 0.20
        df.loc[mask, column] = np.nan

    return df


def main():
    df = generate_data()

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"Generated {len(df):,} support tickets.")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nMissing data percentage:")
    print(
        (df.isnull().mean() * 100)
        .round(2)
        .sort_values(ascending=False)
    )

    print("\nSample data:")
    print(df.head())


if __name__ == "__main__":
    main()