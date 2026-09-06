import numpy as np
import pandas as pd

RANDOM_SEED = 42
DEFAULT_ROWS = 100_000
OUTPUT_PATH = "/IT_Optimization_Assignment/Data/raw_tickets.csv"

def generate_data(n_rows=DEFAULT_ROWS):
    rng = np.random.default_rng(RANDOM_SEED)

    categories = ["Hardware", "Software", "Network", "Account/Access", "Other"]
    priorities = ["Low", "Medium", "High", "Critical"]
    clients = ["Small", "Medium", "Enterprise"]

    # Generate ticket characteristics using realistic distributions
    issue_category = rng.choice(categories, n_rows, p=[.20, .30, .20, .20, .10])
    priority = rng.choice(priorities, n_rows, p=[.30, .40, .23, .07])
    client_type = rng.choice(clients, n_rows, p=[.50, .30, .20])

    # Assign hourly cost based on client type
    cost_per_hour = pd.Series(client_type).map({
        "Small": 40, "Medium": 60, "Enterprise": 90
    }).to_numpy()

    # Apply priority and issue-category effects to resolution time
    priority_factor = pd.Series(priority).map({
        "Low": .8, "Medium": 1.0, "High": 1.4, "Critical": 2.0
    }).to_numpy()

    category_factor = pd.Series(issue_category).map({
        "Hardware": 1.15, "Software": 1.0, "Network": 1.30,
        "Account/Access": .80, "Other": 1.05
    }).to_numpy()

    resolution_time = np.clip(
        rng.gamma(2.0, 1.0, n_rows) * priority_factor * category_factor,
        .25, 24
    )

    # Increase escalation probability for complex or high-priority tickets
    escalation_probability = np.clip(
        .08
        + .025 * (resolution_time > 3)
        + .05 * np.isin(priority, ["High", "Critical"])
        + .03 * np.isin(issue_category, ["Network", "Hardware"]),
        .02, .70
    )

    escalation_count = rng.binomial(3, escalation_probability)

    # Calculate support cost from resolution time and escalations
    total_cost = resolution_time * cost_per_hour + escalation_count * 150

    # Generate satisfaction scores based on resolution time and escalations
    satisfaction_score = np.clip(
        5 - .22 * resolution_time - .18 * escalation_count
        + rng.normal(0, .35, n_rows),
        1, 5
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

    # Add approximately 20% missing values to test data cleaning
    for column in ["issue_category", "priority", "resolution_time", "satisfaction_score"]:
        df.loc[rng.random(n_rows) < .20, column] = np.nan

    return df

def main():
    df = generate_data()
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Generated {len(df):,} support tickets.")
    print(f"Saved to: {OUTPUT_PATH}")
    print("\nMissing data percentage:")
    print((df.isnull().mean() * 100).round(2).sort_values(ascending=False))
    print("\nSample data:")
    print(df.head())

if __name__ == "__main__":
    main()
