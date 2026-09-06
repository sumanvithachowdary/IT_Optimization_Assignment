import pandas as pd

INPUT_PATH = "/content/Optimization_Assignment/assignment-2/Data/raw_tickets.csv"
OUTPUT_PATH = "/content/Optimization_Assignment/assignment-2/Data/cleaned_tickets.csv"

REQUIRED_COLUMNS = [
    "ticket_id",
    "issue_category",
    "priority",
    "resolution_time",
    "cost_per_hour",
    "client_type",
    "escalation_count",
    "satisfaction_score",
    "total_cost"
]

def clean_data(df):
    # Validate required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df.copy()

    # Categorical fields
    df["issue_category"] = (
        df["issue_category"]
        .fillna("Unknown")
    )

    priority_mode = df["priority"].mode()

    if not priority_mode.empty:
        df["priority"] = (
            df["priority"]
            .fillna(priority_mode.iloc[0])
        )
    else:
        df["priority"] = (
            df["priority"]
            .fillna("Medium")
        )

    # Resolution time
    overall_resolution_median = (
        df["resolution_time"].median()
    )

    priority_medians = (
        df.groupby("priority")["resolution_time"]
        .median()
    )

    for priority_value in df["priority"].unique():

        mask = (
            (df["priority"] == priority_value)
            & (df["resolution_time"].isna())
        )

        if priority_value in priority_medians.index:
            replacement = priority_medians.loc[
                priority_value
            ]
        else:
            replacement = overall_resolution_median

        df.loc[mask, "resolution_time"] = replacement

    df["resolution_time"] = (
        df["resolution_time"]
        .fillna(overall_resolution_median)
        .clip(lower=0.25, upper=24.0)
    )
    
    # Satisfaction score
    satisfaction_median = (
        df["satisfaction_score"].median()
    )

    df["satisfaction_score"] = (
        df["satisfaction_score"]
        .fillna(satisfaction_median)
        .clip(lower=1.0, upper=5.0)
    )
    
    # Cost per hour
    client_cost_medians = (
        df.groupby("client_type")["cost_per_hour"]
        .median()
    )

    for client_type in df["client_type"].unique():

        mask = (
            (df["client_type"] == client_type)
            & (df["cost_per_hour"].isna())
        )

        if client_type in client_cost_medians.index:
            replacement = client_cost_medians.loc[
                client_type
            ]
        else:
            replacement = df["cost_per_hour"].median()

        df.loc[mask, "cost_per_hour"] = replacement

    df["cost_per_hour"] = (
        df["cost_per_hour"]
        .fillna(df["cost_per_hour"].median())
    )

    # Derived feature
    df["cost_efficiency"] = (
        df["satisfaction_score"]
        / (df["total_cost"] + 1)
    )

    # Final validation
    if df.isnull().sum().sum() > 0:
        print("Warning: Some missing values remain.")
    return df

def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Input records: {len(df):,}")
    cleaned_df = clean_data(df)
    cleaned_df.to_csv(
        OUTPUT_PATH,
        index=False
    )
    print(f"Cleaned records: {len(cleaned_df):,}")
    print(f"Saved to: {OUTPUT_PATH}")
    print("\nRemaining missing values:")
    print(
        cleaned_df.isnull().sum()
        .sort_values(ascending=False)
        .head(10)
    )
if __name__ == "__main__":
    main()
