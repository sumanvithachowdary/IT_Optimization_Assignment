import pandas as pd

INPUT_PATH = "IT_Optimization_Assignment/Data/raw_tickets.csv"
OUTPUT_PATH = "IT_Optimization_Assignment/Data/cleaned_tickets.csv"

REQUIRED_COLUMNS = [
    "ticket_id", "issue_category", "priority", "resolution_time",
    "cost_per_hour", "client_type", "escalation_count",
    "satisfaction_score", "total_cost"
]

def clean_data(df):
    # Ensure all columns required for analysis are present
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # Fill missing categorical values with a meaningful default
    df["issue_category"] = df["issue_category"].fillna("Unknown")
    df["priority"] = df["priority"].fillna(df["priority"].mode()[0])

    # Fill resolution time using the median for each priority level.
    # Fall back to the overall median if a priority has no valid values.
    df["resolution_time"] = (
        df["resolution_time"]
        .fillna(df.groupby("priority")["resolution_time"].transform("median"))
        .fillna(df["resolution_time"].median())
        .clip(0.25, 24)
    )

    # Replace missing satisfaction scores with the overall median
    # and keep scores within the valid 1–5 range.
    df["satisfaction_score"] = (
        df["satisfaction_score"]
        .fillna(df["satisfaction_score"].median())
        .clip(1, 5)
    )

    # Fill missing hourly costs using the median cost for each client type.
    # Use the overall median as a fallback.
    df["cost_per_hour"] = (
        df["cost_per_hour"]
        .fillna(df.groupby("client_type")["cost_per_hour"].transform("median"))
        .fillna(df["cost_per_hour"].median())
    )

    # Create a derived metric to measure satisfaction relative to total cost
    df["cost_efficiency"] = (
        df["satisfaction_score"] / (df["total_cost"] + 1)
    )

    return df

def main():
    # Load raw ticket data
    df = pd.read_csv(INPUT_PATH)
    print(f"Input records: {len(df):,}")

    # Clean and prepare the data for analysis
    cleaned_df = clean_data(df)

    # Save the cleaned dataset
    cleaned_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Cleaned records: {len(cleaned_df):,}")
    print(f"Saved to: {OUTPUT_PATH}")

    # Verify that no unexpected missing values remain
    print("\nMissing values:")
    print(cleaned_df.isnull().sum().sort_values(ascending=False).head(10))

if __name__ == "__main__":
    main()
