"""
Train a Logistic Regression model to predict whether
an IT support ticket is likely to be escalated.

The model uses only information available when the ticket
is created:
- Issue category
- Priority
- Client type
- Cost per engineer hour

The assignment requires prediction within 5 seconds.
Inference performance is therefore explicitly benchmarked.
"""

import time

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


INPUT_PATH = "/content/Optimization_Assignment/assignment-2/Data/cleaned_tickets.csv"

MODEL_METRICS_PATH = "/content/Optimization_Assignment/assignment-2/Data/model_metrics.txt"
CONFUSION_MATRIX_PATH = "/content/Optimization_Assignment/assignment-2/Visuals/confusion_matrix.png"


def main():

    df = pd.read_csv(INPUT_PATH)

    # ------------------------------------------------
    # Target
    # ------------------------------------------------

    df["is_escalated"] = (
        df["escalation_count"] > 0
    ).astype(int)

    # ------------------------------------------------
    # Features available when ticket is created
    # ------------------------------------------------

    features = [
        "issue_category",
        "priority",
        "client_type",
        "cost_per_hour"
    ]

    target = "is_escalated"

    X = df[features]
    y = df[target]

    categorical_features = [
        "issue_category",
        "priority",
        "client_type"
    ]

    numerical_features = [
        "cost_per_hour"
    ]

    # ------------------------------------------------
    # Train/test split
    # ------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ------------------------------------------------
    # Preprocessing
    # ------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            ),
            (
                "numerical",
                "passthrough",
                numerical_features
            )
        ]
    )

    # ------------------------------------------------
    # Logistic Regression
    # ------------------------------------------------

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    # ------------------------------------------------
    # Training
    # ------------------------------------------------

    print("Training Logistic Regression model...")

    training_start = time.perf_counter()

    pipeline.fit(
        X_train,
        y_train
    )

    training_time = (
        time.perf_counter()
        - training_start
    )

    # ------------------------------------------------
    # Predictions
    # ------------------------------------------------

    prediction_start = time.perf_counter()

    y_pred = pipeline.predict(X_test)
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    total_inference_time = (
        time.perf_counter()
        - prediction_start
    )

    benchmark_count = len(X_test)

    average_prediction_time = (
        total_inference_time / benchmark_count
    )

    predictions_per_second = (
        benchmark_count / total_inference_time
        if total_inference_time > 0
        else float("inf")
    )

    # ------------------------------------------------
    # Metrics
    # ------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\n===== MODEL PERFORMANCE =====")

    print(
        f"Training time: "
        f"{training_time:.4f} seconds"
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1 Score: {f1:.4f}"
    )

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print("\n===== INFERENCE PERFORMANCE =====")

    print(
        f"Test records: {benchmark_count:,}"
    )

    print(
        f"Total inference time: "
        f"{total_inference_time:.6f} seconds"
    )

    print(
        f"Average prediction time: "
        f"{average_prediction_time:.8f} seconds"
    )

    print(
        f"Predictions per second: "
        f"{predictions_per_second:,.2f}"
    )

    # ------------------------------------------------
    # Confusion matrix visualization
    # ------------------------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Not Escalated",
            "Escalated"
        ]
    )

    display.plot()

    plt.title(
        "Escalation Prediction Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nSaved: {CONFUSION_MATRIX_PATH}"
    )

    # ------------------------------------------------
    # Save metrics
    # ------------------------------------------------

    with open(
        MODEL_METRICS_PATH,
        "w"
    ) as file:

        file.write(
            "IT SUPPORT ESCALATION MODEL RESULTS\n"
        )

        file.write(
            "====================================\n\n"
        )

        file.write(
            f"Training records: {len(X_train):,}\n"
        )

        file.write(
            f"Test records: {len(X_test):,}\n\n"
        )

        file.write(
            f"Training time: "
            f"{training_time:.6f} seconds\n"
        )

        file.write(
            f"Accuracy: {accuracy:.6f}\n"
        )

        file.write(
            f"Precision: {precision:.6f}\n"
        )

        file.write(
            f"Recall: {recall:.6f}\n"
        )

        file.write(
            f"F1 Score: {f1:.6f}\n"
        )

        file.write(
            f"ROC-AUC: {roc_auc:.6f}\n\n"
        )

        file.write(
            "Confusion Matrix:\n"
        )

        file.write(
            f"{cm}\n\n"
        )

        file.write(
            "Inference Performance:\n"
        )

        file.write(
            f"Total inference time: "
            f"{total_inference_time:.6f} seconds\n"
        )

        file.write(
            f"Average prediction time: "
            f"{average_prediction_time:.8f} seconds\n"
        )

        file.write(
            f"Predictions per second: "
            f"{predictions_per_second:.2f}\n"
        )

    print(
        f"Saved: {MODEL_METRICS_PATH}"
    )


if __name__ == "__main__":
    main()