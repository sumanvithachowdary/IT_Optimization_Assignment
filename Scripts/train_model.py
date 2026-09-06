import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

INPUT_PATH = "/IT_Optimization_Assignment/Data/cleaned_tickets.csv"
CONFUSION_MATRIX_PATH = "/IT_Optimization_Assignment/Visuals/confusion_matrix.png"

def main():
    df = pd.read_csv(INPUT_PATH)
    df["is_escalated"] = (df["escalation_count"] > 0).astype(int)

    # Use only information available when the ticket is created
    features = ["issue_category", "priority", "client_type", "cost_per_hour"]
    X, y = df[features], df["is_escalated"]

    categorical = ["issue_category", "priority", "client_type"]
    numerical = ["cost_per_hour"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numerical)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
    ])

    # Train model
    print("Training Logistic Regression model...")
    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    # Benchmark inference
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    inference_time = time.perf_counter() - start

    # Model performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nTraining time: {training_time:.4f}s")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Inference time: {inference_time:.4f}s")

    # Save confusion matrix
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Escalated", "Escalated"]
    ).plot()
    plt.title("Escalation Prediction Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix saved: {CONFUSION_MATRIX_PATH}")

if __name__ == "__main__":
    main()
