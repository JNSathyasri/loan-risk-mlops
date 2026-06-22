import json
import pickle

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("Loading model...")

with open(
    "models/final_model.pkl",
    "rb"
) as f:
    model = pickle.load(f)

print("Loading scaler...")

with open(
    "models/scaler.pkl",
    "rb"
) as f:
    scaler = pickle.load(f)

print("Loading test dataset...")

df = pd.read_csv(
    "data/processed/test_data_v2.csv"
)

X_test = df.drop(
    "Loan_Status",
    axis=1
)

y_test = df["Loan_Status"]

print("Scaling test data...")

X_test_scaled = scaler.transform(
    X_test
)

print("Generating predictions...")

y_pred = model.predict(
    X_test_scaled
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1)
}

with open(
    "reports/final_model_metrics.json",
    "w"
) as f:
    json.dump(
        metrics,
        f,
        indent=4
    )

print("\nFINAL MODEL METRICS")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("Evaluation completed successfully.")