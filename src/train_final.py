import json
import os
import pickle
import yaml

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

print("Loading processed dataset...")

df = pd.read_csv(
    "data/processed/processed_data_v2.csv"
)

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

max_iter = params["logistic_regression"]["max_iter"]

# -------------------------
# FEATURES / TARGET
# -------------------------

X = df.drop(
    "Loan_Status",
    axis=1
)

y = df["Loan_Status"]

# -------------------------
# TRAIN TEST SPLIT
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------
# SAVE TEST DATASET
# -------------------------

test_df = X_test.copy()
test_df["Loan_Status"] = y_test

test_df.to_csv(
    "data/processed/test_data_v2.csv",
    index=False
)

print("Test dataset saved.")

# -------------------------
# FEATURE SCALING
# -------------------------

print("Scaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)

# -------------------------
# TRAIN FINAL MODEL
# -------------------------

print("Training final model...")

model = LogisticRegression(
    max_iter=max_iter
)

model.fit(
    X_train_scaled,
    y_train
)

print("Model trained successfully.")

# -------------------------
# EVALUATION
# -------------------------

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

print("\nFINAL MODEL METRICS")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

# -------------------------
# SAVE MODEL
# -------------------------

os.makedirs(
    "models",
    exist_ok=True
)

with open(
    "models/final_model.pkl",
    "wb"
) as f:

    pickle.dump(
        model,
        f
    )

print("Final model saved.")

# -------------------------
# SAVE SCALER
# -------------------------

with open(
    "models/scaler.pkl",
    "wb"
) as f:

    pickle.dump(
        scaler,
        f
    )

print("Scaler saved.")

# -------------------------
# SAVE METRICS
# -------------------------

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
}

os.makedirs(
    "reports",
    exist_ok=True
)

with open(
    "reports/final_model_metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

print("Metrics saved.")

print("\nProduction model ready.")