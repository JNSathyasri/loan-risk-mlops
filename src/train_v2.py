import os

import mlflow
import mlflow.sklearn

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from imblearn.over_sampling import SMOTE

print("Loading processed dataset...")

df = pd.read_csv(
    "data/processed/processed_data_v2.csv"
)

print("Dataset Shape:", df.shape)

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

print("Train Shape:", X_train.shape)
print("Test Shape :", X_test.shape)

# -------------------------
# MLFLOW EXPERIMENT
# -------------------------

mlflow.set_experiment(
    "Loan_Prediction_V2"
)

results = []

# =====================================================
# EXPERIMENT 1
# RANDOM FOREST
# =====================================================

with mlflow.start_run(
    run_name="Random_Forest_Baseline"
):

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
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

    mlflow.log_param(
        "model",
        "RandomForest"
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    mlflow.sklearn.log_model(
        model,
        "model"
    )

    results.append([
        "Random Forest",
        accuracy,
        precision,
        recall,
        f1
    ])

# =====================================================
# EXPERIMENT 2
# RANDOM FOREST + SMOTE
# =====================================================

smote = SMOTE(
    random_state=42
)

X_smote, y_smote = smote.fit_resample(
    X_train,
    y_train
)

with mlflow.start_run(
    run_name="Random_Forest_SMOTE"
):

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(
        X_smote,
        y_smote
    )

    y_pred = model.predict(
        X_test
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

    mlflow.log_param(
        "model",
        "RandomForest_SMOTE"
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    mlflow.sklearn.log_model(
        model,
        "model"
    )

    results.append([
        "RF + SMOTE",
        accuracy,
        precision,
        recall,
        f1
    ])

# =====================================================
# EXPERIMENT 3
# LOGISTIC REGRESSION
# =====================================================

with mlflow.start_run(
    run_name="Logistic_Regression"
):

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
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

    mlflow.log_param(
        "model",
        "LogisticRegression"
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    mlflow.sklearn.log_model(
        model,
        "model"
    )

    results.append([
        "Logistic Regression",
        accuracy,
        precision,
        recall,
        f1
    ])

# =====================================================
# RESULTS TABLE
# =====================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]
)

print("\nMODEL COMPARISON")
print(results_df)

os.makedirs(
    "reports/metrics",
    exist_ok=True
)

results_df.to_csv(
    "reports/metrics/model_comparison.csv",
    index=False
)

print(
    "\nComparison report saved."
)