import json
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

print("Loading model.")

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully.")

print("Loading test dataset.")

df = pd.read_csv("data/processed/test_data.csv")

X_test = df.drop("Loan_Approved", axis=1)
y_test = df["Loan_Approved"]

print("Generating predictions")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

os.makedirs("reports/metrics", exist_ok=True)

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
}

with open("reports/metrics/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Metrics saved.")

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.savefig(
    "reports/metrics/confusion_matrix.png",
    bbox_inches="tight"
)

plt.close()

print("Confusion matrix saved.")

print("Evaluation completed successfully.")