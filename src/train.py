import os
import pickle

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("Loading processed dataset.")

df = pd.read_csv("data/processed/processed_data.csv")

print(f"Dataset Shape: {df.shape}")

# Features and Target
X = df.drop("Loan_Approved", axis=1)
y = df["Loan_Approved"]

print("Creating train-test split.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

print("Training Random Forest model.")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model trained successfully.")

# Create models directory
os.makedirs("models", exist_ok=True)

# Save model
model_path = "models/model.pkl"

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to: {model_path}")

# Save test dataset for evaluation stage
test_df = X_test.copy()
test_df["Loan_Approved"] = y_test

test_path = "data/processed/test_data.csv"

test_df.to_csv(test_path, index=False)

print(f"Test dataset saved to: {test_path}")

print("Training pipeline completed successfully.")