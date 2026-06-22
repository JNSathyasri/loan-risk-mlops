import os
import pickle

import pandas as pd

from sklearn.preprocessing import LabelEncoder

print("Loading dataset...")

df = pd.read_csv("data/raw/loan_data.csv")

print("Dataset loaded successfully.")
print("Shape:", df.shape)

# -------------------------
# DROP LOAN_ID
# -------------------------

df.drop("Loan_ID", axis=1, inplace=True)

print("Loan_ID column removed.")

# -------------------------
# HANDLE MISSING VALUES
# -------------------------

categorical_cols = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

numerical_cols = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

for col in categorical_cols:

    mode_value = df[col].mode()[0]

    df[col].fillna(
        mode_value,
        inplace=True
    )

print("Categorical missing values handled.")

for col in numerical_cols:

    median_value = df[col].median()

    df[col].fillna(
        median_value,
        inplace=True
    )

print("Numerical missing values handled.")

# -------------------------
# CREATE ENCODERS FOLDER
# -------------------------

os.makedirs(
    "models/encoders",
    exist_ok=True
)

encoders = {}

# -------------------------
# ENCODE FEATURES
# -------------------------

feature_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

for col in feature_columns:

    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(
        df[col]
    )

    encoders[col] = encoder

print("Feature encoding completed.")

# -------------------------
# ENCODE TARGET
# -------------------------

target_encoder = LabelEncoder()

df["Loan_Status"] = target_encoder.fit_transform(
    df["Loan_Status"]
)

encoders["Loan_Status"] = target_encoder

print("Target encoding completed.")

# -------------------------
# SAVE ENCODERS
# -------------------------

with open(
    "models/encoders/encoders.pkl",
    "wb"
) as f:

    pickle.dump(
        encoders,
        f
    )

print("Encoders saved.")

# -------------------------
# SAVE DATASET
# -------------------------

os.makedirs(
    "data/processed",
    exist_ok=True
)

df.to_csv(
    "data/processed/processed_data_v2.csv",
    index=False
)

print(
    "Processed dataset saved."
)

print("Final Shape:", df.shape)

print("Preprocessing V2 completed successfully.")