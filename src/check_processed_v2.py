import pandas as pd

df = pd.read_csv(
    "data/processed/processed_data_v2.csv"
)

print("\nSHAPE")
print(df.shape)

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nTARGET DISTRIBUTION")
print(df["Loan_Status"].value_counts())

print("\nTARGET PROPORTION")
print(
    df["Loan_Status"].value_counts(
        normalize=True
    )
)