import pandas as pd

df = pd.read_csv("data/raw/loan_prediction_dataset.csv")

print(df["Loan_Approved"].value_counts())

print()

print(df["Loan_Approved"].value_counts(normalize=True))