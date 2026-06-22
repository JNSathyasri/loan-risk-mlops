import pandas as pd

df = pd.read_csv("data/raw/loan_data.csv")

print("\nSHAPE")
print(df.shape)

print("\nCOLUMNS")
print(df.columns.tolist())

print("\nINFO")
print(df.info())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nTARGET DISTRIBUTION")
print(df["Loan_Status"].value_counts())

print("\nTARGET PROPORTION")
print(df["Loan_Status"].value_counts(normalize=True))