import pandas as pd

df = pd.read_csv("data/raw/loan_prediction_dataset.csv")

print("DATASET SHAPE")
print(df.shape)

print("\n")


print("COLUMNS")
print(df.columns.tolist())

print("\n")

print("DATA TYPES")
print(df.dtypes)

print("\n")

print("MISSING VALUES")
print(df.isnull().sum())

print("\n")

print("FIRST 5 ROWS")
print(df.head())