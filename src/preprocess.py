import os
import pickle

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Create directories if they do not exist
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("Loading dataset")

df = pd.read_csv("data/raw/loan_prediction_dataset.csv")

print("Dataset loaded successfully.")
print(f"Shape: {df.shape}")

# Encode Employment_Status
encoder = LabelEncoder()

df["Employment_Status"] = encoder.fit_transform(
    df["Employment_Status"]
)

print("Employment_Status encoded.")

# Save encoder
encoder_path = "models/label_encoder.pkl"

with open(encoder_path, "wb") as f:
    pickle.dump(encoder, f)

print(f"Encoder saved to: {encoder_path}")

# Save processed dataset
output_path = "data/processed/processed_data.csv"

df.to_csv(output_path, index=False)

print(f"Processed dataset saved to: {output_path}")

print("Preprocessing completed successfully.")