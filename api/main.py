import pickle

import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Loan Risk Prediction API",
    description="MLOps Assignment API",
    version="1.0"
)

print("Loading model...")

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

print("Loading encoder...")

with open("models/label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

print("API ready.")


class LoanRequest(BaseModel):
    Age: int
    Income: int
    Credit_Score: int
    Loan_Amount: int
    Loan_Term: int
    Employment_Status: str


@app.get("/")
def root():
    return {
        "message": "Loan Risk Prediction API Running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: LoanRequest):

    employment_encoded = encoder.transform(
        [data.Employment_Status]
    )[0]

    input_df = pd.DataFrame(
        [{
            "Age": data.Age,
            "Income": data.Income,
            "Credit_Score": data.Credit_Score,
            "Loan_Amount": data.Loan_Amount,
            "Loan_Term": data.Loan_Term,
            "Employment_Status": employment_encoded
        }]
    )

    prediction = int(model.predict(input_df)[0])

    result = (
        "Loan Approved"
        if prediction == 1
        else "Loan Rejected"
    )

    return {
        "prediction": prediction,
        "result": result
    }