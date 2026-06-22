import pickle

import pandas as pd

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Loan Approval Prediction API",
    description="Production Loan Approval System",
    version="2.0"
)

templates = Jinja2Templates(
    directory="templates"
)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

print("Loading production model...")

with open("models/final_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Loading scaler...")

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print("Loading encoders...")

with open(
    "models/encoders/encoders.pkl",
    "rb"
) as f:
    encoders = pickle.load(f)

print("API Ready.")


class LoanRequest(BaseModel):

    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str

    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: float

    Property_Area: str


@app.get(
    "/",
    response_class=HTMLResponse
)
async def dashboard(
    request: Request
):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: LoanRequest):

    gender = encoders["Gender"].transform(
        [data.Gender]
    )[0]

    married = encoders["Married"].transform(
        [data.Married]
    )[0]

    dependents = encoders["Dependents"].transform(
        [data.Dependents]
    )[0]

    education = encoders["Education"].transform(
        [data.Education]
    )[0]

    self_employed = encoders[
        "Self_Employed"
    ].transform(
        [data.Self_Employed]
    )[0]

    property_area = encoders[
        "Property_Area"
    ].transform(
        [data.Property_Area]
    )[0]

    input_df = pd.DataFrame(
        [{
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": data.ApplicantIncome,
            "CoapplicantIncome": data.CoapplicantIncome,
            "LoanAmount": data.LoanAmount,
            "Loan_Amount_Term": data.Loan_Amount_Term,
            "Credit_History": data.Credit_History,
            "Property_Area": property_area
        }]
    )

    input_scaled = scaler.transform(
        input_df
    )

    prediction = model.predict(
        input_scaled
    )[0]

    probabilities = model.predict_proba(
        input_scaled
    )[0]

    result = (
        "Approved"
        if prediction == 1
        else "Rejected"
    )

    confidence = round(
        max(probabilities) * 100,
        2
    )

    return {
        "result": result,
        "confidence": confidence
    }

    # prediction = int(
    #     model.predict(
    #         input_scaled
    #     )[0]
    # )



    # return {
    #     "prediction": prediction,
    #     "result": result
    # }