import pickle

import pandas as pd

import logging

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

from prometheus_client import (
    Counter,
    generate_latest,
    CONTENT_TYPE_LATEST
)

from fastapi import Response

app = FastAPI(
    title="Loan Approval Prediction API",
    description="Production Loan Approval System",
    version="2.0"
)

TOTAL_REQUESTS = Counter(
    "total_requests",
    "Total API Requests"
)

PREDICTION_REQUESTS = Counter(
    "prediction_requests",
    "Total Prediction Requests"
)

HEALTH_REQUESTS = Counter(
    "health_requests",
    "Total Health Requests"
)

logging.basicConfig(
    filename="loan_api.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

templates = Jinja2Templates(
    directory="templates"
)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


model = None
scaler = None
encoders = None


def load_artifacts():

    global model
    global scaler
    global encoders

    if (
        model is None
        or scaler is None
        or encoders is None
    ):

        logging.info(
            "Loading model artifacts"
        )

        print("Loading production model...")

        with open(
            "models/final_model.pkl",
            "rb"
        ) as f:
            model = pickle.load(f)

        print("Loading scaler...")

        with open(
            "models/scaler.pkl",
            "rb"
        ) as f:
            scaler = pickle.load(f)

        print("Loading encoders...")

        with open(
            "models/encoders/encoders.pkl",
            "rb"
        ) as f:
            encoders = pickle.load(f)

        print("Artifacts loaded successfully.")

# print("Loading production model...")

# with open("models/final_model.pkl", "rb") as f:
#     model = pickle.load(f)

# print("Loading scaler...")

# with open("models/scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# print("Loading encoders...")

# with open(
#     "models/encoders/encoders.pkl",
#     "rb"
# ) as f:
#     encoders = pickle.load(f)

# print("API Ready.")


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
    TOTAL_REQUESTS.inc()
    HEALTH_REQUESTS.inc()
    logging.info(
        "Health endpoint called"
    )

    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: LoanRequest):

    TOTAL_REQUESTS.inc()
    PREDICTION_REQUESTS.inc()

    load_artifacts()
    logging.info(
        "Prediction request received"
    )
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
    logging.info(
        f"Prediction result: {result}"
    )

    logging.info(
        f"Confidence: {confidence}"
    )
    return {
        "result": result,
        "confidence": confidence
    }
@app.get("/metrics")
def metrics():

    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
