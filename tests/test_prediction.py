from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_prediction_endpoint():

    payload = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 2000,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": "Urban"
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "result" in data

    assert "confidence" in data
