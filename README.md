# Loan Approval Prediction Platform using MLOps

## Project Overview

The Loan Approval Prediction Platform is an end-to-end Machine Learning Operations (MLOps) project designed to automate loan approval prediction using a production-ready workflow. The project integrates machine learning, data versioning, experiment tracking, API deployment, CI/CD automation, monitoring, and security scanning into a unified pipeline.

The objective is to demonstrate how modern MLOps practices can be applied to a real-world loan approval prediction problem using a reproducible and scalable architecture.

---

## Problem Statement

Financial institutions receive thousands of loan applications and must evaluate applicant information before making approval decisions. Manual assessment is time-consuming and may lead to inconsistencies.

This project aims to build a machine learning system capable of predicting loan approval status based on applicant demographics, income, credit history, and loan characteristics while implementing industry-standard MLOps practices.

---

## Objectives

* Build a loan approval prediction model using supervised machine learning.
* Implement automated data preprocessing and model training pipelines.
* Track experiments and model performance using MLflow.
* Version datasets and model artifacts using DVC.
* Deploy the model using FastAPI and Docker.
* Automate testing and validation using GitHub Actions.
* Monitor model health and data drift using Prometheus and Evidently AI.
* Perform security scanning using Trivy.

---

## Technology Stack

### Machine Learning

* Python
* Scikit-learn
* Logistic Regression

### MLOps

* DVC
* MLflow
* GitHub Actions

### Deployment

* FastAPI
* Docker

### Monitoring

* Prometheus
* Evidently AI

### DevSecOps

* Trivy

### Version Control

* Git
* GitHub

---

## Dataset Information

### Dataset

Loan Prediction Dataset

### Features

* Gender
* Married
* Dependents
* Education
* Self_Employed
* ApplicantIncome
* CoapplicantIncome
* LoanAmount
* Loan_Amount_Term
* Credit_History
* Property_Area

### Target Variable

* Loan_Status

### Dataset Size

* Rows: 614
* Features: 12
* Target Classes:

  * Approved (Y): 422
  * Rejected (N): 192

---

## Machine Learning Pipeline

1. Data Ingestion
2. Data Preprocessing
3. Missing Value Handling
4. Feature Encoding
5. Feature Scaling
6. Model Training
7. Model Evaluation
8. Model Versioning
9. Deployment
10. Monitoring

---

## Model Development

Three models were evaluated:

### Random Forest

Accuracy: 83.74%

### Random Forest + SMOTE

Accuracy: 80.48%

### Logistic Regression (Scaled)

Accuracy: 86.18%

The Scaled Logistic Regression model was selected as the final production model due to its superior performance, simplicity, and interpretability.

---

## Final Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 86.18% |
| Precision | 84.00% |
| Recall    | 98.82% |
| F1 Score  | 90.81% |

---

## MLOps Components

### DVC

* Dataset Versioning
* Model Versioning
* Reproducible Pipelines
* DVC DAG

### MLflow

* Experiment Tracking
* Model Comparison
* Metric Logging
* Artifact Management

### FastAPI

* Prediction Endpoint
* Health Endpoint
* Interactive Swagger Documentation

### Docker

* Containerized Deployment
* Environment Consistency
* Portable Execution

### GitHub Actions

* Automated Testing
* Linting
* Security Validation
* Docker Build Validation

### Prometheus

* API Request Monitoring
* Health Monitoring
* Metrics Collection

### Evidently AI

* Data Drift Detection
* Monitoring Dashboard
* Dataset Stability Analysis

### Trivy

* Container Vulnerability Scanning
* Automated Security Checks
* DevSecOps Integration

---

## Project Structure

```text
loan-risk-mlops/
├── api/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── monitoring/
├── reports/
├── security/
├── src/
├── static/
├── templates/
├── tests/
├── dvc.yaml
├── params.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Running the Project

### Clone Repository

```bash
git clone https://github.com/JNSathyasri/loan-risk-mlops.git
cd loan-risk-mlops
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run API

```bash
uvicorn api.main:app --reload
```

### Open Application

```text
http://127.0.0.1:8000
```

### Swagger Documentation

```text
http://127.0.0.1:8000/docs
```

---

## Monitoring and Security

### Monitoring

* Prometheus Metrics Endpoint
* API Logging
* Evidently AI Drift Monitoring

### Security

* Trivy Vulnerability Scanning
* Automated Security Validation in GitHub Actions

---

## Future Improvements

* Kubernetes Deployment
* Model Registry Integration
* Automated Retraining
* Cloud Deployment (AWS/Azure)
* Real-Time Monitoring Dashboard
* Canary Model Releases

---

## Author

J N Sathyasri
Loan Approval Prediction Platform using MLOps