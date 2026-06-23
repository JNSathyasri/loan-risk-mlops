import os
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

print("Loading reference dataset.")

reference_data = pd.read_csv(
    "data/processed/processed_data_v2.csv"
)

print("Loading current dataset.")

current_data = pd.read_csv(
    "data/processed/test_data_v2.csv"
)

print("Generating drift report.")

report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

report.run(
    reference_data=reference_data,
    current_data=current_data
)

os.makedirs(
    "reports/drift",
    exist_ok=True
)

report.save_html(
    "reports/drift/drift_report.html"
)

print("Drift report saved successfully.")