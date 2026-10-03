"""SageMaker inference entry point for Credit Score Prediction.

Generic sklearn Pipeline loader. Works for any pipeline saved as .joblib,
regardless of which classifier is inside (RandomForest, XGBoost, CatBoost, dll).

Four functions form the SageMaker contract:
    model_fn   - load model from disk (called once per container)
    input_fn   - parse request body (called per request)
    predict_fn - run inference (called per request)
    output_fn  - serialize response (called per request)
"""

import json
import os
import io

import joblib
import numpy as np
import pandas as pd


JSON_CONTENT_TYPE = "application/json"
CSV_CONTENT_TYPE = "text/csv"

# Mapping: 0 -> Poor, 1 -> Standard, 2 -> Good
CLASS_NAMES = ["Poor", "Standard", "Good"]

FEATURE_NAMES = [
    "Month",
    "Age",
    "Occupation",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Credit_Mix",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Payment_of_Min_Amount",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Payment_Behaviour",
    "Monthly_Balance",
    "Credit_History_Age_Months",
    "Auto_Loan", 
    "Credit_Builder_Loan", 
    "Debt_Consolidation_Loan", 
    "Home_Equity_Loan", 
    "Mortgage_Loan", 
    "Not_Specified", 
    "Payday_Loan", 
    "Personal_Loan", 
    "Student_Loan", 
    "Unknown"
]


def model_fn(model_dir: str):
    """Load the pickled sklearn Pipeline from best_credit_score_pipeline.joblib."""
    # Menyesuaikan dengan nama file yang disimpan di train.py
    model_path = os.path.join(model_dir, "best_credit_score_pipeline.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)


def input_fn(request_body, request_content_type: str) -> pd.DataFrame:
    """Parse incoming request body into a DataFrame.

    Accepts JSON: {"instances": [[...32 values...], ...]}
    Or CSV: one row per instance, 32 comma-separated values (mixed strings/numbers).
    """
    if request_content_type == JSON_CONTENT_TYPE:
        payload = json.loads(request_body)
        instances = payload["instances"]
        return pd.DataFrame(instances, columns=FEATURE_NAMES)

    if request_content_type == CSV_CONTENT_TYPE:
        if isinstance(request_body, (bytes, bytearray)):
            request_body = request_body.decode("utf-8")
            
        return pd.read_csv(
            io.StringIO(request_body.strip()), 
            header=None, 
            names=FEATURE_NAMES
        )

    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data: pd.DataFrame, pipeline) -> dict:
    """Run inference. Returns probabilities, predicted class IDs, and labels."""
    # Pipeline scikit-learn secara otomatis memanggil preprocessor -> classifier
    probs = pipeline.predict_proba(input_data)
    class_ids = np.argmax(probs, axis=1)
    
    # Konversi ID (0, 1, 2) menjadi nama kelas yang relevan
    labels = [CLASS_NAMES[int(i)] for i in class_ids]
    
    return {
        "probabilities": probs.tolist(),
        "predictions": class_ids.tolist(),
        "labels": labels,
    }


def output_fn(prediction: dict, accept_content_type: str):
    """Serialize the prediction dict for the response body."""
    if accept_content_type == JSON_CONTENT_TYPE:
        return json.dumps(prediction), JSON_CONTENT_TYPE
    
    raise ValueError(f"Unsupported accept type: {accept_content_type}")