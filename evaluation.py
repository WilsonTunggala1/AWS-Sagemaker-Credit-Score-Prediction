import os
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

def evaluate_model():
    if os.environ.get("SM_CHANNEL_TEST") or os.path.exists("/opt/ml/processing"):
        model_path = "/opt/ml/processing/model/best_credit_score_pipeline.joblib"
        test_path = "/opt/ml/processing/test/test.csv"
        output_dir = "/opt/ml/processing/evaluation"
    else:
        model_path = "/home/ec2-user/SageMaker/uas/model/best_credit_score_pipeline.joblib"
        test_path = "/home/ec2-user/SageMaker/uas/test/test.csv"
        output_dir = "/home/ec2-user/SageMaker/uas/eval"

    os.makedirs(output_dir, exist_ok=True)

    print(f"Mencari model di: {model_path}")
    print(f"Mencari data test di: {test_path}")
    if os.path.exists(model_path) and os.path.exists(test_path):
        
        print("📥 Memuat model terbaik")
        model = joblib.load(model_path)
        test_df = pd.read_csv(test_path)
        
        target_col = "Credit_Score"
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        print("Melakukan prediksi pada Unseen Test Data...")
        predictions = model.predict(X_test)
        
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, average="weighted", zero_division=0)
        rec = recall_score(y_test, predictions, average="weighted", zero_division=0)

        target_names = ['Poor', 'Standard', 'Good']
        print("\nClassification Report (Test Data):")
        print(classification_report(y_test, predictions, target_names=target_names))

        report_dict = {
            "classification_metrics": {
                "accuracy": {
                    "value": acc,
                    "standard_deviation": "NaN"
                },
                "precision": {
                    "value": prec,
                    "standard_deviation": "NaN"
                },
                "recall": {
                    "value": rec,
                    "standard_deviation": "NaN"
                }
            }
        }
        output_file_path = os.path.join(output_dir, "evaluation.json")
        with open(output_file_path, "w") as f:
            json.dump(report_dict, f, indent=4)
            
        print("Evaluasi: ")
        print(f"Test Accuracy  = {acc:.4f}")
        print(f"Test Precision = {prec:.4f}")
        print(f"Test Recall    = {rec:.4f}")
        print(f"Report JSON disimpan di: {output_file_path}")
        
    else:
        print("❌ Error: File yang dibutuhkan untuk evaluasi tidak lengkap!")
        if not os.path.exists(model_path): print(f"-> Model hilang di: {model_path}")
        if not os.path.exists(test_path): print(f"-> Data Test hilang di: {test_path}")

if __name__ == "__main__":
    evaluate_model()