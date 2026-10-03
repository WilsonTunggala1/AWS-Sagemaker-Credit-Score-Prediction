import os
from pathlib import Path
from typing import Tuple
import joblib
import pandas as pd
import numpy as np

# Scikit-learn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_validate

# Models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

class CreditScoreTrainer:
    """Handles scikit-learn preprocessing, model training, and evaluation without MLflow."""
    
    def __init__(self, cv_splits: int = 5, random_state: int = 42):
        self.cv_splits = cv_splits
        self.random_state = random_state
        
        self.nominal_features = ['Month', 'Occupation', 'Payment_Behaviour']
        self.ordinal_features = ['Credit_Mix', 'Payment_of_Min_Amount']
        self.numeric_features = [
            'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
            'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 
            'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries', 
            'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month', 
            'Amount_invested_monthly', 'Monthly_Balance', 'Credit_History_Age_Months'
        ]
        self.binary_features = [
            'Auto_Loan', 'Credit_Builder_Loan', 'Debt_Consolidation_Loan', 
            'Home_Equity_Loan', 'Mortgage_Loan', 'Not_Specified', 'Payday_Loan', 
            'Personal_Loan', 'Student_Loan', 'Unknown'
        ]

    def _get_transformer(self) -> ColumnTransformer:
        """Membangun ColumnTransformer untuk pipeline Scikit-Learn."""
        onehot_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')), 
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        ordinal_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(
                categories=[['Bad', 'Standard', 'Good'], ['No', 'NM', 'Yes']],
                handle_unknown='use_encoded_value', unknown_value=-1
            ))
        ])

        robust_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        
        binary_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])

        return ColumnTransformer([
            ('onehot', onehot_pipeline, self.nominal_features),
            ('ordinal', ordinal_pipeline, self.ordinal_features),
            ('num', robust_pipeline, self.numeric_features),
            ('binary', binary_pipeline, self.binary_features)
        ], remainder='drop')

    def _get_models(self) -> dict:
        return {
            'RandomForest_Tuned': RandomForestClassifier(
                n_estimators=200,
                max_depth=24,
                min_samples_split=5,
                min_samples_leaf=1,
                class_weight=None,
                random_state=self.random_state, 
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(random_state=self.random_state, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1),
            'LightGBM': LGBMClassifier(random_state=self.random_state, n_jobs=-1, verbose=-1),
            'CatBoost': CatBoostClassifier(random_state=self.random_state, verbose=False, thread_count=-1, allow_writing_files=False)
        }

    def run(self, X_train: pd.DataFrame, y_train: pd.Series, model_dir: str) -> str:
        """
        Menjalankan training untuk semua model, memilih yang terbaik, 
        lalu menyimpan model terbaik tersebut ke direktori SageMaker.
        """
        print("Model Training & Selection")
        transformer = self._get_transformer()
        models = self._get_models()
        
        skf = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
        scoring = {'accuracy': 'accuracy', 'f1_weighted': 'f1_weighted'}
        
        best_model_name = ""
        best_f1_score = 0.0
        best_pipeline = None

        for model_name, model in models.items():
            print(f"Training & Evaluating {model_name}")
            
            clf_pipeline = Pipeline([
                ('preprocessor', transformer),
                ('classifier', model)
            ])
                
            # Cross Validation Evaluation
            cv_scores = cross_validate(clf_pipeline, X_train, y_train, cv=skf, scoring=scoring, n_jobs=1)
            mean_acc = np.mean(cv_scores['test_accuracy'])
            mean_f1 = np.mean(cv_scores['test_f1_weighted'])
            
            if mean_f1 > best_f1_score:
                best_f1_score = mean_f1
                best_model_name = model_name
                best_pipeline = clf_pipeline

            print(f"   -> Acc: {mean_acc:.4f} | F1: {mean_f1:.4f}")

        print("\n🏆 PENCARIAN MODEL TERBAIK SELESAI!")
        print(f"⭐ Model Terbaik: {best_model_name} dengan F1-Score: {best_f1_score:.4f}")
        
        print(f"Melatih {best_model_name} menggunakan seluruh data training...")
        best_pipeline.fit(X_train, y_train)

        best_model_path = os.path.join(model_dir, "best_credit_score_pipeline.joblib")
        joblib.dump(best_pipeline, best_model_path)
        print(f"✅ Model terbaik disimpan di: {best_model_path}")
        
        return best_model_name


if __name__ == "__main__":
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/home/ec2-user/SageMaker/uas/train")
    model_dir = os.environ.get("SM_MODEL_DIR", "/home/ec2-user/SageMaker/uas/model")
    
    os.makedirs(model_dir, exist_ok=True)
    train_file = os.path.join(train_dir, "train.csv")
    
    print(f"Mencari file training di: {train_file}")
    
    if os.path.exists(train_file):
        df = pd.read_csv(train_file)
        
        target_col = "Credit_Score" 
        X_train = df.drop(columns=[target_col])
        y_train = df[target_col]
        
        trainer = CreditScoreTrainer()
        trainer.run(X_train, y_train, model_dir)