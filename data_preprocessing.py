import os
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        
        # Mengganti column dari objek jd numerik
        self.cols_to_check = [
            'Age', 'Annual_Income', 'Num_of_Delayed_Payment', 
            'Changed_Credit_Limit', 'Outstanding_Debt', 
            'Amount_invested_monthly', 'Monthly_Balance',
            'Num_Bank_Accounts', 'Num_of_Loan', 'Delay_from_due_date'
        ]

    def _initial_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = ['Unnamed: 0', 'Name', 'ID', 'Customer_ID', 'SSN']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

        for col in self.cols_to_check:
            df[col] = df[col].astype(str).str.replace(r'_+', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['Occupation'] = df['Occupation'].replace({'_______': 'Unknown'})
        df['Payment_Behaviour'] = df['Payment_Behaviour'].replace({'!@9#%8': 'Unknown'})
        df['Credit_Mix'] = df['Credit_Mix'].replace('_', np.nan)

        return df

    def _handle_missing_and_outliers(self, train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Imputasi manual dan clipping outlier"""
        train['Monthly_Inhand_Salary'] = train['Monthly_Inhand_Salary'].fillna(train['Annual_Income'] / 12)
        test['Monthly_Inhand_Salary'] = test['Monthly_Inhand_Salary'].fillna(test['Annual_Income'] / 12)

        train['Type_of_Loan'] = train['Type_of_Loan'].fillna('Unknown')
        test['Type_of_Loan'] = test['Type_of_Loan'].fillna('Unknown')
        
        train['Age'] = train['Age'].clip(lower=18, upper=100)
        test['Age'] = test['Age'].clip(lower=18, upper=100)

        cols_cant_be_negative = [
            'Num_Bank_Accounts', 'Num_of_Loan', 
            'Num_of_Delayed_Payment', 'Delay_from_due_date'
        ]
        for col in cols_cant_be_negative:
            train[col] = train[col].clip(lower=0)
            test[col] = test[col].clip(lower=0)

        numeric_cols = train.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [c for c in numeric_cols if c != 'Credit_Score']

        for col in numeric_cols:
            lower_bound = train[col].quantile(0.01)
            upper_bound = train[col].quantile(0.99)
            
            train[col] = train[col].clip(lower=lower_bound, upper=upper_bound)
            test[col] = test[col].clip(lower=lower_bound, upper=upper_bound)

        return train, test

    def _feature_engineering(self, train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        def extract_months(df: pd.DataFrame) -> pd.DataFrame:
            extracted = df['Credit_History_Age'].str.extract(r'(\d+).*?(\d+)')
            extracted.columns = ['Years', 'Months']
            extracted = extracted.astype(float)
            df['Credit_History_Age_Months'] = (extracted['Years'] * 12) + extracted['Months']
            return df.drop(columns=['Credit_History_Age'])

        train = extract_months(train)
        test = extract_months(test)

        median_history = train['Credit_History_Age_Months'].median()
        train['Credit_History_Age_Months'] = train['Credit_History_Age_Months'].fillna(median_history)
        test['Credit_History_Age_Months'] = test['Credit_History_Age_Months'].fillna(median_history)

        def get_loan_dummies(df: pd.DataFrame) -> pd.DataFrame:
            clean_loans = (df['Type_of_Loan']
                           .str.replace('and ', '', regex=False)
                           .str.replace(', ', ',', regex=False))
            return clean_loans.str.get_dummies(sep=',')

        train_loans = get_loan_dummies(train)
        test_loans = get_loan_dummies(test)

        # Align test columns dengan train columns
        test_loans = test_loans.reindex(columns=train_loans.columns, fill_value=0)

        train = pd.concat([train.drop(columns=['Type_of_Loan']), train_loans], axis=1)
        test = pd.concat([test.drop(columns=['Type_of_Loan']), test_loans], axis=1)

        train.columns = train.columns.str.replace(' ', '_').str.replace('-', '_')
        test.columns = test.columns.str.replace(' ', '_').str.replace('-', '_')

        return train, test

    def run(self, data_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        print("Step 2: Data Preprocessing & Feature Engineering")
        
        df = pd.read_csv(Path(data_path))
        df = self._initial_cleaning(df)
        
        mapping = {'Poor': 0, 'Standard': 1, 'Good': 2}
        df['Credit_Score'] = df['Credit_Score'].map(mapping)
        
        train, test = train_test_split(df, test_size=self.test_size, random_state=self.random_state, stratify=df['Credit_Score'])
        
        train, test = self._handle_missing_and_outliers(train, test)
        train, test = self._feature_engineering(train, test)
        
        X_train = train.drop(columns=['Credit_Score'])
        y_train = train['Credit_Score']
        
        X_test = test.drop(columns=['Credit_Score'])
        y_test = test['Credit_Score']
        
        print(f"Preprocessing Selesai. Dimensi X_train: {X_train.shape}, X_test: {X_test.shape}\n")
        
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    if os.environ.get("SM_CHANNEL_INPUT") or os.path.exists("/opt/ml/processing"):
        # Path di dalam Container AWS Processing Job
        input_dir = "/opt/ml/processing/ingested"
        out_train = "/opt/ml/processing/train"
        out_test = "/opt/ml/processing/test"
    else:
        # Fallback jika dijalankan di lokal/Jupyter Notebook AWS
        input_dir = "/home/ec2-user/SageMaker/uas/ingested"
        out_train = "/home/ec2-user/SageMaker/uas/train"
        out_test = "/home/ec2-user/SageMaker/uas/test"

    # Pastikan direktori output tersedia
    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_test, exist_ok=True)

    input_file = os.path.join(input_dir, "data_A.csv")
    print(f"Mencari dataset di: {input_file}")
    
    if os.path.exists(input_file):
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.run(input_file)
        
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        train_path = os.path.join(out_train, "train.csv")
        test_path = os.path.join(out_test, "test.csv")
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"✅ Data berhasil diproses dan disimpan untuk tahapan Training!")
        print(f"-> Train path: {train_path} ({len(train_df)} rows)")
        print(f"-> Test path:  {test_path} ({len(test_df)} rows)")
    else:
        print(f"❌ Error: File input {input_file} tidak ditemukan!")