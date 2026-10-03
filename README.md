# Credit Score Prediction with AWS SageMaker

An end-to-end machine learning project for predicting customer credit score categories using **Python, Scikit-learn, AWS SageMaker, Amazon EC2, and Streamlit**.

The project demonstrates the complete machine learning lifecycle: data ingestion, preprocessing, feature engineering, model training and evaluation, cloud deployment, and real-time inference through a web application.

> **Deployment Status**
>
> The AWS infrastructure originally used for this project is no longer active.
> The repository preserves the source code, machine learning pipeline, and deployment workflow used during development.

---

## Overview

The goal of this project is to build a machine learning system capable of classifying a customer's credit score into one of three categories:

- **Poor**
- **Standard**
- **Good**

The system processes customer financial and credit information, performs data cleaning and feature engineering, trains and compares multiple machine learning models, selects the best-performing model, and prepares it for deployment through an **AWS SageMaker real-time endpoint**.

A **Streamlit web application** acts as the user interface and communicates with the SageMaker endpoint to perform real-time credit score predictions.

---

## Features

- Data ingestion
- Data preprocessing and cleaning
- Missing-value handling
- Invalid-value handling
- Outlier handling
- Feature engineering
- Stratified train/test splitting
- Multiple machine learning model comparison
- Stratified K-Fold cross-validation
- Automated best-model selection
- Model serialization with Joblib
- Model evaluation
- SageMaker-compatible inference logic
- Model packaging for AWS SageMaker
- Real-time cloud inference
- Streamlit web interface
- EC2 startup automation with `systemd`

---

## System Architecture

```text
                         Raw Dataset
                             |
                             v
                       Data Ingestion
                             |
                             v
                    Data Preprocessing
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
           Cleaning       Outlier        Feature
                          Handling      Engineering
              \              |              /
               \             |             /
                +------------+------------+
                             |
                             v
                    Train / Test Split
                             |
                             v
                       Model Training
                             |
          +------------------+------------------+------------------+
          |                  |                  |                  |
          v                  v                  v                  v
    Random Forest         XGBoost           LightGBM            CatBoost
          \                  |                  |                  /
           +-----------------+------------------+-----------------+
                             |
                             v
                Stratified Cross-Validation
                             |
                             v
                   Best Model Selection
                             |
                             v
                     Model Evaluation
                             |
                             v
          best_credit_score_pipeline.joblib
                             |
                             v
                       model.tar.gz
                             |
                             v
                         Amazon S3
                             |
                             v
                      AWS SageMaker
                             |
                             v
                Real-Time SageMaker Endpoint
                             |
                             v
                  Streamlit Application
                             |
                             v
                            User
```

---

## Machine Learning Pipeline

### 1. Data Ingestion

The data ingestion process is implemented in:

```text
data_ingestion.py
```

The script loads the source dataset and prepares it for the next stage of the machine learning pipeline.

It supports both:

- AWS/SageMaker processing paths
- Local development paths

The ingested dataset is written to:

```text
ingested/data_A.csv
```

The generated `ingested/` directory is excluded from Git because it can be recreated by running the pipeline.

---

### 2. Data Preprocessing

The preprocessing stage is implemented in:

```text
data_preprocessing.py
```

The preprocessing pipeline performs several operations to prepare the dataset for model training.

#### Data Cleaning

Unnecessary identifier columns are removed, including:

```text
Unnamed: 0
Name
ID
Customer_ID
SSN
```

Incorrectly formatted numeric fields are also converted into numerical values.

#### Invalid Value Handling

Invalid categorical values are converted into cleaner representations.

Examples:

| Feature | Replacement |
|---|---|
| `Occupation` | `Unknown` |
| `Payment_Behaviour` | `Unknown` |
| `Credit_Mix` | Missing value |

#### Missing Value Handling

Missing values are handled according to the corresponding feature.

For example, missing monthly salary values can be estimated using:

```text
Annual Income / 12
```

Missing loan information is represented as:

```text
Unknown
```

#### Outlier Handling

Numerical values are constrained to reduce the effect of invalid or extreme observations.

For example:

```text
Age: 18-100
```

Features that cannot logically contain negative values are clipped at zero.

Additional numerical outliers are clipped using percentile-based boundaries.

#### Feature Engineering

Several derived features are created before training.

##### Credit History Age

Credit history represented in years and months is converted into total months.

Example:

```text
5 Years and 6 Months
```

becomes:

```text
66 months
```

##### Loan Type Encoding

Customers may have multiple loan types. These are transformed into individual binary features such as:

- `Auto_Loan`
- `Credit_Builder_Loan`
- `Debt_Consolidation_Loan`
- `Home_Equity_Loan`
- `Mortgage_Loan`
- `Not_Specified`
- `Payday_Loan`
- `Personal_Loan`
- `Student_Loan`
- `Unknown`

##### Target Encoding

The `Credit_Score` target variable is encoded as:

| Credit Score | Class |
|---|---:|
| Poor | `0` |
| Standard | `1` |
| Good | `2` |

#### Train/Test Split

The processed dataset is divided into training and testing datasets using a stratified split.

Generated files:

```text
train/train.csv
test/test.csv
```

These directories are generated during execution and excluded from Git.

---

### 3. Model Training

Model training is implemented in:

```text
train.py
```

The training process compares several machine learning classification algorithms.

#### Models

The following models are evaluated:

- Random Forest
- XGBoost
- LightGBM
- CatBoost

#### Scikit-learn Pipeline

A Scikit-learn `Pipeline` and `ColumnTransformer` combine feature preprocessing and model prediction into one reusable workflow.

This ensures that the same preprocessing operations used during training are automatically applied during inference.

#### Feature Processing

Different preprocessing techniques are applied according to feature type.

##### Nominal Features

Examples:

- `Month`
- `Occupation`
- `Payment_Behaviour`

Processing:

```text
SimpleImputer
     |
     v
OneHotEncoder
```

##### Ordinal Features

Examples:

- `Credit_Mix`
- `Payment_of_Min_Amount`

Processing:

```text
OrdinalEncoder
```

##### Numerical Features

Examples:

- `Age`
- `Annual_Income`
- `Monthly_Inhand_Salary`
- `Num_Bank_Accounts`
- `Interest_Rate`
- `Outstanding_Debt`
- `Credit_Utilization_Ratio`
- `Monthly_Balance`

Processing:

```text
SimpleImputer
     |
     v
RobustScaler
```

##### Binary Features

Encoded loan features are passed through a dedicated preprocessing pipeline.

Examples:

- `Auto_Loan`
- `Mortgage_Loan`
- `Student_Loan`
- `Personal_Loan`

#### Model Selection

The models are evaluated using **Stratified K-Fold Cross-Validation**.

The training process calculates:

- Accuracy
- Weighted F1 score

The model with the highest weighted F1 score is selected as the final model and retrained using the complete training dataset.

The final pipeline is saved as:

```text
model/best_credit_score_pipeline.joblib
```

The trained model file is not stored in Git because it can be regenerated through the training pipeline.

---

### 4. Model Evaluation

Model evaluation is implemented in:

```text
evaluation.py
```

The selected model is evaluated against unseen test data.

The following metrics are calculated:

- Accuracy
- Weighted precision
- Weighted recall
- Classification report

The evaluation result is exported to:

```text
eval/evaluation.json
```

Example structure:

```json
{
  "classification_metrics": {
    "accuracy": {
      "value": 0.0,
      "standard_deviation": "NaN"
    },
    "precision": {
      "value": 0.0,
      "standard_deviation": "NaN"
    },
    "recall": {
      "value": 0.0,
      "standard_deviation": "NaN"
    }
  }
}
```

The `eval/` directory is generated during execution and excluded from Git.

---

## SageMaker Inference

The SageMaker inference implementation is located in:

```text
inference.py
```

AWS SageMaker uses four primary inference functions:

- `model_fn()`
- `input_fn()`
- `predict_fn()`
- `output_fn()`

### `model_fn()`

Loads the trained Scikit-learn pipeline from:

```text
best_credit_score_pipeline.joblib
```

The model is loaded once when the SageMaker inference container starts.

### `input_fn()`

Converts incoming requests into a Pandas DataFrame.

Supported content types:

- `application/json`
- `text/csv`

Example JSON request:

```json
{
  "instances": [
    [
      "January",
      30,
      "Engineer",
      50000,
      4166.67,
      3,
      2,
      15,
      2,
      14,
      4,
      10,
      2,
      "Standard",
      1500,
      30,
      "NM",
      200,
      100,
      "Low_spent_Small_value_payments",
      500,
      150,
      1,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ]
  ]
}
```

### `predict_fn()`

Runs the trained model against the incoming data and returns:

- Prediction probabilities
- Predicted class ID
- Human-readable label

### `output_fn()`

Serializes the model prediction into JSON before returning the response to the client.

Example response:

```json
{
  "probabilities": [
    [
      0.05,
      0.80,
      0.15
    ]
  ],
  "predictions": [
    1
  ],
  "labels": [
    "Standard"
  ]
}
```

Class mapping:

| Class | Label |
|---:|---|
| `0` | Poor |
| `1` | Standard |
| `2` | Good |

---

## AWS Deployment

The trained model was packaged for AWS SageMaker as:

```text
model.tar.gz
```

The deployment process used:

- Amazon S3
- AWS SageMaker
- SageMaker Scikit-learn container
- SageMaker real-time endpoint
- AWS IAM
- Boto3

The deployment workflow was implemented in:

```text
deploy_endpoint.ipynb
```

### Deployment Workflow

```text
best_credit_score_pipeline.joblib
               |
               v
          model.tar.gz
               |
               v
           Amazon S3
               |
               v
         SageMaker Model
               |
               v
      Endpoint Configuration
               |
               v
        SageMaker Endpoint
               |
               v
       Real-Time Inference
```

The generated `model.tar.gz` file is excluded from Git because it is a deployment artifact.

---

## Streamlit Web Application

The frontend application is implemented in:

```text
app_streamlit.py
```

The application provides a web interface for interacting with the deployed SageMaker model.

Users can provide information such as:

- Month
- Age
- Occupation
- Annual income
- Monthly salary
- Number of bank accounts
- Number of credit cards
- Interest rate
- Number of loans
- Payment delays
- Credit limit changes
- Number of credit inquiries
- Credit mix
- Outstanding debt
- Credit utilization ratio
- EMI amount
- Investment amount
- Payment behaviour
- Monthly balance
- Credit history age
- Loan types

### Prediction Flow

```text
User
 |
 v
Streamlit Form
 |
 v
Feature Vector
 |
 v
Boto3 SageMaker Runtime Client
 |
 v
AWS SageMaker Endpoint
 |
 v
ML Pipeline
 |
 v
Prediction
 |
 v
Streamlit Result
```

### Prediction Output

The application displays:

- Predicted credit score
- Customer risk level
- Prediction probability distribution

Example:

```text
Credit Score: Standard
Risk Level: Medium

Prediction Probabilities
Poor:      10%
Standard:  75%
Good:      15%
```

---

## EC2 Deployment

The project contains:

```text
user-data.sh
```

This script was designed to automate deployment of the Streamlit application on an Amazon EC2 instance.

The script:

1. Updates the EC2 instance.
2. Installs Python.
3. Installs Git.
4. Clones the GitHub repository.
5. Creates a Python virtual environment.
6. Installs application dependencies.
7. Configures SageMaker endpoint environment variables.
8. Creates a `systemd` service.
9. Starts the Streamlit application automatically.

### Streamlit `systemd` Service

The EC2 instance runs Streamlit as a Linux service.

The application listens on:

```text
0.0.0.0:8501
```

Using `systemd` allows the application to:

- Start automatically after EC2 boot
- Restart automatically if the process fails
- Run independently of an SSH session

---

## Project Structure

The repository is organized approximately as follows:

```text
.
├── app_streamlit.py
├── data_A.csv
├── data_ingestion.py
├── data_preprocessing.py
├── deploy_endpoint.ipynb
├── evaluation.py
├── inference.py
├── train.py
├── user-data.sh
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

The following directories and artifacts are generated during execution and are therefore excluded from version control:

```text
__pycache__/
.ipynb_checkpoints/

ingested/
train/
test/
eval/

*.joblib
*.pkl
*.pickle

model.tar
model.tar.gz
```

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost, LightGBM, CatBoost |
| Cloud | AWS SageMaker, Amazon S3, Amazon EC2, AWS IAM, Boto3 |
| Web Application | Streamlit |
| Model Serialization | Joblib |
| Development Tools | Jupyter Notebook, Git, GitHub, Bash, systemd |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```bat
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Machine Learning Pipeline

The project can be executed step-by-step.

### Step 1 — Data Ingestion

```bash
python data_ingestion.py
```

Generates:

```text
ingested/data_A.csv
```

### Step 2 — Data Preprocessing

```bash
python data_preprocessing.py
```

Generates:

```text
train/train.csv
test/test.csv
```

### Step 3 — Model Training

```bash
python train.py
```

The script:

1. Loads the training dataset.
2. Builds the preprocessing pipeline.
3. Trains multiple classifiers.
4. Performs cross-validation.
5. Selects the model with the best weighted F1 score.
6. Retrains the selected model.
7. Saves the complete pipeline.

Generated model:

```text
model/best_credit_score_pipeline.joblib
```

### Step 4 — Model Evaluation

```bash
python evaluation.py
```

Generates:

```text
eval/evaluation.json
```

---

## Running the Streamlit Application

Start Streamlit with:

```bash
streamlit run app_streamlit.py
```

By default, Streamlit is accessible at:

```text
http://localhost:8501
```

> The frontend can run locally, but cloud predictions require an active SageMaker endpoint.

---

## AWS Configuration

The Streamlit application reads SageMaker endpoint information from environment variables.

Required variables:

```env
ENDPOINT_NAME=credit-score-endpoint
AWS_REGION=us-east-1
```

### Linux / macOS

```bash
export ENDPOINT_NAME="credit-score-endpoint"
export AWS_REGION="us-east-1"
```

### Windows PowerShell

```powershell
$env:ENDPOINT_NAME="credit-score-endpoint"
$env:AWS_REGION="us-east-1"
```

---

## AWS Credentials

The application uses Boto3 to communicate with AWS.

When running on Amazon EC2, credentials should preferably be provided through an **IAM instance role** rather than storing AWS access keys directly in the repository.

Never commit credentials such as:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_SESSION_TOKEN
```

Local credentials can instead be configured using:

```bash
aws configure
```

or another secure AWS credential provider.

### Environment Example

An optional `.env.example` file can contain:

```env
ENDPOINT_NAME=credit-score-endpoint
AWS_REGION=us-east-1
```

Do not place real AWS credentials inside `.env.example`.

---

## AWS Deployment Status

The AWS resources originally used for this project are no longer active.

Resources that may no longer exist include:

- SageMaker endpoint
- SageMaker endpoint configuration
- S3 model artifact
- EC2 instance

As a result, running:

```bash
streamlit run app_streamlit.py
```

will start the frontend, but cloud predictions require a newly deployed SageMaker endpoint.

The deployment code remains in the repository to demonstrate the original cloud architecture and deployment workflow.

---

## What I Learned

This project provided hands-on experience across data engineering, machine learning, deployment, and application development.

### Data Engineering

- Loading and validating datasets
- Data cleaning
- Missing-value handling
- Outlier handling
- Feature engineering
- Train/test dataset preparation

### Machine Learning

- Scikit-learn pipelines
- `ColumnTransformer`
- One-hot encoding
- Ordinal encoding
- Feature scaling
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Stratified cross-validation
- Multi-class classification
- Model comparison
- Model selection

### MLOps / Deployment

- Model serialization
- Model artifact packaging
- Amazon S3 model storage
- AWS SageMaker deployment
- SageMaker inference handlers
- SageMaker real-time endpoints
- Boto3
- EC2 deployment
- Linux `systemd`
- Environment variable configuration

### Application Development

- Building a Streamlit interface
- Sending real-time inference requests
- Handling prediction results
- Displaying model probability distributions

---

## Future Improvements

Possible improvements include:

- Add automated unit and integration tests
- Add GitHub Actions CI/CD
- Containerize the application with Docker
- Use Infrastructure as Code with Terraform
- Add MLflow experiment tracking
- Add model versioning
- Add dataset versioning
- Add model monitoring
- Add data drift detection
- Add prediction drift detection
- Add centralized logging
- Add automated SageMaker deployment
- Add SageMaker Pipelines
- Add application authentication
- Add API validation
- Separate development and production configuration
- Add automated dependency security scanning

---

## Security Considerations

When deploying the project publicly:

- Never commit AWS credentials.
- Never commit `.env`.
- Use IAM roles whenever possible.
- Apply the principle of least privilege to IAM policies.
- Restrict EC2 security group ports.
- Avoid exposing the SageMaker endpoint directly.
- Store production secrets using AWS Secrets Manager or AWS Systems Manager Parameter Store.

---

## Disclaimer

This project was created for educational and machine learning deployment practice.

The predictions generated by this model are **not intended for actual credit approval, lending, financial risk assessment, or other real-world financial decisions**.

---

## Author

Developed as part of a machine learning and AWS deployment exercise.
