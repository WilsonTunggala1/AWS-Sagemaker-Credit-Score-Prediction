"""
Streamlit UI for the Credit Score classifier hosted on SageMaker.

Reads endpoint name and region from environment variables.
boto3 picks up AWS credentials from the EC2 instance profile.
"""

import json
import os
import boto3
import streamlit as st
import pandas as pd
from botocore.exceptions import ClientError, NoCredentialsError

st.set_page_config(page_title="Credit Score Predictor", page_icon="💳", layout="wide")

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-score-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")

@st.cache_resource
def get_runtime_client():
    """Membuat koneksi ke AWS SageMaker Runtime"""
    return boto3.client("sagemaker-runtime", region_name=REGION)

def invoke_endpoint(features: list) -> dict:
    """Mengirim data ke AWS SageMaker Endpoint dalam format JSON"""
    runtime = get_runtime_client()
    payload = {"instances": [features]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))

def main():
    st.title('💳 Intelligent Credit Score Predictor (AWS Cloud)')
    st.markdown("Masukkan informasi finansial nasabah di bawah ini untuk memprediksi kategori *Credit Score* mereka langsung dari cloud AWS.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("👤 Profil Personal")
        month = st.selectbox("Month of Record", ["January", "February", "March", "April", "May", "June", "July", "August"])
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        occupation = st.selectbox("Occupation", [
            "Scientist", "Teacher", "Engineer", "Entrepreneur", "Developer", 
            "Lawyer", "Media_Manager", "Doctor", "Journalist", "Manager", 
            "Accountant", "Musician", "Mechanic", "Writer", "Architecture", "Other"
        ])
        
        st.header("💰 Informasi Pendapatan")
        annual_income = st.number_input("Annual Income ($)", min_value=0.0, value=50000.0, step=1000.0)
        monthly_salary = st.number_input("Monthly Inhand Salary ($)", min_value=0.0, value=(annual_income/12), step=100.0)
        monthly_balance = st.number_input("Monthly Balance ($)", min_value=0.0, value=500.0, step=100.0)
        amount_invested = st.number_input("Amount Invested Monthly ($)", min_value=0.0, value=100.0, step=50.0)

    with col2:
        st.header("🏦 Rekening & Kartu Kredit")
        num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, max_value=20, value=3)
        num_credit_card = st.number_input("Number of Credit Cards", min_value=0, max_value=15, value=2)
        interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=40.0, value=15.0)
        num_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0, max_value=20, value=2)
        credit_history_months = st.number_input("Credit History Age (in Months)", min_value=0.0, value=150.0)
        
        st.header("⚠️ Riwayat Keterlambatan")
        delay_due_date = st.number_input("Average Delay from Due Date (Days)", min_value=0, value=14)
        num_delayed_payment = st.number_input("Number of Delayed Payments", min_value=0, value=4)

    with col3:
        st.header("📉 Rasio & Perilaku Utang")
        outstanding_debt = st.number_input("Outstanding Debt ($)", min_value=0.0, value=1500.0, step=100.0)
        utilization_ratio = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=30.0)
        total_emi = st.number_input("Total EMI per Month ($)", min_value=0.0, value=200.0, step=50.0)
        changed_limit = st.number_input("Changed Credit Limit (%)", value=10.0)
        
        credit_mix = st.selectbox("Credit Mix", ["Bad", "Standard", "Good"])
        min_amount = st.selectbox("Payment of Minimum Amount", ["No", "NM", "Yes"])
        payment_behavior = st.selectbox("Payment Behaviour", [
            "Low_spent_Small_value_payments", "High_spent_Medium_value_payments", 
            "Low_spent_Medium_value_payments", "High_spent_Large_value_payments", 
            "High_spent_Small_value_payments", "Low_spent_Large_value_payments", "Other"
        ])

    st.markdown("---")
    st.header("📝 Portofolio Pinjaman (Loans)")
    loan_types = ["Auto_Loan", "Credit_Builder_Loan", "Debt_Consolidation_Loan", 
                  "Home_Equity_Loan", "Mortgage_Loan", "Not_Specified", "Payday_Loan", 
                  "Personal_Loan", "Student_Loan", "Unknown"]
    
    selected_loans = st.multiselect("Pilih semua jenis pinjaman yang dimiliki nasabah:", loan_types)

    # Mengeksekusi Prediksi
    if st.button("🔍 Make Prediction via SageMaker", type="primary"):
        loan_values = [(1 if loan in selected_loans else 0) for loan in loan_types]
        num_of_loan = len(selected_loans)

        model_occupation = "Unknown" if occupation == "Other" else occupation
        model_payment_behavior = "Unknown" if payment_behavior == "Other" else payment_behavior

        features = [
            month, age, model_occupation, annual_income, monthly_salary, num_bank_accounts, 
            num_credit_card, interest_rate, num_of_loan, delay_due_date, num_delayed_payment, 
            changed_limit, num_credit_inquiries, credit_mix, outstanding_debt, utilization_ratio, 
            min_amount, total_emi, amount_invested, model_payment_behavior, monthly_balance, 
            credit_history_months
        ]
        # Gabungkan daftar list fitur utama dengan list fitur pinjaman
        features.extend(loan_values)

        try:
            with st.spinner(f"Menghubungi AWS SageMaker Endpoint: {ENDPOINT_NAME}..."):
                result = invoke_endpoint(features)
                
        except NoCredentialsError:
            st.error("❌ Kredensial AWS tidak ditemukan. Pastikan EC2 Anda menggunakan IAM LabInstanceProfile yang tepat.")
        except ClientError as e:
            st.error(f"❌ Error dari AWS: {e.response['Error'].get('Message', str(e))}")
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan pada sistem: {e}")
        else:
            label = result["labels"][0]
            probs = result["probabilities"][0]

            # Menampilkan hasil
            st.markdown("### Hasil Prediksi SageMaker:")
            if label == "Good":
                st.success(f"Credit Score: **{label}** - Nasabah berisiko rendah.")
            elif label == "Standard":
                st.warning(f"Credit Score: **{label}** - Nasabah memiliki risiko menengah.")
            else:
                st.error(f"Credit Score: **{label}** - Nasabah berisiko tinggi (Ditolak).")

            st.write("Detail Probabilitas Model:")
            prob_dict = {"Poor": probs[0], "Standard": probs[1], "Good": probs[2]}
            
            chart_data = pd.DataFrame([prob_dict])
            st.bar_chart(chart_data)

if __name__ == "__main__":
    main()