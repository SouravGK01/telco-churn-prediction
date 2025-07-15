import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Telecom Churn Predictor", layout="centered")
st.title("📞 Telecom Customer Churn Prediction")
st.markdown("Upload Telco customer data (CSV) to predict churn likelihood.")

# Load trained model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# Hardcoded list of features used during training
model_features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
    'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

uploaded_file = st.file_uploader("📤 Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Raw Input Data")
        st.dataframe(df.head())

        # Preprocessing
        df.replace(" ", np.nan, inplace=True)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df.dropna(inplace=True)
        df.drop("customerID", axis=1, inplace=True, errors="ignore")

        # Binary encoding
        binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map({"Yes": 1, "No": 0})
        if "gender" in df.columns:
            df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

        # One-hot encode
        df_encoded = pd.get_dummies(df)

        # Align columns to match model training
        df_encoded = df_encoded.reindex(columns=model_features, fill_value=0)

        # Scale
        df_scaled = scaler.transform(df_encoded)

        # Predict
        preds = model.predict(df_scaled)
        probs = model.predict_proba(df_scaled)[:, 1]

        result_df = pd.DataFrame({
            "Prediction": ["Churn" if p == 1 else "No Churn" for p in preds],
            "Churn Probability (%)": (probs * 100).round(2)
        })

        st.subheader("📊 Prediction Results")
        st.dataframe(result_df)

    except Exception as e:
        st.error(f"🚨 Error processing the file: {e}")
