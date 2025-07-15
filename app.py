import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config
st.set_page_config(page_title="📞 Churn Predictor", layout="centered")
st.title("📞 Telecom Customer Churn Predictor")
st.markdown("Upload customer CSV and predict if they are likely to churn.")

# Load model, scaler and feature list
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")  # This must match training features

# File upload
uploaded_file = st.file_uploader("📤 Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Raw Input Data")
        st.dataframe(df.head())

        # Preprocessing (match training)
        df.replace(" ", np.nan, inplace=True)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df.dropna(inplace=True)
        df.drop("customerID", axis=1, inplace=True, errors="ignore")

        # Encode binary columns
        binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map({"Yes": 1, "No": 0})

        if "gender" in df.columns:
            df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

        # One-hot encode
        df_encoded = pd.get_dummies(df)

        # Align with training features
        df_encoded = df_encoded.reindex(columns=features, fill_value=0)

        # Scale
        df_scaled = scaler.transform(df_encoded)

        # Predict
        predictions = model.predict(df_scaled)
        probabilities = model.predict_proba(df_scaled)[:, 1]

        # Display results
        result_df = pd.DataFrame({
            "Prediction": ["Churn" if p == 1 else "No Churn" for p in predictions],
            "Churn Probability (%)": (probabilities * 100).round(2)
        })

        st.subheader("📊 Prediction Results")
        st.dataframe(result_df)

    except Exception as e:
        st.error(f"🚨 Error processing the file:\n\n{e}")
