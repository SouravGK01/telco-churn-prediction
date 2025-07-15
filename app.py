import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="📞 Churn Predictor", layout="centered")
st.title("📞 Telecom Customer Churn Predictor")
st.markdown("Upload customer CSV and predict if they will churn.")

# Load model, scaler and feature list
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_list = joblib.load("features.pkl")  # List of columns used during training

uploaded_file = st.file_uploader("📤 Upload a CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Raw Input Data")
        st.dataframe(df.head())

        # === Minimal Preprocessing like training ===
        df.replace(" ", np.nan, inplace=True)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df.dropna(inplace=True)

        # Drop customerID if it exists
        df.drop("customerID", axis=1, inplace=True, errors="ignore")

        # Encode binaries
        binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map({"Yes": 1, "No": 0})
        if "gender" in df.columns:
            df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

        # One-hot encode all categoricals
        df_encoded = pd.get_dummies(df)

        # Reindex to match model's training features
        df_encoded = df_encoded.reindex(columns=feature_list, fill_value=0)

        # Scale
        df_scaled = scaler.transform(df_encoded)

        # Predict
        preds = model.predict(df_scaled)
        probs = model.predict_proba(df_scaled)[:, 1]

        # Display
        result_df = pd.DataFrame({
            "Prediction": ["Churn" if p == 1 else "No Churn" for p in preds],
            "Churn Probability (%)": (probs * 100).round(2)
        })

        st.subheader("📊 Prediction Results")
        st.dataframe(result_df)

    except Exception as e:
        st.error(f"🚨 Error processing the file:\n\n{e}")
