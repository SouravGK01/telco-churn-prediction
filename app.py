import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Telecom Churn Predictor", layout="centered")
st.title("📞 Telecom Customer Churn Prediction")
st.markdown("Upload Telco customer data (CSV) to predict churn likelihood.")

# Load model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

if os.path.exists("features.pkl"):
    model_features = joblib.load("features.pkl")
else:
    model_features = None  

uploaded_file = st.file_uploader("Upload Telco customer data (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Raw Input Data")
    st.dataframe(df.head())

    try:
        # Preprocess input
        df.replace(" ", np.nan, inplace=True)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df.dropna(inplace=True)

        # Dropping ID if exists
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

        # Matching training model features
        if model_features is not None:
            df_encoded = df_encoded.reindex(columns=model_features, fill_value=0)
        else:
            # Try best effort matching
            df_encoded = df_encoded.reindex(columns=model.feature_importances_.shape[0], fill_value=0)

        # Scale
        df_scaled = scaler.transform(df_encoded)

        # Predict
        preds = model.predict(df_scaled)
        probs = model.predict_proba(df_scaled)[:, 1]

        # Output results
        result_df = pd.DataFrame({
            "Customer Index": df.index + 1,
            "Prediction": ["Churn" if p == 1 else "No Churn" for p in preds],
            "Churn Probability (%)": (probs * 100).round(2)
        })

        st.subheader("📈 Prediction Results")
        st.dataframe(result_df)

    except Exception as e:
        st.error(f"⚠️ Error while processing: {e}")
