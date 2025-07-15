import streamlit as st
import pandas as pd
import joblib

st.title("📱 Telecom Churn Predictor")

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Raw Input Data")
    st.write(data.head())

    data_encoded = pd.get_dummies(data)
    model_features = model.feature_names_in_
    data_encoded = data_encoded.reindex(columns=model_features, fill_value=0)
    data_scaled = scaler.transform(data_encoded)

    prediction = model.predict(data_scaled)
    prediction_prob = model.predict_proba(data_scaled)[:, 1]

    result_df = pd.DataFrame({
        'Customer': data.get('customerID', data.index),
        'Churn Prediction': prediction,
        'Churn Probability': prediction_prob
    })

    st.write("🔮 Prediction Results")
    st.dataframe(result_df)
