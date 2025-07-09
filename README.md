Telco Customer Churn Prediction

Welcome!   
This project is all about predicting whether a customer is likely to **churn** (leave a telecom company) based on their account details, usage, and services.
The goal? Help businesses take action *before* customers walk away.

---

## What This Project Does

We use real customer data to build a machine learning model that predicts whether a customer will churn. It includes:

- Cleaning and preparing the dataset
- Visualizing patterns (EDA)
- Training a model (Random Forest Classifier)
- Evaluating performance
- Saving the final model for future use

---

##  Dataset Info

- **Source:** IBM Telco Customer Churn Dataset  
- **Size:** ~7,000 records  
- **Target Column:** `Churn` (Yes/No)

Features include:
- Demographics (gender, age, senior citizen, etc.)
- Services (internet, phone, streaming, etc.)
- Charges and contract types

---

## Tools & Libraries Used

- **pandas** and **numpy** – for data handling
- **matplotlib** and **seaborn** – for visualization
- **scikit-learn** – for building and evaluating the model
- **joblib** – for saving the model
- **Google Colab** – for coding

---

## How the Model Works

1. **Preprocessing**  
   - Removed duplicates and missing values  
   - Converted categories to numbers using one-hot encoding  
   - Scaled numerical features for better model performance  

2. **Training**  
   - Used a Random Forest Classifier  
   - Tuned it with GridSearchCV for best performance  

3. **Evaluation**  
   - Accuracy Score  
   - ROC AUC  
   - Confusion Matrix  
   - Cross-validation  
   - Feature Importance

4. **Visualization**  
   - Who churns more? What’s their tenure? Monthly charges?  
   - Correlation heatmaps to understand feature relationships  

5. **Saving the Model**  
   - Final model and scaler are saved as `.pkl` files so you can reuse them!

