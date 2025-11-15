import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow
from tensorflow.keras.models import load_model

model = load_model("model.keras")
with open("label_encoder_geo.pkl", "rb") as f:
    encoder_geo = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Customer Churn Prediction")

credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=40)
tenure = st.number_input("Tenure", min_value=0, max_value=10, value=3)
balance = st.number_input("Balance", min_value=0, value=60000)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0, value=50000)

input_data = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary
}

input_df = pd.DataFrame([input_data])

geo_encoded = encoder_geo.transform(input_df[["Geography"]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=encoder_geo.get_feature_names_out(["Geography"]))

input_df = pd.concat([input_df.drop("Geography", axis=1), geo_encoded_df], axis=1)

input_df["Gender"] = input_df["Gender"].apply(lambda x: 1 if x == "Female" else 0)

input_scaled = scaler.transform(input_df)

if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)
    predicted_probability = prediction[0][0]
    predicted_class = 1 if predicted_probability >= 0.5 else 0

    st.write(f"Predicted Probability: {predicted_probability:.4f}")
    st.write(f"Churn Prediction: {'Yes' if predicted_class == 1 else 'No'}")
