import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model

# Load model
model = load_model("model.h5")

st.title("Customer Churn Prediction")

# Inputs
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_products = st.slider("Number of Products", 1, 4)
has_credit_card = st.selectbox("Has Credit Card", [0,1])
is_active = st.selectbox("Is Active Member", [0,1])

# Encoding
geo_dict = {"France":0, "Germany":1, "Spain":2}
gender_dict = {"Male":0, "Female":1}

geo = geo_dict[geography]
gen = gender_dict[gender]

# Input array
input_data = np.array([[credit_score, geo, gen, age, tenure,
                        balance, num_products, has_credit_card,
                        is_active, estimated_salary]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    prob = prediction[0][0]

    st.write(f"Churn Probability: {prob:.2f}")

    if prob > 0.5:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer is not likely to churn")

