import streamlit as st
from src.inferance import predict

st.title("Diabetes Risk Predictor- 5 min DL")

age = st.slider("Age", 18, 90)
sex = st.selectbox("Sex", ["Male", "Female"])
ethnicity = st.selectbox(
    "Ethnicity",
    ["White", "Black", "Asian", "Hispanic"]
)
bmi = st.slider("BMI", 10.0, 50.0)
alcohol = st.selectbox(
    "Alcohol Consumption",
    ["None", "Low", "Moderate", "Heavy"]
)
smoking = st.selectbox(
    "Smoking Status",
    ["Never", "Former", "Current"]
)
family = st.selectbox(
    "Family History",
    [0,1]
)

if st.button("Predict"):
    data = {
        "Age":age,
        "Sex":sex,
        "Ethnicity":ethnicity,
        "BMI":bmi,
        "Alcohol_Consumption":alcohol,
        "Smoking_Status":smoking,
        "Family_History_of_Diabetes":family
    }
    prob = predict(data)
    st.success(f"Diabetes Probability = {prob:.3f}")