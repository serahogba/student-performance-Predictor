# file name: 4.10_app_predictor.py
# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------------------------------------------------
# Load trained model
model = joblib.load("../../models/clf_pipeline_v1.joblib")
# ----------------------------------------------------------------------------------------------
# Streamlit App Interface for Student Performance Prediction
st.title("🎓 Student Performance Predictor")
st.write(
    "This app predicts whether a student is likely to Pass or Fail based on academic and socio-economic attributes."
)

# Input form for student attributes
study_hours = st.slider("Study Hours per Week", 0, 50, 20)
attendance = st.slider("Attendance Rate (%)", 0, 100, 75)
past_scores = st.number_input("Past Exam Score", 0, 100, 70)
gender = st.selectbox("Gender", ["Male", "Female"])
parent_edu = st.selectbox(
    "Parental Education Level", ["High School", "Bachelors", "Masters", "PhD"]
)
internet = st.selectbox("Internet Access at Home", ["Yes", "No"])
extra = st.selectbox("Participates in Extracurricular Activities", ["Yes", "No"])

# Create DataFrame for prediction
input_data = pd.DataFrame(
    {
        "Gender": [gender],
        "Study_Hours_per_Week": [study_hours],
        "Attendance_Rate": [attendance],
        "Past_Exam_Scores": [past_scores],
        "Parental_Education_Level": [parent_edu],
        "Internet_Access_at_Home": [internet],
        "Extracurricular_Activities": [extra],
    }
)

# Prediction button
if st.button("Predict Performance"):
    prediction = model.predict(input_data)[0]  # 0 = Fail, 1 = Pass
    result = "  ✔ PASS" if prediction == 1 else "  ✖ FAIL"
    st.success(f"Predicted Outcome: {result}")  # Display result


