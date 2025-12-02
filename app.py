import streamlit as st
import joblib
import numpy as np
import pandas as pd
model = joblib.load('logistic_model.pkl')
st.title('Logistic Regression Model - Prediction of outcome')
st.write("Enter the details below to predict outcome")
Pregnancies=st.number_input("Pregnancies",min_value=0.00,max_value=13.5,value=8.5)
Glucose=st.number_input("Glucose",min_value=38.00,max_value=199.00,value=89.00)
BloodPressure=st.number_input("BloodPressure",min_value=35.00,max_value=107.00,value=72.00)
SkinThickness=st.number_input("SkinThickness",min_value=0.00,max_value=80.00,value=48.00)
Insulin=st.number_input("Insulin",min_value=0.00,max_value=318.00,value=180.00)
BMI=st.number_input("BMI",min_value=14.00,max_value=50.00,value=30.00)
DiabetesPedigreeFunction=st.number_input("DiabetesPedigreeFunction",min_value=0.078000,max_value=1.2,value=0.95)
Age=st.number_input("Age",min_value=21.00,max_value=66.00,value=32.00)
input_data= np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
if st.button("Predict"):
    prediction = model.predict(input_data)
    result = "1" if prediction[0] == 1 else "0"
    st.write(f"Prediction: {result}")






