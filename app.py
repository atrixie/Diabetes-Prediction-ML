import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open('diabetes_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

st.title("🩺 Diabetes Prediction System")

pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=1)

if st.button("Predict"):

    input_df = pd.DataFrame(
        [[pregnancies,
          glucose,
          blood_pressure,
          skin_thickness,
          insulin,
          bmi,
          dpf,
          age]],
        columns=[
            'Pregnancies',
            'Glucose',
            'BloodPressure',
            'SkinThickness',
            'Insulin',
            'BMI',
            'DiabetesPedigreeFunction',
            'Age'
        ]
    )

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    if prediction[0] == 0:
        st.success("✅ Person is NOT Diabetic")
    else:
        st.error("⚠️ Person is Diabetic")