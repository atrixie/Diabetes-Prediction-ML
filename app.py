import streamlit as st
import pickle
import pandas as pd
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #141e30, #243b55);
}

h1 {
    color: white !important;
    text-align: center;
}

h2, h3, p, label {
    color: white !important;
}

[data-testid="stSidebar"] {
    background-color: #1e293b;
}

div.stButton > button {
    background-color: #10b981;
    color: white;
    border-radius: 10px;
    font-size: 18px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #059669;
}

[data-testid="stNumberInput"] {
    background-color: rgba(255,255,255,0.15);
    border-radius: 10px;
    padding: 5px;
}

</style>
""", unsafe_allow_html=True)

model = pickle.load(open('diabetes_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

st.title("🩺 Diabetes Prediction using Machine Learning")
st.markdown("""
### About This Project

This application uses a Support Vector Machine (SVM) Machine Learning model to predict whether a patient is likely to have diabetes based on health indicators.

**Features Used:**
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
""")

st.markdown("""
### Predict Diabetes Risk using Health Parameters

Enter patient information below and click **Predict**.
""")

st.markdown("""
### Developed by Nikijon Kakati
**InternPe Internship Project**

This application predicts whether a patient is likely to have diabetes based on health parameters.
""")
st.info("Model Accuracy: 77.27%")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies")
    glucose = st.number_input("Glucose")
    blood_pressure = st.number_input("Blood Pressure")
    skin_thickness = st.number_input("Skin Thickness")

with col2:
    insulin = st.number_input("Insulin")
    bmi = st.number_input("BMI")
    dpf = st.number_input("Diabetes Pedigree Function")
    age = st.number_input("Age")

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

        st.markdown("---")
st.markdown(
    "Developed by **Nikijon Kakati** | InternPe Machine Learning Project"
)