import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

model = pickle.load(open('model.pkl', 'rb'))

st.set_page_config(page_title="Diabetes Predictor", layout="wide")

st.markdown("<h1 style='text-align: center;'>🩺 Diabetes Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

demo_data = {
    "None": [0, 0, 0, 0, 0, 0.0, 0.0, 1],
    "Age 25 (Healthy)": [1, 85, 70, 20, 80, 24.5, 0.3, 25],
    "Age 30 (Low Risk)": [2, 95, 72, 22, 85, 26.0, 0.4, 30],
    "Age 35 (Moderate)": [3, 110, 75, 25, 100, 28.5, 0.5, 35],
    "Age 40 (Borderline)": [3, 130, 80, 25, 120, 30.0, 0.6, 40],
    "Age 50 (High Risk)": [5, 150, 85, 30, 180, 33.0, 0.9, 50],
    "Age 60 (Very High)": [6, 170, 90, 35, 220, 36.5, 1.1, 60],
    "Age 70 (Critical)": [7, 180, 95, 38, 250, 38.0, 1.3, 70],
    "Age 80 (Severe Case)": [8, 190, 100, 40, 300, 40.0, 1.5, 80]
}

st.subheader("🎯 Demo Profiles")
selected_demo = st.selectbox("Choose a demo case (optional):", list(demo_data.keys()))

values = demo_data[selected_demo]

st.subheader("📋 Patient Details")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, value=int(values[0]))
    glucose = st.number_input("Glucose Level", 0, 200, value=int(values[1]))
    bp = st.number_input("Blood Pressure", 0, 150, value=int(values[2]))
    skin = st.number_input("Skin Thickness", 0, 100, value=int(values[3]))

with col2:
    insulin = st.number_input("Insulin", 0, 900, value=int(values[4]))
    bmi = st.number_input("BMI", 0.0, 70.0, value=float(values[5]))
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, value=float(values[6]))
    age = st.number_input("Age", 1, 120, value=int(values[7]))

st.markdown("---")

if st.button("🔍 Predict Diabetes"):

    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")

    col3, col4 = st.columns(2)

    with col3:
        if prediction == 1:
            st.error("⚠️ High Risk of Diabetes")
        else:
            st.success("✅ Low Risk of Diabetes")

        st.metric(label="Probability", value=f"{probability*100:.2f}%")

    with col4:
        fig, ax = plt.subplots()
        ax.bar(["No Diabetes", "Diabetes"], [1-probability, probability])
        ax.set_ylabel("Probability")
        st.pyplot(fig)

    st.markdown("---")

    st.subheader("📄 Patient Report")

    report = pd.DataFrame({
        "Feature": ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
                    "Insulin", "BMI", "DPF", "Age"],
        "Value": [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
    })

    st.table(report)

    csv = report.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Report",
        data=csv,
        file_name='diabetes_report.csv',
        mime='text/csv',
    )