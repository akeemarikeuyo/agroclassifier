import streamlit as st
import pandas as pd
import joblib

LABEL_INFO = {
    0: ("✅", "success", "Low concern"),
    1: ("⚠️", "warning", "Moderate concern"),
    2: ("❌", "error", "High concern"),
}

try:
    model = joblib.load('agrochemical_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please add agrochemical_model.pkl.")
    st.stop()

st.set_page_config(page_title="Agrotoxicity Predictor", layout="centered")
st.title("Agrochemical Toxicity Classifier")
st.markdown("Predict the **toxicity level** (0=Low, 1=Moderate, 2=High) from chemical properties.")

col1, col2 = st.columns(2)
with col1:
    year = st.number_input("Year of introduction", min_value=1800, max_value=2025, value=2000)
    herbicide = st.selectbox("Herbicide", [0, 1], format_func=lambda x: "Yes" if x else "No")
    fungicide = st.selectbox("Fungicide", [0, 1], format_func=lambda x: "Yes" if x else "No")
    insecticide = st.selectbox("Insecticide", [0, 1], format_func=lambda x: "Yes" if x else "No")
with col2:
    other_agro = st.selectbox("Other agrochemical", [0, 1], format_func=lambda x: "Yes" if x else "No")
    source = st.selectbox("Source", ["ECOTOX", "PPDB", "BPDB"])
    toxicity_type = st.selectbox("Toxicity type", ["Contact", "Oral", "Other"])

input_data = pd.DataFrame([{
    'year': year, 'herbicide': herbicide, 'fungicide': fungicide,
    'insecticide': insecticide, 'other_agrochemical': other_agro,
    'source': source, 'toxicity_type': toxicity_type
}])

if st.button("🔍 Predict", type="primary"):
    prediction = model.predict(input_data)[0]
    icon, level, label = LABEL_INFO[prediction]
    getattr(st, level)(f"{icon} Predicted label: **{prediction}** ({label})")

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(input_data)[0]
        st.write("**Class probabilities:**")
        prob_df = pd.DataFrame({
            'Toxicity Level': ['Low (0)', 'Moderate (1)', 'High (2)'],
            'Probability': proba
        })
        st.bar_chart(prob_df.set_index('Toxicity Level'))

st.markdown("---")
st.markdown("<center>ML Project done by Akeem Arikeuyo</center>", unsafe_allow_html=True)
