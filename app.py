import streamlit as st
import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Spammer Detection System",
    page_icon="🚨",
    layout="wide"
)

# ---------------------------------
# Load Model
# ---------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/spammer_model.pkl")

model = load_model()

# ---------------------------------
# Sidebar
# ---------------------------------
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Data Preview", "Prediction", "Model Performance"]
)

# ---------------------------------
# Home Page
# ---------------------------------
if page == "Home":
    st.title("🚨 Predict Potential Spammers on Freelance Platforms")

    st.markdown("""
    ### 📌 Problem Overview
    Attackers are misusing freelance job platforms to distribute malware
    through fake job offers and malicious attachments.

    This application uses **Machine Learning (XGBoost)** to predict
    **potential spammers** based on anonymized user behavior.

    ### 🎯 Objective
    - Detect high-risk users early
    - Reduce financial and reputational damage
    - Improve platform trust
    """)

# ---------------------------------
# Data Preview Page
# ---------------------------------
elif page == "Data Preview":
    st.title("📊 Dataset Preview")

    df = pd.read_csv("data/train.csv")
    st.write("Shape of dataset:", df.shape)

    if st.checkbox("Show raw data"):
        st.dataframe(df.head(50))

    st.subheader("Target Distribution")
    st.bar_chart(df["label"].value_counts())

# ---------------------------------
# Prediction Page
# ---------------------------------
elif page == "Prediction":
    st.title("🧪 Spammer Prediction")

    st.markdown("### Enter User Feature Values")

    input_data = {}

    # Dynamically create inputs for X1–X51
    for i in range(1, 52):
        input_data[f"X{i}"] = st.number_input(
            f"X{i}",
            value=0.0,
            step=0.1
        )

    input_df = pd.DataFrame([input_data])

    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("🔎 Prediction Result")

        if probability < 0.3:
            st.success(f"Low Risk User ✅ (Probability: {probability:.2f})")
        elif probability < 0.6:
            st.warning(f"Medium Risk User ⚠️ (Probability: {probability:.2f})")
        else:
            st.error(f"High Risk Spammer 🚨 (Probability: {probability:.2f})")

# ---------------------------------
# Model Performance Page
# ---------------------------------
elif page == "Model Performance":
    st.title("📈 Model Performance Overview")

    st.markdown("""
    ### Model Used
    - **XGBoost Classifier**
    - Handles anonymized and non-linear behavioral features

    ### Evaluation Metrics
    - Precision
    - Recall
    - F1-Score
    - ROC-AUC

    ### Key Strength
    High recall ensures most spammers are detected.
    """)

    st.info("Model evaluation performed during training phase.")

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("---")
st.markdown("**Capstone Project | Spammer Detection System**")