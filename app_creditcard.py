# app.py
import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Fraud Detection Dashboard", layout="centered")
st.title("🛡️ AdeyGuard Fraud Detection Dashboard")
st.write("Upload a transaction to predict fraud risk.")

# Load the full pipeline
try:
    pipeline = joblib.load("models/XGBoost_pipeline.pkl")
    st.success("✅ Model pipeline loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Access the model using correct name
try:
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['scaler']
except KeyError as e:
    st.error(f"❌ Pipeline missing expected step: {e}")
    st.write("Available steps:", list(pipeline.named_steps.keys()))
    st.stop()

# Show pipeline steps in sidebar for debugging
st.sidebar.header("🔧 Pipeline Info")
for name in pipeline.named_steps:
    st.sidebar.write(f"- `{name}`")

# Upload data
uploaded = st.file_uploader("Upload transaction CSV", type="csv")
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        st.write("📄 Uploaded Data Sample:")
        st.dataframe(df.head())

        # Ensure correct columns are present
        expected_features = preprocessor.feature_names_in_
        missing_cols = [col for col in expected_features if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns: {missing_cols}")
            st.stop()

        # Reorder columns to match training
        X = df[expected_features]

        # Predict probabilities
        proba = pipeline.predict_proba(X)[0, 1]
        pred = "🚨 Fraud" if proba > 0.5 else "✅ Legitimate"

        st.write(f"### Prediction: {pred}")
        st.write(f"**Fraud Probability:** {proba:.2%}")

        # SHAP Explanation
        st.subheader("🔍 Why This Was Flagged")

        # Use the trained XGBoost model directly with SHAP
        explainer = shap.TreeExplainer(model)
        X_processed = preprocessor.transform(X)
        shap_values = explainer.shap_values(X_processed)

        # Waterfall plot for first row
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X.iloc[0]
            ),
            max_display=8
        )
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")