import streamlit as st
import requests

# ========================
# CONFIG
# ========================
API_URL = "http://127.0.0.1:8000/predict"  # FastAPI endpoint

st.set_page_config(page_title="Fraud Detection Dashboard", layout="centered")

st.title("🔎 Fraud Detection Dashboard")
st.markdown("Interactive fraud detection for **E-Commerce & Banking transactions**.")

# ========================
# INPUT FORM
# ========================
st.subheader("Enter Transaction Details")

with st.form("fraud_form"):
    col1, col2 = st.columns(2)

    with col1:
        user_id = st.number_input("User ID", value=247547)
        signup_time = st.text_input("Signup Time", "2015-06-28 03:00:34")
        purchase_time = st.text_input("Purchase Time", "2015-08-09 03:57:29")
        purchase_value = st.number_input("Purchase Value", value=47.0)
        device_id = st.text_input("Device ID", "KIXYSVCHIPQBR")
        source = st.selectbox("Source", ["SEO", "Ads", "Direct"])
        browser = st.selectbox("Browser", ["Chrome", "Safari", "Firefox", "IE"])
    
    with col2:
        sex = st.selectbox("Sex", ["M", "F"])
        age = st.number_input("Age", value=30)
        ip_address = st.text_input("IP Address", "43.173.1.96")
        transaction_country = st.text_input("Transaction Country", "Australia")
        amount = st.number_input("Banking Amount (optional)", value=149.62)
        time_val = st.number_input("Banking Time (optional)", value=25432)
    
    submitted = st.form_submit_button("🔍 Check Fraud Risk")

# ========================
# CALL FASTAPI
# ========================
if submitted:
    payload = {
        "user_id": user_id,
        "signup_time": signup_time,
        "purchase_time": purchase_time,
        "purchase_value": purchase_value,
        "device_id": device_id,
        "source": source,
        "browser": browser,
        "sex": sex,
        "age": age,
        "ip_address": ip_address,
        "transaction_country": transaction_country,
        "Amount": amount,
        "Time": time_val
    }

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            fraud_prob = result["fraud_probability"]
            fraud_label = result["fraud_label"]

            st.subheader("📊 Prediction Results")
            st.metric("Fraud Probability", f"{fraud_prob * 100:.2f}%")
            st.metric("Fraud Label", "🚨 Fraud" if fraud_label == 1 else "✅ Genuine")

            if fraud_label == 1:
                st.error("⚠️ High risk: This transaction is likely FRAUDULENT.")
            else:
                st.success("✅ Safe: This transaction is likely genuine.")
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Could not connect to API. Error: {str(e)}")
