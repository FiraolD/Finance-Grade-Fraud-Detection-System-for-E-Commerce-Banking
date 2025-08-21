import streamlit as st
import requests
import time
from datetime import datetime
import json

# ========================
# CONFIG
# ========================
API_URL = "http://127.0.0.1:8000/predict"  # FastAPI endpoint
HEALTH_URL = "http://127.0.0.1:8000/health"  # Health check endpoint

# Enhanced page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .fraud-alert {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .safe-alert {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🔎 Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Interactive fraud detection for **E-Commerce & Banking transactions**")

# Sidebar with API status
with st.sidebar:
    st.header("🔧 System Status")
    
    # Check API health
    try:
        health_response = requests.get(HEALTH_URL, timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            st.success("✅ API Connected")
            st.info(f"Model: {health_data.get('model_type', 'Unknown')}")
            st.info(f"Features: {health_data.get('features', 0)}")
        else:
            st.error("❌ API Error")
    except Exception as e:
        st.error("❌ API Unreachable")
        st.error(f"Error: {str(e)}")
    
    st.divider()
    st.markdown("### 📊 Model Performance")
    st.info("ROC-AUC: 0.7693")
    st.info("Accuracy: 95.67%")
    
    st.divider()
    st.markdown("### 📝 Instructions")
    st.markdown("""
    1. Fill in transaction details OR upload JSON
    2. Click 'Check Fraud Risk'
    3. Review prediction results
    """)


# ========================
# INPUT FORM
# ========================
st.markdown("### 📝 Enter Transaction Details")

# Create tabs for better organization
tab1, tab2 = st.tabs(["🛒 E-Commerce Data", "🏦 Banking Data (Optional)"])

with st.form("fraud_form"):
    with tab1:
        st.markdown("#### Required E-Commerce Information")
        col1, col2 = st.columns(2)

        with col1:
            user_id = st.number_input("👤 User ID", value=247547, help="Unique identifier for the user")
            signup_time = st.text_input("📅 Signup Time", "2015-06-28 03:00:34", help="Format: YYYY-MM-DD HH:MM:SS")
            purchase_time = st.text_input("🛍️ Purchase Time", "2015-08-09 03:57:29", help="Format: YYYY-MM-DD HH:MM:SS")
            purchase_value = st.number_input("💰 Purchase Value ($)", value=47.0, min_value=0.0)
            device_id = st.text_input("📱 Device ID", "KIXYSVCHIPQBR")
        
        with col2:
            source = st.selectbox("🔗 Source", ["SEO", "Ads", "Direct"])
            browser = st.selectbox("🌐 Browser", ["Chrome", "Safari", "Firefox", "IE"])
            sex = st.selectbox("👥 Sex", ["M", "F"])
            age = st.number_input("🎂 Age", value=30, min_value=1, max_value=120)
            ip_address = st.text_input("🌍 IP Address", "43.173.1.96")
            transaction_country = st.text_input("🏳️ Transaction Country", "Australia")
    
    with tab2:
        st.markdown("#### Optional Banking Information")
        st.info("💡 These fields are optional and can be left default.")

        col3, col4 = st.columns(2)
        with col3:
            amount = st.number_input("🏦 Banking Amount", value=149.62, min_value=0.0)
        with col4:
            time_val = st.number_input("⏰ Banking Time", value=25432, min_value=0)

    # Submit button inside the form
    submitted = st.form_submit_button("🔍 Check Fraud Risk", use_container_width=True, type="primary")

# ========================
# CALL FASTAPI (Form Data)
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


# ========================
# JSON Upload Option
# ========================
st.markdown("---")
st.markdown("### 📂 Or Upload a JSON File")

uploaded_file = st.file_uploader("Upload JSON transaction file", type=["json"])

if uploaded_file is not None:
    try:
        transaction_data = json.load(uploaded_file)
        st.json(transaction_data)

        response = requests.post(API_URL, json=transaction_data)
        if response.status_code == 200:
            result = response.json()
            fraud_prob = result["fraud_probability"]
            fraud_label = result["fraud_label"]

            st.subheader("📊 Prediction Results (from JSON)")
            st.metric("Fraud Probability", f"{fraud_prob * 100:.2f}%")
            st.metric("Fraud Label", "🚨 Fraud" if fraud_label == 1 else "✅ Genuine")

            if fraud_label == 1:
                st.error("⚠️ High risk: This transaction is likely FRAUDULENT.")
            else:
                st.success("✅ Safe: This transaction is likely genuine.")
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Invalid JSON file. Error: {str(e)}")
