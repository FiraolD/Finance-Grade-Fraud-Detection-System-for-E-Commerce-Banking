from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
import uvicorn
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="E-Commerce Fraud Detection API",
    description="API for detecting fraud in E-Commerce transactions",
    version="1.0.0"
)

# Load the E-commerce model pipeline and info
try:
    pipeline = joblib.load("models/XGBoost_ecommerce_pipeline.pkl")
    model_info = joblib.load("models/ecommerce_model_info.pkl")
    print("✅ E-commerce model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading E-commerce model: {e}")
    pipeline = None
    model_info = None

# Pydantic model for request validation
class TransactionData(BaseModel):
    user_id: int
    signup_time: str
    purchase_time: str
    purchase_value: float
    device_id: str
    source: str
    browser: str
    sex: str
    age: int
    ip_address: str
    transaction_country: str
    Amount: float = 0.0  # Optional banking amount
    Time: float = 0.0    # Optional banking time

# Pydantic model for response
class PredictionResponse(BaseModel):
    fraud_probability: float
    fraud_label: int
    confidence: float

def preprocess_transaction(transaction: TransactionData, encoders: Dict, feature_columns: list):
    """
    Preprocess a single transaction for prediction
    """
    # Convert datetime strings to datetime objects
    signup_time = pd.to_datetime(transaction.signup_time)
    purchase_time = pd.to_datetime(transaction.purchase_time)
    
    # Extract time-based features
    signup_hour = signup_time.hour
    signup_day = signup_time.day
    signup_month = signup_time.month
    signup_weekday = signup_time.weekday()
    
    purchase_hour = purchase_time.hour
    purchase_day = purchase_time.day
    purchase_month = purchase_time.month
    purchase_weekday = purchase_time.weekday()
    
    # Time difference between signup and purchase
    time_to_purchase = (purchase_time - signup_time).total_seconds()
    
    # Encode categorical variables
    source_encoded = encoders['source'].transform([transaction.source])[0]
    browser_encoded = encoders['browser'].transform([transaction.browser])[0]
    sex_encoded = encoders['sex'].transform([transaction.sex])[0]
    
    # Create device_id features
    device_id_length = len(transaction.device_id)
    device_id_unique_chars = len(set(transaction.device_id))
    
    # IP address features
    ip_address_length = len(str(transaction.ip_address))
    
    # Transaction country features
    country_encoded = encoders['country'].transform([transaction.transaction_country])[0]
    
    # Create feature vector
    features = {
        'user_id': transaction.user_id,
        'purchase_value': transaction.purchase_value,
        'age': transaction.age,
        'signup_hour': signup_hour,
        'signup_day': signup_day,
        'signup_month': signup_month,
        'signup_weekday': signup_weekday,
        'purchase_hour': purchase_hour,
        'purchase_day': purchase_day,
        'purchase_month': purchase_month,
        'purchase_weekday': purchase_weekday,
        'time_to_purchase': time_to_purchase,
        'source_encoded': source_encoded,
        'browser_encoded': browser_encoded,
        'sex_encoded': sex_encoded,
        'device_id_length': device_id_length,
        'device_id_unique_chars': device_id_unique_chars,
        'ip_address_length': ip_address_length,
        'country_encoded': country_encoded
    }
    
    # Add banking features if provided
    if transaction.Amount > 0:
        features['Amount'] = transaction.Amount
    if transaction.Time > 0:
        features['Time'] = transaction.Time
    
    # Create DataFrame with correct column order
    df = pd.DataFrame([features])
    
    # Ensure all required features are present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0  # Default value for missing features
    
    # Reorder columns to match training data
    X = df[feature_columns]
    
    return X

@app.get("/")
async def root():
    return {"message": "E-Commerce Fraud Detection API is running! Use /predict endpoint for predictions."}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model_loaded": pipeline is not None,
        "model_type": "XGBoost E-commerce",
        "features": len(model_info['feature_columns']) if model_info else 0
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionData):
    """
    Predict fraud probability for an E-commerce transaction
    """
    if pipeline is None or model_info is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Preprocess the transaction
        X = preprocess_transaction(transaction, model_info['encoders'], model_info['feature_columns'])
        
        # Make prediction
        fraud_prob = pipeline.predict_proba(X)[0, 1]
        fraud_label = 1 if fraud_prob > 0.5 else 0
        
        # Calculate confidence (distance from decision boundary)
        confidence = abs(fraud_prob - 0.5) * 2
        
        return PredictionResponse(
            fraud_probability=fraud_prob,
            fraud_label=fraud_label,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
async def model_info_endpoint():
    """
    Get information about the loaded model
    """
    if model_info is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        return {
            "model_type": "XGBoost E-commerce",
            "features": model_info['feature_columns'],
            "n_features": len(model_info['feature_columns']),
            "best_model": model_info['best_model'],
            "best_auc": model_info['best_auc'],
            "encoders_available": list(model_info['encoders'].keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
