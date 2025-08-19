import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

def explain_model(model_path="models/XGBoost_pipeline.pkl", sample_size=500):
    """
    Explain fraud detection model predictions with SHAP.
    
    Args:
        model_path (str): path to the saved pipeline (.pkl).
        sample_size (int): number of samples for SHAP explanation.
    """
    # =====================
    # 1. Load Model & Data
    # =====================
    print(f"📂 Loading model from {model_path}")
    pipeline = joblib.load(model_path)

    # Load creditcard dataset (same one used for training)
    df = pd.read_csv("Data/creditcard.csv")
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Sample data for faster SHAP
    X_sample = X.sample(n=min(sample_size, len(X)), random_state=42)

    # =====================
    # 2. Extract Model from Pipeline
    # =====================
    model = pipeline.named_steps['model']
    print(f"✅ Model inside pipeline: {model.__class__.__name__}")

    # =====================
    # 3. Initialize SHAP Explainer
    # =====================
    explainer = shap.Explainer(model, pipeline.named_steps['scaler'].transform(X_sample))
    shap_values = explainer(pipeline.named_steps['scaler'].transform(X_sample))

    # =====================
    # 4. Global Feature Importance
    # =====================
    print("📊 Generating global feature importance...")
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    os.makedirs("reports/figures", exist_ok=True)
    plt.savefig("reports/figures/shap_summary.png")
    plt.close()

    # =====================
    # 5. Local Explanation for a Fraud Case
    # =====================
    fraud_indices = df[df["Class"] == 1].index
    if len(fraud_indices) > 0:
        idx = fraud_indices[0]
        X_fraud = X.loc[[idx]]
        print(f"🔍 Explaining fraud case at index {idx}...")

        shap_values_single = explainer(pipeline.named_steps['scaler'].transform(X_fraud))
        shap.plots.waterfall(shap_values_single[0], show=False)
        plt.title("Local Explanation for Fraud Transaction")
        plt.tight_layout()
        plt.savefig("reports/figures/shap_fraud_case.png")
        plt.close()

    print("✅ SHAP explanations saved in reports/figures/")

if __name__ == "__main__":
    explain_model()
