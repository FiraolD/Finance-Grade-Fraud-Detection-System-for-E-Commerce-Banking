# src/models/Train.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =====================
# 1. Load Data
# =====================
data_path = "Data/creditcard.csv"   # replace with your path
df = pd.read_csv(data_path)
print(f"✅ Data loaded: {df.shape}")

# =====================
# 2. Feature Engineering
# =====================
# Example feature: signup-to-purchase time (if available in fraud_data.csv)
if "signup_time" in df.columns and "purchase_time" in df.columns:
    df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce")
    df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce")
    df["signup_to_purchase_secs"] = (df["purchase_time"] - df["signup_time"]).dt.total_seconds()
    df["signup_to_purchase_secs"].fillna(0, inplace=True)

# Frequency encoding for device_id, browser, source (if exist)
for col in ["device_id", "browser", "source", "ip_address"]:
    if col in df.columns:
        freq_map = df[col].value_counts().to_dict()
        df[f"{col}_freq"] = df[col].map(freq_map)

# =====================
# 3. Features/Target
# =====================
y = df["Class"] if "Class" in df.columns else df["class"]
X = df.drop(columns=["Class"], errors="ignore").drop(columns=["class"], errors="ignore")

# Fill NaNs
X = X.fillna(0)

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# =====================
# 4. Balance Classes (SMOTE)
# =====================
print("⚖️ Applying SMOTE to balance fraud/non-fraud...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"Resampled training set size: {X_train_res.shape}")

# =====================
# 5. Train Models
# =====================
models = {
    "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=200),
    "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    "XGBoost": XGBClassifier(scale_pos_weight=(len(y_train_res) - sum(y_train_res)) / sum(y_train_res),
                             eval_metric="logloss", use_label_encoder=False),
    "LightGBM": LGBMClassifier(class_weight="balanced", random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    roc_auc = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)

    print(f"--- {name} ---")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

    results[name] = {"roc_auc": roc_auc, "pr_auc": pr_auc}

    # Save model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/{name}.pkl")
    print(f"✅ Saved {name} model at {model_dir}/{name}.pkl")

# =====================
# 6. Confusion Matrix for Best Model
# =====================
best_model_name = max(results, key=lambda x: results[x]["pr_auc"])
print(f"\n🏆 Best model by PR-AUC: {best_model_name}")
best_model = joblib.load(f"models/{best_model_name}.pkl")

y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Genuine", "Fraud"], yticklabels=["Genuine", "Fraud"])
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
