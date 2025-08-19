import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # pipeline with SMOTE

# Gradient Boosting Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def main():
    # =====================
    # 1. Load dataset
    # =====================
    df = pd.read_csv("Data/creditcard.csv")
    print("✅ Data loaded:", df.shape)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # =====================
    # 2. Train-Test Split
    # =====================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # =====================
    # 3. Models
    # =====================
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1, eval_metric="logloss"
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
    }

    # Ensure models/ dir exists
    os.makedirs("models", exist_ok=True)

    # =====================
    # 4. Train & Evaluate
    # =====================
    best_model_name, best_auc = None, 0.0

    for name, model in models.items():
        print(f"\n🔹 Training {name}...")

        # Build pipeline: scaling + SMOTE + model
        pipeline = ImbPipeline(steps=[
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, digits=4))
        print("ROC-AUC:", auc)

        # Save pipeline
        save_path = f"models/{name}_pipeline.pkl"
        joblib.dump(pipeline, save_path)
        print(f"💾 Saved {name} pipeline → {save_path}")

        if auc > best_auc:
            best_model_name, best_auc = name, auc

    print(f"\n🏆 Best model: {best_model_name} with ROC-AUC = {best_auc:.4f}")

if __name__ == "__main__":
    main()
