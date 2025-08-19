  💳 Dual-Channel Fraud Detection System for E-Commerce & Banking  

A robust   fraud detection system   designed to identify fraudulent transactions in both   e-commerce   and   banking channels  .  
This project leverages advanced   machine learning (XGBoost, LightGBM, Random Forest)   combined with   IP-to-country enrichment   to improve accuracy, reliability, and reduce risk for financial institutions.  


   📌 Business Understanding  
Fraudulent transactions cause   billions of dollars in losses annually   for banks and e-commerce platforms.  
The goal of this project is to:  

-   Detect fraud early   → minimize financial loss.  
-   Reduce false positives   → avoid unnecessary customer friction.  
-   Provide explainability   → support compliance & risk management teams.  

This aligns with what financial institutions value most:   reliability, precision, and risk reduction  .  


   📊 Dataset  

We used multiple datasets to enrich fraud detection features:  

1.   `creditcard.csv`   → European credit card transactions (imbalanced fraud dataset).  
2.   `Fraud_Data.csv`   → E-commerce transactions with user, device, and IP info.  
3.   `IpAddress_to_Country.csv`   → IP-to-country mapping for geographic fraud detection.  

After preprocessing & merging, we created a   dual-channel fraud dataset  .  


   ⚙️ Project Pipeline  

    🔹 1. Data Engineering  
- Handle missing & corrupted values.  
- Convert IP → Country features (`ip_to_int`, `merge_asof`).  
- Feature engineering (time deltas, frequency encoding).  
- Train-test split &   imbalanced data handling   (SMOTE).  

    🔹 2. Modeling  
Trained multiple ML models:  

| Model               | Precision | Recall | ROC-AUC |
|----------------------|-----------|--------|---------|
| Logistic Regression | 0.0578    | 0.9184 | 0.9708  |
| Random Forest       | 0.8710    | 0.8265 | 0.9754  |
| XGBoost             | 0.7167    | 0.8776 | 0.9797  |
| LightGBM            | 0.7757    | 0.8469 | 0.9593  |

✅   Best models:    
-   Random Forest   → Best balance of precision & recall.  
-   XGBoost   → Best overall ROC-AUC (aggressive fraud catching).  

    🔹 3. Explainability  
-   SHAP values   to explain model predictions.  
- Helps risk teams understand  why  a transaction was flagged.  

    🔹 4. Deployment (Planned)  
- Save models using `joblib`.  
- Expose fraud detection API via   FastAPI  .  
- Monitor drift & retrain periodically.  


   🛠️ Tech Stack  

-   Python 3.10+    
-   Pandas, NumPy, Scikit-learn   → Data prep & ML  
-   XGBoost, LightGBM   → Gradient boosting models  
-   SHAP   → Model explainability  
-   Matplotlib, Seaborn   → Visualizations  
-   FastAPI (planned)   → API deployment  


   📂 Repository Structure  

```bash
Dual-Channel-Fraud-Detection-System-for-E-Commerce-Banking/
│── Data/                    Raw & processed datasets (gitignored)
│── src/
│   ├── data/                Data loading & preprocessing scripts
│   │   └── load&merge.py
│   ├── models/              Model training & evaluation
│   │   └── Train.py
│   │   └── shap_explainer.py
│   └── utils/               Helper functions
│── notebooks/               Jupyter notebooks (exploration)
│── reports/                 Interim/final reports
│── models/                  Saved ML models (.joblib) (gitignored)
│── README.md                Project documentation
│── requirements.txt         Dependencies
│── .gitignore               Exclude data & large files
