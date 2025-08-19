# check_pipeline.py
import joblib

# Load the pipeline
pipeline = joblib.load("models/XGBoost_pipeline.pkl")

# Print all step names
print("🔧 Pipeline Steps:")
for name in pipeline.named_steps:
    print(f" - {name}")

# Optional: print the whole pipeline
print("\n📊 Full Pipeline:")
print(pipeline)