import joblib
import pandas as pd
import numpy as np

# 1. Load the trained model
print("Loading trained model...")
model = joblib.load("models/fraud_detector.pkl")

# 2. Get feature names used during training
feature_names = model.feature_names_in_

# 3. Create dummy sample data (simulating 1 new transaction)
# Replacing raw Amount and Time with scaled values as during preprocessing
sample_data = {feat: [0.0] for feat in feature_names}
sample_df = pd.DataFrame(sample_data)

# 4. Make Prediction
prediction = model.predict(sample_df)[0]
probability = model.predict_proba(sample_df)[0][1]

# 5. Output Result
print("\n" + "="*40)
print("       INFERENCE TEST RESULT")
print("="*40)
print(f"Fraud Probability: {probability * 100:.2f}%")
if prediction == 1:
    print("STATUS: ⚠️ ALERT - Potential Fraudulent Transaction Detected!")
else:
    print("STATUS: ✅ CLEAR - Legitimate Transaction")