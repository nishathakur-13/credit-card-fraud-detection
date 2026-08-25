import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from preprocess import load_and_preprocess

# 1. Load preprocessed dataset
X_train, X_test, y_train, y_test = load_and_preprocess()

# 2. Train Random Forest Model
print("\nTraining Random Forest model (this may take 1-2 minutes)...")
model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    n_jobs=-1
)
model.fit(X_train, y_train)

# 3. Predict on Test Set
print("\nEvaluating model performance...")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# 4. Print Evaluation Metrics
print("\n" + "="*40)
print("          CONFUSION MATRIX")
print("="*40)
print(confusion_matrix(y_test, y_pred))

print("\n" + "="*40)
print("       CLASSIFICATION REPORT")
print("="*40)
print(classification_report(y_test, y_pred, digits=4))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# 5. Save Model artifact
os.makedirs("models", exist_ok=True)
model_path = "models/fraud_detector.pkl"
joblib.dump(model, model_path)
print(f"\nTrained model successfully saved to '{model_path}'!")