import pandas as pd

# Load dataset
print("Loading dataset...")
df = pd.read_csv("data/creditcard.csv")

print("\n--- Dataset Info ---")
print(f"Total Rows: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}")

print("\n--- Class Distribution ---")
# 0 = Normal, 1 = Fraud
counts = df['Class'].value_counts()
print(f"Legitimate Transactions (0): {counts[0]}")
print(f"Fraudulent Transactions (1): {counts[1]}")

percentage = df['Class'].value_counts(normalize=True) * 100
print(f"\nFraud Percentage: {percentage[1]:.4f}%")
