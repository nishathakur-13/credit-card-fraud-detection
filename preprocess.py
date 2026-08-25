import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_and_preprocess():
    print("Loading dataset for preprocessing...")
    df = pd.read_csv("data/creditcard.csv")

    # Scale raw 'Amount' and 'Time' features
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
    df['scaled_time'] = scaler.fit_transform(df[['Time']])
    
    # Drop unscaled columns
    df = df.drop(['Amount', 'Time'], axis=1)

    # Separate features (X) and target label (y)
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Original Training Fraud Count: {sum(y_train == 1)}")

    # Apply SMOTE to rebalance training data
    print("Applying SMOTE oversampling to training set...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"Resampled Training Fraud Count: {sum(y_train_resampled == 1)}")

    return X_train_resampled, X_test, y_train_resampled, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess()
    print("\nPreprocessing complete!")
    print(f"Resampled Training Features Shape: {X_train.shape}")
    print(f"Test Features Shape: {X_test.shape}")