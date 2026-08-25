from pathlib import Path

import pandas as pd


FEATURES = [f"V{i}" for i in range(1, 29)] + ["scaled_amount", "scaled_time"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
COMPRESSED_DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv.gz"


def _scaling_stats():
	path = DATA_PATH if DATA_PATH.exists() else COMPRESSED_DATA_PATH
	data = pd.read_csv(path, usecols=["Amount", "Time"], compression="infer")
	return (
		float(data["Amount"].mean()),
		float(data["Amount"].std(ddof=0)) or 1.0,
		float(data["Time"].mean()),
		float(data["Time"].std(ddof=0)) or 1.0,
	)


def prepare_transaction(transaction, expected_features=None):
	"""Convert raw or model-shaped transaction data into model input."""
	if not isinstance(transaction, dict):
		raise TypeError("transaction must be an object")

	features = list(expected_features) if expected_features is not None else FEATURES
	amount_mean, amount_std, time_mean, time_std = _scaling_stats()
	prepared = {}
	for feature in features:
		if feature == "scaled_amount":
			if "scaled_amount" in transaction:
				prepared[feature] = float(transaction[feature])
			elif "Amount" in transaction:
				prepared[feature] = (float(transaction["Amount"]) - amount_mean) / amount_std
			else:
				raise ValueError("transaction requires Amount or scaled_amount")
		elif feature == "scaled_time":
			if "scaled_time" in transaction:
				prepared[feature] = float(transaction[feature])
			elif "Time" in transaction:
				prepared[feature] = (float(transaction["Time"]) - time_mean) / time_std
			else:
				raise ValueError("transaction requires Time or scaled_time")
		elif feature not in transaction:
			prepared[feature] = 0.0
		else:
			prepared[feature] = float(transaction[feature])

	return pd.DataFrame([prepared], columns=features)
