import pandas as pd

from backend.services.risk_engine import RiskEngine
from backend.services.model_service import ModelService
from backend.utils.explainability import ExplainabilityEngine
from backend.utils.feature_engineering import prepare_transaction


class PredictionService:

    @staticmethod
    def predict(transaction_data):
        """
        Predict whether a transaction is fraudulent
        and generate risk/explanation information.
        """

        # Load trained AI model
        model = ModelService.load_model()

        if not isinstance(transaction_data, dict):
            raise TypeError("transaction must be an object")

        transaction_dict = transaction_data.copy()
        transaction_data = prepare_transaction(
            transaction_dict,
            getattr(model, "feature_names_in_", None),
        )

        # Make prediction
        prediction = model.predict(transaction_data)[0]

        # Get probabilities
        probability = model.predict_proba(transaction_data)[0]

        fraud_probability = float(probability[1])
        genuine_probability = float(probability[0])

        # Convert DataFrame back to dictionary for explainability
        prepared_transaction = transaction_data.iloc[0].to_dict()

        # Generate risk information
        risk_result = RiskEngine.calculate_risk(
            fraud_probability
        )

        # Generate AI explanation
        explanation = ExplainabilityEngine.generate_explanation(
            prepared_transaction
        )

        return {
            "prediction": int(prediction),
            "fraud_probability": fraud_probability,
            "genuine_probability": genuine_probability,
            "risk_score": risk_result["risk_score"],
            "severity": risk_result["severity"],
            "recommendation": risk_result["recommendation"],
            "explanation": explanation
        }