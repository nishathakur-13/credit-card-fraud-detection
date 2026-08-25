class RiskEngine:

    @staticmethod
    def calculate_risk(fraud_probability):
        """
        Convert fraud probability into a risk score,
        severity level, and recommendation.
        """

        probability = float(fraud_probability)

        # Risk score: 0-100
        risk_score = round(probability * 100)

        # Risk classification
        if probability >= 0.80:
            severity = "High"
            recommendation = (
                "Block the transaction and investigate immediately."
            )

        elif probability >= 0.50:
            severity = "Medium"
            recommendation = (
                "Transaction is suspicious. Additional verification is recommended."
            )

        else:
            severity = "Low"
            recommendation = (
                "Transaction appears safe."
            )

        return {
            "risk_score": risk_score,
            "severity": severity,
            "recommendation": recommendation
        }