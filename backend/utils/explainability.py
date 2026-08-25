class ExplainabilityEngine:

    @staticmethod
    def generate_explanation(transaction):
        """
        Generate human-readable reasons for the transaction risk.
        """

        explanations = []

        # Amount anomaly
        if transaction.get("scaled_amount", 0) > 2:
            explanations.append(
                "The transaction amount is unusually high."
            )

        # Time anomaly
        if transaction.get("scaled_time", 0) > 2:
            explanations.append(
                "The transaction occurred at an unusual time."
            )

        # Important fraud-related PCA features
        if transaction.get("V14", 0) < -2:
            explanations.append(
                "V14 shows an abnormal transaction pattern."
            )

        if transaction.get("V17", 0) < -2:
            explanations.append(
                "V17 shows an abnormal transaction pattern."
            )

        if transaction.get("V12", 0) < -2:
            explanations.append(
                "V12 indicates suspicious transaction behaviour."
            )

        if transaction.get("V10", 0) < -2:
            explanations.append(
                "V10 indicates an unusual transaction pattern."
            )

        # Normal transaction
        if not explanations:
            explanations.append(
                "No significant transaction anomalies were detected."
            )

        return explanations