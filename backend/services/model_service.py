import os
import joblib
from pathlib import Path


class ModelService:
    _model = None

    @classmethod
    def load_model(cls):
        """
        Load the trained fraud detection model only once.
        """
        if cls._model is None:
            project_root = Path(__file__).resolve().parents[2]
            model_path = project_root / "models" / "fraud_detector.pkl"

            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model not found at: {model_path}"
                )

            cls._model = joblib.load(model_path)
            print("✅ AI Model Loaded Successfully")

        return cls._model