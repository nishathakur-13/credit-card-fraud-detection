from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from backend.services.prediction_service import PredictionService


app = FastAPI(
    title="FraudShield AI API",
    description="AI-powered credit card fraud detection API",
    version="1.0.0"
)


class TransactionRequest(BaseModel):
    transaction: Dict[str, Any]


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "FraudShield AI",
        "message": "Fraud detection API is running."
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict_transaction(request: TransactionRequest):
    try:
        result = PredictionService.predict(request.transaction)

        return {
            "success": True,
            "result": result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )