"""Prediction and health endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.model import predictor
from app.schemas.input_schema import HealthResponse, PredictionRequest, PredictionResponse
from app.services.preprocessing import request_to_features
from app.utils.config import FeatureOrderError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(payload: PredictionRequest) -> PredictionResponse:
    """Validate input, run inference, and return the model output."""
    try:
        features = request_to_features(payload)
    except FeatureOrderError as exc:
        logger.warning("Invalid feature payload: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    try:
        result = predictor.predict(features)
    except FileNotFoundError as exc:
        logger.error("Model file missing: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    except RuntimeError as exc:
        logger.error("Model not loaded: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    except ValueError as exc:
        logger.error("Model inference failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    return PredictionResponse(prediction=result.prediction, confidence=result.confidence)


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Lightweight readiness probe."""
    try:
        predictor.ensure_model_loaded()
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model file missing",
        )

    return HealthResponse(status="ok")


